import time
import threading
import moderngl
import numpy as np

import glview._io  # noqa: F401 (ensures builtins.print is patched on Windows)


def _build_degamma_lut(n: int, out_dtype) -> np.ndarray:
    """Build an n-entry sRGB inverse gamma LUT (uint index -> linear float)."""
    lut = np.linspace(0.0, 1.0, n, dtype=np.float32)
    srgb_lo = lut / 12.92
    srgb_hi = ((lut + 0.055) / 1.055) ** 2.4
    return np.where(lut > 0.04045, srgb_hi, srgb_lo).astype(out_dtype)


_DEGAMMA_LUT = {
    np.dtype("uint8"):  _build_degamma_lut(256, np.float16),
    np.dtype("uint16"): _build_degamma_lut(65536, np.float32),
}


def _linearized_dtype(img_dtype: np.dtype) -> np.dtype:
    """Return the output dtype after degamma for the given input dtype, or the dtype unchanged."""
    lut = _DEGAMMA_LUT.get(img_dtype)
    return np.dtype(lut.dtype) if lut is not None else img_dtype


class Texture:

    def __init__(self, ctx: moderngl.Context, img: np.ndarray, idx: int, verbose: bool, linearize: bool = False, rawmax: int = 65535):
        self.ctx = ctx
        self.img = img
        self.idx = idx
        self.verbose = verbose
        self.linearize = linearize
        self.rawmax = rawmax
        self._img_dtype = img.dtype if img is not None else np.dtype(np.float32)
        self.texture = None
        self.dtype = np.float32
        self.components = 3
        self.minval = 0.0
        self.maxval = 1.0
        self.meanval = 0.5
        self.diffuse_white = 1.0
        self.percentiles = np.ones(4)
        self.upload_done = False
        self.mipmaps_done = False
        self.stats_done = False
        self._rows_uploaded = 0
        if self.img is not None:
            self.create_empty()
            self.compute_stats()
        else:
            self.create_dummy()

    @property
    def done(self) -> bool:
        return self.upload_done and self.mipmaps_done and self.stats_done

    @property
    def bitdepth_scale(self) -> float:
        """
        Scale factor to bring nu2-normalized uint16 values into [0, 1] in texture.fs,
        where 1.0 corresponds to the bit-depth maximum (e.g. 1023 for 10-bit, 4095 for
        12-bit, 65535 for full 16-bit). For all other dtypes this is 1.0 (no-op).
        """
        if self._img_dtype == np.uint16 and not self.linearize:
            return 65535.0 / self.rawmax
        return 1.0

    def create_dummy(self):
        """Create a placeholder texture."""
        dummy_img = np.random.default_rng().integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        self.texture = self.ctx.texture((32, 32), 3, dummy_img, dtype='f1')
        self.upload_done = True
        self.mipmaps_done = True
        self.stats_done = True

    def create_empty(self):
        """
        Create an empty texture based on image properties.

        ModernGL texture dtypes that actually work with sampler2D:
          'f1': fixed-point [0, 1] internal format (GL_RGB8), uint8 input
          'f2': float16 internal format (GL_RGB16F), float16 input
          'f4': float32 internal format (GL_RGB32F), float32 input
          'nu2': normalized uint16 [0, 1] (GL_RGB16), uint16 input — for linear data only

        'nu2' is NOT used for sRGB uint16 PNGs even though it could represent the values
        without CPU conversion: mipmap levels are constructed by the GPU averaging texels,
        and averaging gamma-encoded values produces incorrect results. Degamma must happen
        before upload so that all GPU-side filtering and mipmap construction operate in
        linear light. The LUT converts uint16 → float32 linear before each upload slice.

        dtypes yielding constant zero in fragment shader (require usampler2D instead):
          'u1': integer [0, 255] internal format (GL_RGB8UI), uint8 input
          'u2': integer [0, 65535] internal format (GL_RGB16UI), uint16 input
          'u4': integer [0, 2^32-1] internal format (GL_RGB32UI), uint32 input
        """
        self._refresh_image_properties()
        h, w = self.img.shape[:2]
        _gl_dtype = {np.dtype("uint8"): "f1", np.dtype("uint16"): "nu2",
                     np.dtype("float16"): "f2", np.dtype("float32"): "f4"}
        dtype = _gl_dtype[np.dtype(self.dtype)]
        self.texture = self.ctx.texture((w, h), self.components, data=None, dtype=dtype)

    def reuse(self, img: np.ndarray):
        """Reuse an existing GPU texture to store the given image, if possible."""
        self.img = img
        self._img_dtype = img.dtype
        self.upload_done = False
        self.mipmaps_done = False
        self.stats_done = False
        self._rows_uploaded = 0
        self.compute_stats()
        components = img.shape[2] if img.ndim == 3 else 1
        sizes_match = self.texture.size[::-1] == img.shape[:2]
        effective_dtype = _linearized_dtype(img.dtype) if self.linearize else img.dtype
        dtypes_match = self.dtype == effective_dtype
        nchans_match = self.components == components
        if not (sizes_match and dtypes_match and nchans_match):
            self.release()
            self.create_empty()

    def upload(self, piecewise: bool):
        """
        Upload image data to the GPU and generate mipmaps. Piecewise uploading
        (100 rows per call) helps avoid freezing the UI, but short glitches may
        still occur with large images.
        """
        t0 = time.time()
        if not self.upload_done:
            nrows = 100 if piecewise else self.texture.height
            self._upload_slice(nrows)
            if self.upload_done:
                elapsed = (time.time() - t0) * 1000
                self._vprint(f"Completed uploading texture #{self.idx}, took {elapsed:.1f} ms")
        self.build_mipmaps()

    def build_mipmaps(self):
        """Generate mipmaps for the texture."""
        if self.upload_done and not self.mipmaps_done:
            t0 = time.time()
            self.texture.build_mipmaps()
            self.mipmaps_done = True
            elapsed = (time.time() - t0) * 1000
            self._vprint(f"Generated mipmaps for texture #{self.idx}, took {elapsed:.1f} ms")

    def compute_stats(self):
        """ Compute statistics from a downsampled version of the image. """
        if self.img is not None and not self.stats_done:
            t0 = time.time()
            stats_img = self.img[::6, ::6]
            if self.linearize and (lut := _DEGAMMA_LUT.get(self.img.dtype)) is not None:
                stats_img = lut[stats_img]
            scale = 255.0 if stats_img.dtype == np.uint8 else \
                    65535.0 if stats_img.dtype == np.uint16 else 1.0
            if stats_img.size == 0:
                self.stats_done = True
                return
            pixel_max = np.max(stats_img, axis=-1) / scale
            pixel_max = np.clip(pixel_max, 0, None)
            pixel_max = pixel_max[pixel_max != 0.0]
            if pixel_max.size == 0:
                self.stats_done = True
                return
            self.minval = np.min(pixel_max)
            self.maxval = np.max(pixel_max)
            self.meanval = np.mean(pixel_max)
            self.percentiles = np.percentile(pixel_max, [99.5, 98, 95, 90])
            self.diffuse_white = self.estimate_diffuse_white(pixel_max)
            self.stats_done = True
            h, w = stats_img.shape[:2]
            elapsed = (time.time() - t0) * 1000
            self._vprint(f"Computed stats for texture #{self.idx} from {w * h} pixels, took {elapsed:.1f} ms")

    def estimate_diffuse_white(self, img: np.ndarray) -> float:
        """ Estimate diffuse white level using geometric mean over all non-zero pixels. """
        if np.max(img) == 1.0:  # already clipped to 1.0
            return 1.0
        img = img[img != 0.0]
        if img.size == 0:
            return 1.0
        img = img.astype(np.float32)  # float16 is not enough
        mean_level = np.exp(np.mean(np.log(img + 1e-6)))
        diffuse_white = mean_level / 0.18
        return diffuse_white

    def release(self):
        """Release the GPU memory associated with this Texture."""
        if self.texture:
            self.texture.release()

    def _upload_slice(self, nrows: int):
        """Upload a slice of image data to the GPU."""
        if not (self.upload_done or self.img is None):
            vpy = self._rows_uploaded
            vph = min(self.img.shape[0] - vpy, nrows)
            if vph <= 0:
                self.upload_done = True
                self.img = None
                return
            rows = self.img[vpy : vpy + vph]
            if self.linearize and (lut := _DEGAMMA_LUT.get(rows.dtype)) is not None:
                rows = lut[rows]
            self.texture.write(rows.ravel(), viewport=(0, vpy, self.texture.width, vph))
            self._rows_uploaded += vph
            if self._rows_uploaded >= self.texture.height:
                self.upload_done = True
                self.img = None

    def _vprint(self, message, log_level=1):
        if self.verbose >= log_level:
            print(f"[Texture/{threading.current_thread().name}] {message}")

    def _refresh_image_properties(self):
        """Update cached image metadata used when creating or reusing textures."""
        if self.img is None:
            self.dtype = np.float32
            self.components = 3
            return
        self.dtype = _linearized_dtype(self.img.dtype) if self.linearize else self.img.dtype
        self.components = self.img.shape[2] if self.img.ndim == 3 else 1
