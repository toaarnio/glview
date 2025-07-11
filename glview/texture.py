import time
import threading
import moderngl
import numpy as np


class Texture:

    def __init__(self, ctx: moderngl.Context, img: np.ndarray, idx: int, verbose: bool):
        self.ctx = ctx
        self.img = img
        self.idx = idx
        self.verbose = verbose
        self.texture = None
        self.dtype = img.dtype if img is not None else np.float32
        self.components = img.shape[2] if img is not None and img.ndim == 3 else 3
        self.minval = 0.0
        self.maxval = 1.0
        self.meanval = 0.5
        self.percentiles = np.ones(4)
        self.upload_done = False
        self.mipmaps_done = False
        self.stats_done = False
        self._rows_uploaded = 0
        if self.img is not None:
            self.create_empty()
            self.precompute_stats()
        else:
            self.create_dummy()

    @property
    def done(self) -> bool:
        return self.upload_done and self.mipmaps_done and self.stats_done

    def create_dummy(self):
        """Create a placeholder texture."""
        dummy_img = np.random.default_rng().random((32, 32, 3)).astype(np.uint8)
        self.texture = self.ctx.texture((32, 32), 3, dummy_img, dtype='f1')
        self.upload_done = True
        self.mipmaps_done = True
        self.stats_done = True

    def create_empty(self):
        """
        Create an empty texture based on image properties.

        ModernGL texture dtypes that actually work:
          'f1': fixed-point [0, 1] internal format (GL_RGB8), uint8 input
          'f2': float16 internal format (GL_RGB16F), float16 input
          'f4': float32 internal format (GL_RGB32F), float32 input

        dtypes yielding constant zero in fragment shader (as of ModernGL 5.7.4):
          'u1': integer [0, 255] internal format (GL_RGB8UI), uint8 input
          'u2': integer [0, 65535] internal format (GL_RGB16UI), uint16 input
          'u4': integer [0, 2^32-1] internal format (GL_RGB32UI), uint32 input
        """
        h, w = self.img.shape[:2]
        dtype = f"f{self.img.itemsize}"  # uint8 => 'f1', float16 => 'f2', float32 => 'f4'
        self.texture = self.ctx.texture((w, h), self.components, data=None, dtype=dtype)

    def reuse(self, img: np.ndarray):
        """Reuse an existing GPU texture to store the given image, if possible."""
        self.img = img
        self.upload_done = False
        self.mipmaps_done = False
        self.stats_done = False
        self._rows_uploaded = 0
        self.precompute_stats()
        sizes_match = self.texture.size[::-1] == img.shape[:2]
        dtypes_match = self.dtype == img.dtype
        nchans_match = self.components == (img.shape[2] if img.ndim == 3 else 1)
        if not (sizes_match and dtypes_match and nchans_match):
            self.release()
            self.create_empty()

    def upload(self, piecewise: bool):
        """
        Upload image data to the GPU, generate mipmaps, and calculate statistics.
        Piecewise uploading (100 rows per call) helps avoid freezing the UI, but
        short glitches may still occur with large images.
        """
        t0 = time.time()
        if not self.upload_done:
            nrows = 100 if piecewise else self.texture.height
            self._upload_slice(nrows)
            if self.upload_done:
                elapsed = (time.time() - t0) * 1000
                self._vprint(f"Completed uploading texture #{self.idx}, took {elapsed:.1f} ms")
        self.build_mipmaps()
        self.compute_stats()

    def build_mipmaps(self):
        """Generate mipmaps for the texture."""
        if self.upload_done and not self.mipmaps_done:
            t0 = time.time()
            self.texture.build_mipmaps()
            self.mipmaps_done = True
            elapsed = (time.time() - t0) * 1000
            self._vprint(f"Generated mipmaps for texture #{self.idx}, took {elapsed:.1f} ms")

    def compute_stats(self):
        """Calculate statistics from a low-resolution mipmap level."""
        if self.mipmaps_done and not self.stats_done:
            t0 = time.time()
            mip_lvl = 4 if min(self.texture.size) >= 128 else 0
            stats = self.texture.read(level=int(mip_lvl))
            stats = np.frombuffer(stats, dtype=self.dtype)
            stats = stats.reshape(-1, self.components)
            scale = 255.0 if self.dtype == np.uint8 else 1.0
            pixel_max = np.max(stats, axis=-1) / scale
            self.minval = np.min(pixel_max)
            self.maxval = np.max(pixel_max)
            self.meanval = np.mean(pixel_max)
            self.percentiles = np.percentile(pixel_max, [99.5, 98, 95, 90])
            self.stats_done = True
            p = stats.shape[0]
            elapsed = (time.time() - t0) * 1000
            self._vprint(f"Generated final stats for texture #{self.idx} from {p} pixels, took {elapsed:.1f} ms")

    def precompute_stats(self):
        """ Pre-compute preliminary stats from a downsampled version of the image. """
        if self.img is not None:
            t0 = time.time()
            scale = 255.0 if self.img.dtype == np.uint8 else 1.0
            stats_img = self.img[::16, ::16]
            if stats_img.size > 0:
                pixel_max = np.max(stats_img, axis=-1) / scale
                self.minval = np.min(pixel_max)
                self.maxval = np.max(pixel_max)
                self.meanval = np.mean(pixel_max)
                self.percentiles = np.percentile(pixel_max, [99.5, 98, 95, 90])
                h, w = stats_img.shape[:2]
                elapsed = (time.time() - t0) * 1000
                self._vprint(f"Generated preliminary stats for texture #{self.idx} from {w * h} pixels, took {elapsed:.1f} ms")

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
            self.texture.write(rows.ravel(), viewport=(0, vpy, self.texture.width, vph))
            self._rows_uploaded += vph
            if self._rows_uploaded >= self.texture.height:
                self.upload_done = True
                self.img = None

    def _vprint(self, message, log_level=1):
        if self.verbose >= log_level:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
