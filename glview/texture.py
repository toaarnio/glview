import moderngl
import numpy as np


class Texture:
    def __init__(self, ctx: moderngl.Context, img: np.ndarray = None, idx: int = -1):
        self.ctx = ctx
        self.texture = None
        self.img = img
        self.idx = idx
        self.dtype = img.dtype if img is not None else np.float32
        self.components = img.shape[2] if img is not None and img.ndim == 3 else 3
        self.rows_uploaded = 0
        self.minval = 0.0
        self.maxval = 1.0
        self.meanval = 0.5
        self.percentiles = np.ones(4)
        self.upload_done = False
        self.mipmaps_done = False
        self.stats_done = False
        if self.img is None:
            self._create_dummy()
        else:
            self._create_from_image()
            self.precompute_stats()

    def _create_dummy(self):
        """Create a placeholder texture."""
        dummy_img = np.random.default_rng().random((32, 32, 3)).astype(np.uint8)
        self.texture = self.ctx.texture((32, 32), 3, dummy_img, dtype='f1')
        self.upload_done = True
        self.mipmaps_done = True
        self.stats_done = True

    def _create_from_image(self):
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

    def precompute_stats(self):
        """ Pre-compute basic stats from a downsampled version of the image. """
        if self.img is None:
            return
        scale = 255.0 if self.img.dtype == np.uint8 else 1.0
        stats_img = self.img[::16, ::16]
        if stats_img.size > 0:
            pixel_max = np.max(stats_img, axis=-1) / scale
            self.minval = np.min(pixel_max)
            self.maxval = np.max(pixel_max)
            self.meanval = np.mean(pixel_max)
            self.percentiles = np.percentile(pixel_max, [99.5, 98, 95, 90])

    def upload_slice(self, nrows: int):
        """Uploads a slice of the image data to the GPU texture."""
        if self.upload_done or self.img is None:
            return
        vpy = self.rows_uploaded
        vph = min(self.img.shape[0] - vpy, nrows)
        if vph <= 0:
            self.upload_done = True
            self.img = None
            return
        rows = self.img[vpy : vpy + vph]
        self.texture.write(rows.ravel(), viewport=(0, vpy, self.texture.width, vph))
        self.rows_uploaded += vph
        if self.rows_uploaded >= self.texture.height:
            self.upload_done = True
            self.img = None

    def build_mipmaps(self):
        """Generate mipmaps for the texture."""
        if self.mipmaps_done or not self.upload_done:
            return
        self.texture.build_mipmaps()
        self.mipmaps_done = True

    def compute_stats(self):
        """Calculate statistics from a low-resolution mipmap level."""
        if self.stats_done or not self.mipmaps_done:
            return
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

    @property
    def done(self) -> bool:
        return self.upload_done and self.mipmaps_done and self.stats_done

    def release(self):
        if self.texture:
            self.texture.release()
