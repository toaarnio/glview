""" A tiled image renderer with zoom & pan support based on OpenGL. """

import os                      # built-in library
import time                    # built-in library
import struct                  # built-in library
import threading               # built-in library
import types                   # built-in library
import numpy as np             # pip install numpy
import scipy                   # pip install scipy
import moderngl                # pip install moderngl


class GLRenderer:
    """ A tiled image renderer with zoom & pan support based on OpenGL. """

    filter_nearest = (moderngl.NEAREST, moderngl.NEAREST)
    filter_linear = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    filters = {"LINEAR": filter_linear, "NEAREST": filter_nearest}  # noqa: RUF012
    tile_debug_colors = (0xE0BBE4, 0x957DAD, 0xD291BC, 0xFEC8D8)  # pastel shades
    tile_normal_colors = (0, 0, 0, 0)

    def __init__(self, ui, files, loader, verbose=False):
        """
        Create a new GLRenderer with the given (hardcoded) PygletUI, FileList, and
        ImageProvider instances.
        """
        self.thread_name = "RenderThread"
        self.verbose = verbose
        self.ui = ui                # <PygletUI> State variables controlled by user
        self.loader = loader        # <ImageProvider> Still image loader
        self.files = files          # <FileList> Image files + metadata
        self.ctx = None             # <Context> OpenGL rendering context
        self.fbo = None             # <Framebuffer> Off-screen framebuffer
        self.prog = None            # <Program> image renderer with zoom & pan
        self.vbo = None             # <Buffer> xy vertex coords for 2D rectangle
        self.vao = None             # <VertexArray> 2D rectangle vertices + shader
        self.postprocess = None     # <Program> screen-space postprocessing shader
        self.vao_post = None        # <VertexArray> 2D rectangle vertices + shader
        self.texture_filter = None  # filter_nearest or filter_linear
        self.running = None
        self.render_thread = None
        self.tprev = None
        self.tile_colors = self.tile_debug_colors if verbose else self.tile_normal_colors
        self.fps = np.zeros(20)

    def init(self):
        """ Initialize an OpenGL context and attach it to an existing window. """
        # OpenGL window must already exist and be owned by this thread
        self._vprint("attaching to native OpenGL window...")
        self.ctx = moderngl.create_context(require=310)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._vprint("compiling shaders...")
        # Initialize the main shader
        shader_path = os.path.dirname(os.path.realpath(__file__))
        vshader = open(os.path.join(shader_path, "panzoom.vs"), encoding="utf-8").read()
        fshader = open(os.path.join(shader_path, "texture.fs"), encoding="utf-8").read()
        self.prog = self.ctx.program(vertex_shader=vshader, fragment_shader=fshader)
        self.vbo = self.ctx.buffer(struct.pack('8f', -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0))
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "vert")])
        # Initialize the screen-space postprocessing shader
        fshader = open(os.path.join(shader_path, "postprocess.fs"), encoding="utf-8").read()
        self.postprocess = self.ctx.program(vertex_shader=vshader, fragment_shader=fshader)
        self.vao_post = self.ctx.vertex_array(self.postprocess, [(self.vbo, "2f", "vert")])
        self.tprev = time.time()
        _ = self.ctx.error  # clear the GL error flag (workaround for a bug that prevents interoperability with Pyglet)


    def redraw(self, target: moderngl.Framebuffer | None = None):  # noqa: PLR0915
        """ Redraw the tiled image view with refreshed pan & zoom, filtering, etc. """
        t0 = time.time()
        target = target or self.ctx.screen
        w, h = self.ui.window.get_size()
        vpw, vph = self.ui.viewports[0][2:]

        if not self.fbo or self.fbo.size != (vpw, vph):
            offscreen_tile = self.ctx.texture((vpw, vph), components=3, dtype="f4")
            offscreen_tile.repeat_x = False
            offscreen_tile.repeat_y = False
            offscreen_tile.filter = self.filters["NEAREST"]
            self.fbo = self.ctx.framebuffer([offscreen_tile])

        for i in range(self.ui.numtiles):
            imgidx = self.ui.img_per_tile[i]
            texture = self.upload_texture(imgidx, piecewise=False)
            texture.filter = self.filters[self.ui.texture_filter]
            if not texture.extra.mipmaps_done:
                texture.filter = self.filters["NEAREST"]
                self._vprint(f"Texture #{imgidx} not fully uploaded yet, disabling mipmaps")
            texture.repeat_x = False
            texture.repeat_y = False
            texture.swizzle = 'RGB1'
            texture.use(location=0)
            orientation = self.files.orientations[imgidx]
            texw, texh = texture.width, texture.height
            texw, texh = (texh, texw) if orientation in [90, 270] else (texw, texh)
            maxval = texture.extra.maxval
            minval = texture.extra.minval
            meanval = texture.extra.meanval
            percentiles = texture.extra.percentiles  # [99.5, 98, 95, 90]
            norm_maxvals = np.r_[1, maxval, maxval, percentiles, meanval / 0.18]
            norm_minvals = np.r_[0, 0, minval, 0, 0, 0, 0, 0]

            # Render the image into an offscreen texture representing the current
            # tile; note that all tiles are the same size, so we can use the same
            # texture for as long as window size and tiling scheme are unchanged

            self.fbo.use()
            self.fbo.clear(*self._get_debug_tile_color(i))
            self.prog['texture'] = 0
            self.prog['mousepos'] = tuple(self.ui.mousepos[i])
            self.prog['scale'] = self.ui.scale[i]
            self.prog['aspect'] = self._get_aspect_ratio(vpw, vph, texw, texh)
            self.prog['orientation'] = orientation
            self.prog['grayscale'] = (texture.components == 1)
            self.prog['degamma'] = self.files.linearize[imgidx]
            self.prog['tonemap'] = self.ui.tonemap_per_tile[i]
            self.prog['gtm_ymax'] = self.ui.gtm_ymax
            self.prog['cs_in'] = self.ui.cs_in
            self.prog['cs_out'] = self.ui.cs_out
            self.prog['maxval'] = norm_maxvals[self.ui.normalize]
            self.prog['minval'] = norm_minvals[self.ui.normalize]
            self.prog['ev'] = self.ui.ev
            self.prog['gamut.compress'] = (self.ui.gamut_fit != 0)
            self.prog['gamut.power'] = self.ui.gamut_pow
            self.prog['gamut.thr'] = self.ui.gamut_thr
            self.prog['gamut.scale'] = self._gamut(imgidx)
            self.vao.render(moderngl.TRIANGLE_STRIP)

            # Render the current tile from an offscreen texture to the screen (or
            # the given render target), applying any screen-space postprocessing
            # effects on the fly, plus gamma as the final step

            target.use()
            target.viewport = self.ui.viewports[i]
            target.clear(viewport=target.viewport)
            self.fbo.color_attachments[0].use(location=0)
            magnification = vpw / (texture.width / self.ui.scale[i])
            sharpen_strength = np.clip(1.0 - magnification, 0.5, 0.9)
            max_kernel_size = self.postprocess['kernel'].array_length
            kernel, kernw = self._sharpen(sharpen_strength)
            kernel = np.resize(kernel, max_kernel_size)
            self.postprocess['texture'] = 0
            self.postprocess['mousepos'] = (0.0, 0.0)
            self.postprocess['scale'] = 1.0
            self.postprocess['aspect'] = (1.0, 1.0)
            self.postprocess['resolution'] = (vpw, vph)
            self.postprocess['magnification'] = magnification
            self.postprocess['sharpen'] = self.ui.sharpen_per_tile[i]
            self.postprocess['kernel'] = kernel
            self.postprocess['kernw'] = kernw
            self.postprocess['gamma'] = self.ui.gamma
            self.postprocess['debug'] = self.ui.debug_mode
            self.vao_post.render(moderngl.TRIANGLE_STRIP)

        self.ctx.finish()
        elapsed = (time.time() - t0) * 1000
        interval = (time.time() - self.tprev) * 1000
        self.fps[:-1] = self.fps[1:]
        self.fps[-1] = 1000 / interval
        self.tprev = time.time()
        self._vprint(f"rendering {w} x {h} pixels took {elapsed:.1f} ms, frame-to-frame interval was {interval:.1f} ms", log_level=2)
        return elapsed

    def upload_texture(self, idx, piecewise):
        """
        Upload the given image to GPU memory, either all at once or piecewise
        (100 rows per call). Progressive uploading helps avoid freezing the
        user interface, although short glitches may still occur with large
        images.
        """
        img = self.loader.get_image(idx)
        assert isinstance(img, np.ndarray | str), type(img)
        if isinstance(img, np.ndarray):
            texture = self._create_empty_texture(img)
            scale = 255 if img.dtype == np.uint8 else 1.0
            texture.extra.maxval = np.max(img[img >= 0.0], initial=0.0) / scale
            texture.extra.minval = np.min(img[img >= 0.0], initial=0.0) / scale
            texture.extra.idx = idx
            self._vprint(f"Created texture #{idx}, piecewise={piecewise}")
            nrows = 100 if piecewise else texture.height
            self._upload_texture_slice(texture, nrows)
            self.files.textures[idx] = texture
            self.loader.release_image(idx)
        elif self.files.textures[idx] is None:  # PENDING | INVALID | RELEASED
            texture = self._create_dummy_texture()
            self.files.textures[idx] = texture
        else:  # RELEASED
            texture = self.files.textures[idx]
            nrows = 100 if piecewise else texture.height
            self._upload_texture_slice(texture, nrows)
        return texture

    def screenshot(self, dtype=np.uint8):
        """
        Render the current on-screen view into an offscreen buffer and return the image
        as a NumPy array.
        """
        assert dtype in [np.uint8, np.float32], dtype
        t0 = time.time()
        dt = "f4" if dtype == np.float32 else "f1"
        w, h = self.ui.window.get_size()
        fbo = self.ctx.simple_framebuffer((w, h), components=3, dtype=dt)
        gamma = self.ui.gamma
        self.ui.gamma = (dtype == np.uint8)  # float32 => linear RGB
        self.redraw(fbo)
        self.ui.gamma = gamma
        self.ctx.screen.use()
        screenshot = fbo.read(components=3, dtype=dt, clamp=False)
        screenshot = np.frombuffer(screenshot, dtype=dtype)
        screenshot = screenshot.reshape(h, w, 3)
        screenshot = np.ascontiguousarray(screenshot[::-1])
        elapsed = (time.time() - t0) * 1000
        self._vprint(f"Taking a screenshot took {elapsed:.1f} ms")
        return screenshot

    def _sharpen(self, strength):
        """
        Generate an unsharp masking kernel with the given strength in [0, 1]. Return the
        kernel and its width in pixels.
        """
        min_sigma = 1.0
        max_sigma = 2.0
        sigma = min_sigma + strength * max_sigma  # [0, 1] => [1, 3]
        strength = 0.25 + strength * (0.75 - 0.25)  # [0, 1] => [0.25, 0.75]
        k = 2 * int(3.0 * sigma) + 1  # [7, 19]
        center = k // 2
        kernel = scipy.signal.windows.gaussian(k, std=sigma)
        kernel = np.outer(kernel, kernel)
        kernel = kernel / np.sum(kernel, keepdims=True)
        kernel = -strength * kernel
        kernel[center, center] += 1.0  # U = I + S * G
        kernel = kernel / np.sum(kernel, keepdims=True)
        return kernel, k

    def _gamut(self, imgidx):
        """
        Calculate per-color-channel scale factors as required by the shader for gamut
        compression. The calculation is based on three user-defined parameters (power,
        limit, and threshold) that control the shape of the compression curve. Per-image
        control of the 'limit' parameter is supported, but not currently used.
        """
        if (gamut_lim := self.ui.gamut_lim) is None:  # use global limits by default
            gamut_lim = self.files.metadata[imgidx]['gamut_bounds']  # per-image limit
        gamut_lim = np.clip(gamut_lim, 1.01, np.inf)  # >1.01 to ensure no overflows
        scale = self._gamut_curve(self.ui.gamut_pow, self.ui.gamut_thr, gamut_lim)
        return scale

    def _gamut_curve(self, power, thr, lim):
        """
        Calculate a curve shaping scale factor for each color channel, based on the given
        power, threshold and limit.

        Arguments:
          - power: curve exponent; higher value = steeper curve
          - thr: percentage of core gamut to leave unmodified
          - lim: upper bound for values to compress into gamut

        Returns:
          - scale: compression curve global scale factor, required by the shader
        """

        assert np.all(thr <= 0.95), thr  # must be < 1.0 to avoid infs and nans
        assert np.all(lim >= 1.01), lim  # must be > 1.0 to avoid infs and nans
        assert np.all(thr >= 0.0), thr  # not strictly necessary, but helps range analysis
        assert np.all(lim <= 3.0), lim  # not an exact bound, but safe at float32
        assert np.all(power >= 1.0), power  # not an exact bound, but safe at float32
        assert np.all(power <= 20.0), power  # not an exact bound, but safe at float32

        # range analysis for inputs yielding the smallest scale (~0.05):
        #   thr = 0.95
        #   lim = 3.0
        #   pow = 1.0
        #   src_domain = 3.0 - 0.95 = 2.05
        #   dst_domain = 1.0 - 0.95 = 0.05
        #   rel_domain = 2.05 / 0.05 = 2.05 * 20 = 41
        #   pow_domain = 41 ^ 1 = 41
        #   ipow_domain = (41 - 1) ^ (1 / 1) = 40
        #   scale = 2.05 / 40 = 0.05125 > 0.05 = dst_domain

        # range analysis for inputs yielding the largest intermediate value (~1e32):
        #   thr = 0.95
        #   lim = 3.0
        #   pow = 20.0
        #   src_domain = 2.05
        #   dst_domain = 0.05
        #   rel_domain = 2.05 / 0.05 = 41
        #   pow_domain = 41 ^ 20 < 1e33 < inf
        #   ipow_domain = (41 ^ 20 - 1) / (1 / 20) ~ 41
        #   scale = 2.05 / 41 = 0.05 = dst_domain

        # range analysis for inputs minimizing ipow_domain (0.2):
        #   thr = 0.95
        #   lim = 1.01
        #   pow = 1.0
        #   src_domain = 1.01 - 0.95 = 0.06
        #   dst_domain = 1.00 - 0.95 = 0.05
        #   rel_domain = 0.06 / 0.05 = 1.2
        #   pow_domain = 1.2 ^ 1 = 1.2
        #   ipow_domain = 1.2 - 1 = 0.2
        #   scale = 0.06 / 0.2 = 0.30 >> dst_domain

        invp = 1 / power  # [1/20, 1]
        src_domain = lim - thr  # range on the x axis to compress from; [0.06, 3.0]
        dst_domain = 1.0 - thr  # range on the x axis to compress to; [0.05, 1.0]
        rel_domain = src_domain / dst_domain  # [1.01, 41]
        pow_domain = rel_domain ** power  # max = 41^20 < 1e33 < inf
        ipow_domain = (pow_domain - 1) ** invp  # [0.01, 41]
        scale = src_domain / ipow_domain  # [dst_domain, 101 * dst_domain]
        assert np.all(1.01 * scale >= dst_domain), f"{1.01 * scale} vs. {dst_domain}"
        return scale

    def _create_dummy_texture(self):
        dummy = np.random.default_rng().random((32, 32, 3), dtype=np.float32)
        texture = self.ctx.texture((32, 32), 3, dummy, dtype='f4')
        texture.extra = types.SimpleNamespace()
        texture.extra.done = True
        texture.extra.upload_done = False
        texture.extra.mipmaps_done = False
        texture.extra.stats_done = False
        texture.extra.mipmaps_done = False
        texture.extra.dtype = np.float32
        texture.extra.components = 3
        texture.extra.img = None
        texture.extra.maxval = 1.0
        texture.extra.minval = 0.0
        texture.extra.meanval = 1.0
        texture.extra.percentiles = np.ones(4)
        return texture

    def _create_empty_texture(self, img):
        # ModernGL texture dtypes that actually work:
        #   'f1': fixed-point [0, 1] internal format (GL_RGB8), uint8 input
        #   'f2': float16 internal format (GL_RGB16F), float16 input
        #   'f4': float32 internal format (GL_RGB32F), float32 input
        #
        # dtypes yielding constant zero in fragment shader (as of ModernGL 5.7.4):
        #   'u1': integer [0, 255] internal format (GL_RGB8UI), uint8 input
        #   'u2': integer [0, 65535] internal format (GL_RGB16UI), uint16 input
        #   'u4': integer [0, 2^32-1] internal format (GL_RGB32UI), uint32 input
        #
        h, w = img.shape[:2]
        dtype = f"f{img.itemsize}"  # uint8 => 'f1', float16 => 'f2', float32 => 'f4'
        components = img.shape[2] if img.ndim == 3 else 1  # RGB/RGBA/grayscale
        texture = self.ctx.texture((w, h), components, data=None, dtype=dtype)
        texture.extra = types.SimpleNamespace()
        texture.extra.done = False
        texture.extra.upload_done = False
        texture.extra.mipmaps_done = False
        texture.extra.stats_done = False
        texture.extra.dtype = img.dtype
        texture.extra.components = components
        texture.extra.img = img
        texture.extra.rows_uploaded = 0
        texture.extra.maxval = 1.0
        texture.extra.minval = 0.0
        texture.extra.meanval = 1.0
        texture.extra.percentiles = np.ones(4)
        return texture

    def _upload_texture_slice(self, texture, nrows):
        t0 = time.time()
        if not texture.extra.done:
            if not texture.extra.upload_done:
                vpx = 0
                vpy = texture.extra.rows_uploaded
                vpw = texture.width
                vph = texture.height - vpy
                vph = min(vph, nrows)
                img = texture.extra.img
                texture.write(img[vpy:vpy+vph].ravel(), (vpx, vpy, vpw, vph))
                vpy += vph
                texture.extra.rows_uploaded = vpy
                if vpy >= texture.height:
                    texture.extra.upload_done = True
                    texture.extra.img = None
                    elapsed = (time.time() - t0) * 1000
                    self._vprint(f"Completed uploading texture #{texture.extra.idx}, took {elapsed:.1f} ms")
            elif not texture.extra.mipmaps_done:
                texture.build_mipmaps()
                texture.extra.mipmaps_done = True
                elapsed = (time.time() - t0) * 1000
                self._vprint(f"Generated mipmaps for texture #{texture.extra.idx}, took {elapsed:.1f} ms")
            elif not texture.extra.stats_done:
                self._texture_stats(texture)
                texture.extra.stats_done = True
                texture.extra.done = True
                elapsed = (time.time() - t0) * 1000
                self._vprint(f"Generated stats for texture #{texture.extra.idx}, took {elapsed:.1f} ms")

    def _texture_stats(self, texture):
        if texture.extra.mipmaps_done:
            mip_lvl = 3 if min(texture.size) >= 128 else 0
            stats = texture.read(level=int(mip_lvl))
            stats = np.frombuffer(stats, dtype=texture.extra.dtype)
            stats = stats.reshape(-1, texture.extra.components)
            if texture.extra.dtype == np.uint8:
                meanval = np.mean(np.max(stats, axis=-1)) / 255
                pct = np.percentile(np.max(stats, axis=-1), [99.5, 98, 95, 90]) / 255
            else:
                meanval = np.mean(np.max(stats, axis=-1))
                pct = np.percentile(np.max(stats, axis=-1), [99.5, 98, 95, 90])
            texture.extra.meanval = meanval
            texture.extra.percentiles = pct

    def _get_aspect_ratio(self, vpw, vph, texw, texh):
        viewport_aspect = vpw / vph
        texture_aspect = texw / texh
        if texture_aspect > viewport_aspect:
            # image wider than window => squeeze y => black top & bottom
            xscale, yscale = (1.0, viewport_aspect / texture_aspect)
        else:
            # image narrower than window => squeeze x => black sides
            xscale, yscale = (texture_aspect / viewport_aspect, 1.0)
        return xscale, yscale

    def _get_debug_tile_color(self, idx: int) -> np.ndarray:
        hex_to_rgb = lambda h: [h >> 16, (h >> 8) & 0xff, h & 0xff]
        tile_colors = [hex_to_rgb(hexrgb) for hexrgb in self.tile_colors]
        tile_colors = np.array(tile_colors) / 255.0
        return tile_colors[idx]

    def _vprint(self, message, log_level=1):
        if self.verbose >= log_level:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
