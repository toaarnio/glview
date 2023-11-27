""" A tiled image renderer with zoom & pan support based on OpenGL. """

import os                      # built-in library
import time                    # built-in library
import struct                  # built-in library
import threading               # built-in library
import types                   # built-in library
import numpy as np             # pip install numpy
import moderngl                # pip install moderngl


class GLRenderer:
    """ A tiled image renderer with zoom & pan support based on OpenGL. """

    filter_nearest = (moderngl.NEAREST, moderngl.NEAREST)
    filter_linear = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    filters = {"LINEAR": filter_linear, "NEAREST": filter_nearest}
    tile_debug_colors = [0xE0BBE4, 0x957DAD, 0xD291BC, 0xFEC8D8]  # pastel shades
    tile_normal_colors = [0, 0, 0, 0]

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
        self.prog = None            # <Program> image renderer with zoom & pan
        self.vbo = None             # <Buffer> xy vertex coords for 2D rectangle
        self.vao = None             # <VertexArray> 2D rectangle vertices + shader
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
        shader_path = os.path.dirname(os.path.realpath(__file__))
        vshader = open(os.path.join(shader_path, "panzoom.vs"), encoding="utf-8").read()
        fshader = open(os.path.join(shader_path, "texture.fs"), encoding="utf-8").read()
        self.prog = self.ctx.program(vertex_shader=vshader, fragment_shader=fshader)
        self.prog['scale'] = 1.0
        self.prog['orientation'] = 0
        self.prog['mousepos'] = (0.0, 0.0)
        self.vbo = self.ctx.buffer(struct.pack('8f', -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0))
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "vert")])
        self.tprev = time.time()
        _ = self.ctx.error  # clear the GL error flag (workaround for a bug that prevents interoperability with Pyglet)

    def redraw(self):
        """ Redraw the tiled image view with refreshed pan & zoom, filtering, etc. """
        t0 = time.time()
        hex_to_rgb = lambda h: [h >> 16, (h >> 8) & 0xff, h & 0xff]
        tile_colors = [hex_to_rgb(hexrgb) for hexrgb in self.tile_colors]
        tile_colors = np.array(tile_colors) / 255.0
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
            _vpx, _vpy, vpw, vph = self.ui.viewports[i]
            maxval = texture.extra.maxval
            meanval = texture.extra.meanval
            percentiles = texture.extra.percentiles  # [99.5, 98, 95, 90]
            norm_choices = np.r_[1.0, maxval, percentiles, meanval / 0.18]
            self.ctx.viewport = self.ui.viewports[i]
            self.ctx.clear(*tile_colors[i], viewport=self.ctx.viewport)
            self.prog['texture'] = 0
            self.prog['mousepos'] = tuple(self.ui.mousepos)
            self.prog['orientation'] = orientation
            self.prog['aspect'] = self._get_aspect_ratio(vpw, vph, texw, texh)
            self.prog['scale'] = self.ui.scale
            self.prog['grayscale'] = (texture.components == 1)
            self.prog['gamma'] = self.ui.gamma
            self.prog['degamma'] = self.files.linearize[imgidx]
            self.prog['cs_in'] = self.ui.cs_in
            self.prog['cs_out'] = self.ui.cs_out
            self.prog['maxval'] = norm_choices[self.ui.normalize]
            self.prog['ev'] = self.ui.ev
            self.prog['gamut.compress'] = (self.ui.gamut_fit != 0)
            self.prog['gamut.power'] = self.ui.gamut_pow
            self.prog['gamut.thr'] = self.ui.gamut_thr
            self.prog['gamut.scale'] = self._gamut(imgidx)
            self.prog['debug'] = self.ui.debug_mode
            self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()
        elapsed = (time.time() - t0) * 1000
        interval = (time.time() - self.tprev) * 1000
        self.fps[:-1] = self.fps[1:]
        self.fps[-1] = 1000 / interval
        w, h = self.ui.window.get_size()
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
        assert isinstance(img, (np.ndarray, str)), type(img)
        if isinstance(img, np.ndarray):
            texture = self._create_empty_texture(img)
            scale = 255 if img.dtype == np.uint8 else 1.0
            texture.extra.maxval = np.max(img) / scale
            texture.extra.idx = idx
            self._vprint(f"Created texture #{idx}, piecewise={piecewise}")
            nrows = 100 if piecewise else texture.height
            self._upload_texture_slice(texture, nrows)
            self.files.textures[idx] = texture
            self.loader.release_image(idx)
        else:  # PENDING | INVALID | RELEASED
            if self.files.textures[idx] is None:
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
        dt = "f4" if dtype == np.float32 else "f1"
        w, h = self.ui.window.get_size()
        rbo = self.ctx.renderbuffer((w, h), components=3, dtype=dt)
        fbo = self.ctx.framebuffer([rbo])
        fbo.use()
        self.redraw()
        self.ctx.screen.use()
        screenshot = fbo.read(components=3, dtype=dt, clamp=False)
        screenshot = np.frombuffer(screenshot, dtype=dtype)
        screenshot = screenshot.reshape(h, w, 3)
        screenshot = np.ascontiguousarray(screenshot[::-1])
        return screenshot

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

    def _create_dummy_texture(self):
        texture = self.ctx.texture((32, 32), 3, np.random.random((32, 32, 3)).astype(np.float32), dtype='f4')
        texture.extra = types.SimpleNamespace()
        texture.extra.done = True
        texture.extra.mipmaps_done = False
        texture.extra.img = None
        texture.extra.maxval = 1.0
        texture.extra.meanval = 1.0
        texture.extra.percentiles = np.ones(4)
        return texture

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

    def _vprint(self, message, log_level=1):
        if self.verbose >= log_level:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
