""" A tiled image renderer with zoom & pan support based on OpenGL. """

import os                      # built-in library
import time                    # built-in library
import struct                  # built-in library
import threading               # built-in library
import numpy as np             # pip install numpy
import scipy                   # pip install scipy
import moderngl                # pip install moderngl

try:
    # package mode
    from glview import ae
    from glview import texture
except ImportError:
    # stand-alone mode
    import ae
    import texture


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
        self.ae_gain_per_tile = np.ones(4)
        self.ae_converged = [True, True, True, True]

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
            gpu_texture = texture.texture
            gpu_texture.filter = self.filters[self.ui.texture_filter]
            if not texture.mipmaps_done:
                gpu_texture.filter = self.filters["NEAREST"]
                self._vprint(f"Texture #{imgidx} not fully uploaded yet, disabling mipmaps")
            gpu_texture.repeat_x = False
            gpu_texture.repeat_y = False
            gpu_texture.swizzle = 'RGB1'
            gpu_texture.use(location=0)
            orientation = self.files.orientations[imgidx]
            texw, texh = gpu_texture.width, gpu_texture.height
            texw, texh = (texh, texw) if orientation in [90, 270] else (texw, texh)
            scalex, scaley = self._get_aspect_ratio(vpw, vph, texw, texh)

            # Render the image into an offscreen texture representing the current
            # tile; note that all tiles are the same size, so we can use the same
            # texture for as long as window size and tiling scheme are unchanged.
            # The second rendering pass expects linear colors, so it's important
            # to remove sRGB gamma in the first pass.

            self.fbo.use()
            self.fbo.clear(*self._get_debug_tile_color(i))
            self.prog['img'] = 0
            self.prog['mousepos'] = tuple(self.ui.mousepos[i])
            self.prog['scale'] = self.ui.scale[i]
            self.prog['aspect'] = (scalex, scaley)
            self.prog['orientation'] = orientation
            self.prog['grayscale'] = (gpu_texture.components == 1)
            self.prog['degamma'] = self.files.linearize[imgidx]
            self.vao.render(moderngl.TRIANGLE_STRIP)

            # Derive exposure parameters for the current tile, to be applied in the
            # second rendering pass

            norm_maxvals = np.r_[1, texture.maxval, texture.maxval, texture.percentiles, texture.diffuse_white]
            norm_minvals = np.r_[0, 0, texture.minval, 0, 0, 0, 0, 0]
            whitelevel = norm_maxvals[self.ui.normalize]
            blacklevel = norm_minvals[self.ui.normalize]
            diffuse_white = texture.diffuse_white
            peak_white = texture.percentiles[0] / texture.diffuse_white
            peak_white = max(peak_white, 1.0)

            if self.ui.ae_per_tile[i]:
                ae_gain = ae.autoexposure(self.fbo.color_attachments[0], whitelevel, clip_pct=2.0)
                if ae_gain is not None:
                    if self.ui.ae_reset_per_tile[i]:
                        self.ae_gain_per_tile[i] = ae_gain
                        self.ui.ae_reset_per_tile[i] = False
                    else:
                        self.ae_gain_per_tile[i] = ae_gain * 0.1 + self.ae_gain_per_tile[i] * 0.9
                    self.ae_converged[i] = np.isclose(ae_gain, self.ae_gain_per_tile[i], rtol=0.01)
            else:
                self.ae_gain_per_tile[i] = 1.0
                self.ae_converged[i] = True

            # Render the current tile from an offscreen texture to the screen (or
            # the given render target), applying any screen-space postprocessing
            # effects on the fly, plus gamma as the final step. The input must be
            # in a linear color space.

            target.use()
            target.viewport = self.ui.viewports[i]
            target.clear(viewport=target.viewport)
            self.fbo.color_attachments[0].use(location=0)
            magnification = scalex * vpw / (gpu_texture.width / self.ui.scale[i])
            max_kernel_size = self.postprocess['kernel'].array_length
            kernel = self._sharpen(magnification)
            ae_gain = self.ae_gain_per_tile[i]
            self.postprocess['img'] = 0
            self.postprocess['mousepos'] = (0.0, 0.0)
            self.postprocess['scale'] = 1.0
            self.postprocess['aspect'] = (1.0, 1.0)
            self.postprocess['resolution'] = (vpw, vph)
            self.postprocess['magnification'] = magnification
            self.postprocess['mirror'] = self.ui.mirror_per_tile[i]
            self.postprocess['sharpen'] = self.ui.sharpen_per_tile[i]
            self.postprocess['kernel'] = np.resize(kernel, max_kernel_size)
            self.postprocess['kernw'] = kernel.shape[0]
            self.postprocess['minval'] = blacklevel
            self.postprocess['maxval'] = whitelevel
            self.postprocess['diffuse_white'] = diffuse_white
            self.postprocess['peak_white'] = peak_white
            self.postprocess['autoexpose'] = self.ui.ae_per_tile[i]
            self.postprocess['ae_gain'] = ae_gain
            self.postprocess['ev'] = self.ui.ev
            self.postprocess['cs_in'] = self.ui.cs_in
            self.postprocess['cs_out'] = self.ui.cs_out
            self.postprocess['tonemap'] = int(self.ui.tonemap_per_tile[i]) * 3
            self.postprocess['gamut.compress'] = self.ui.gamutmap_per_tile[i]
            self.postprocess['gamut.power'] = self.ui.gamut_pow
            self.postprocess['gamut.thr'] = self.ui.gamut_thr
            self.postprocess['gamut.scale'] = self._gamut(imgidx)
            self.postprocess['contrast'] = 0.25 if self.ui.tonemap_per_tile[i] else 0.0
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

    def upload_texture(self, idx: int, piecewise: bool) -> texture.Texture:
        """
        Upload the image at the given index to GPU memory, either all at once
        or piecewise, to avoid freezing the user interface. Create a new texture
        object on the GPU if necessary, otherwise use an existing one.
        """
        tex = self.files.textures[idx]  # None | Texture
        img = self.loader.get_image(idx)  # <ndarray> | PENDING | RELEASED | ...
        if not tex:
            img = img if isinstance(img, np.ndarray) else None
            tex = texture.Texture(self.ctx, img, idx, self.verbose)
            self.files.textures[idx] = tex
        elif isinstance(img, np.ndarray):
            tex.reuse(img)
        if isinstance(img, np.ndarray):
            self.loader.release_image(idx)
        tex.upload(piecewise)
        return tex

    def _sharpen(self, magnification: float) -> np.ndarray:
        """
        Generate an unsharp masking kernel optimized for the given magnification.
        Magnification is defined as the ratio of screen pixels to image pixels, so
        >1.0 means zooming in and <1.0 means zooming out.

        When zooming in, the "sigma" parameter is smoothly decreased towards 0.5,
        to avoid excessively wide halos around edges. The "strength" parameter is
        similarly decreased towards zero, to make the halos less dark.

        :param magnification: ratio of screen pixels to image pixels
        :returns: the generated unsharp masking kernel
        """
        def smoothstep(x, minn, maxx):  # [minn, maxx] => [0, 1]
            t = np.clip((x - minn) / (maxx - minn), 0, 1)
            y = t * t * (3 - 2 * t)
            return y

        sigma = smoothstep(magnification, 0.5, 2.0)  # [0.5, 2.0] => [0, 1]
        sigma = -sigma * 0.25 + 0.75  # [0, 1] => [0, -0.25] => [0.75, 0.5]
        strength = np.clip(1 / magnification, 0.001, 0.5)
        kernel = self._sharpen_kernel(sigma, strength)
        return kernel

    def _sharpen_kernel(self, sigma: float, strength: float) -> np.ndarray:
        """
        Generate an unsharp masking kernel with the given Gaussian blur sigma and
        sharpening strength. The sigma must be at least 0.25 to have any effect,
        and at most 4.0 to keep the kernel size within the current limits defined
        in the postprocessing shader (25 x 25 pixels). For good results, a sigma
        in [0.5, 1] is recommended.

        :param sigma: Gaussian blur standard deviation; range = [0.25, 4]
        :param strength: sharpening strength; range = [0, 1]
        :returns: the generated unsharp masking kernel
        """
        assert 0.0 <= strength <= 1.0, strength
        assert 0.25 <= sigma <= 4.0, sigma
        strength = 0.6 + strength * (0.95 - 0.6)  # [0, 1] => [0.6, 0.95]
        k = int(2 * np.ceil(3.0 * sigma) + 1)  # [3, 25]
        center = k // 2
        kernel = scipy.signal.windows.gaussian(k, std=sigma)
        kernel = np.outer(kernel, kernel)
        kernel = kernel / np.sum(kernel, keepdims=True)
        kernel = -strength * kernel
        kernel[center, center] += 1.0  # U = I - S * G
        kernel = kernel / np.sum(kernel, keepdims=True)
        return kernel

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
