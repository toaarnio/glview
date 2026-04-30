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
    from glview import rendertargets
    from glview import rendertextures
except ImportError:
    # stand-alone mode
    import ae
    import rendertargets
    import rendertextures


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
        self.tile_target = None     # <TileRenderTarget> Off-screen per-tile framebuffer
        self.prog = None            # <Program> image renderer with zoom & pan
        self.vbo = None             # <Buffer> xy vertex coords for 2D rectangle
        self.vao = None             # <VertexArray> 2D rectangle vertices + shader
        self.postprocess = None     # <Program> screen-space postprocessing shader
        self.vao_post = None        # <VertexArray> 2D rectangle vertices + shader
        self.tprev = None
        self.tile_colors = self.tile_debug_colors if verbose else self.tile_normal_colors
        self.fps = np.zeros(20)
        self.ae_gain_per_tile = np.ones(4)
        self.ae_converged = [True, True, True, True]
        self.textures = None

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
        self.tile_target = rendertargets.TileRenderTarget(self.ctx, self.filters)
        self.textures = rendertextures.RenderTextureManager(self.ctx, self.files, self.loader, self.verbose)
        self.tprev = time.time()
        _ = self.ctx.error  # clear the GL error flag (workaround for a bug that prevents interoperability with Pyglet)

    def redraw(self, target: moderngl.Framebuffer | None = None, gamma_override: int | None = None):
        """ Redraw the tiled image view with refreshed pan & zoom, filtering, etc. """
        t0 = time.time()
        state = self.ui.state
        target = target or self.ctx.screen
        w, h = self.ui.window.get_size()
        vpw, vph = self.ui.viewports[0][2:]
        fbo = self.tile_target.ensure(vpw, vph)

        snapshot = self.files.snapshot()
        self.textures.prune(snapshot)
        for i in range(state.numtiles):
            imgidx = state.img_per_tile[i]
            texture = self.textures.upload(imgidx, piecewise=False, snapshot=snapshot)
            gpu_texture, orientation, scalex, scaley = self._prepare_tile_texture(
                imgidx=imgidx,
                texture=texture,
                snapshot=snapshot,
                vpw=vpw,
                vph=vph,
            )

            # Render the image into an offscreen texture representing the current
            # tile; note that all tiles are the same size, so we can use the same
            # texture for as long as window size and tiling scheme are unchanged.
            # The second rendering pass expects linear colors, so it's important
            # to remove sRGB gamma in the first pass.

            self._render_tile_scene(
                tileidx=i,
                imgidx=imgidx,
                gpu_texture=gpu_texture,
                orientation=orientation,
                scalex=scalex,
                scaley=scaley,
                snapshot=snapshot,
            )

            # Derive exposure parameters for the current tile, to be applied in the
            # second rendering pass

            whitelevel, blacklevel = self._normalization_levels(texture)

            ae_gain, tile_diffuse, tile_peak = ae.autoexposure(fbo.color_attachments[0], whitelevel, clip_pct=1.0)
            ae_gain, diffuse_white, peak_white = self._resolve_tile_exposure(
                tileidx=i,
                texture=texture,
                ae_gain=ae_gain,
                tile_diffuse=tile_diffuse,
                tile_peak=tile_peak,
            )

            # Render the current tile from an offscreen texture to the screen (or
            # the given render target), applying any screen-space postprocessing
            # effects on the fly, plus gamma as the final step. The input must be
            # in a linear color space.

            self._render_postprocess_tile(
                tileidx=i,
                imgidx=imgidx,
                target=target,
                gamma_override=gamma_override,
                vpw=vpw,
                vph=vph,
                gpu_texture=gpu_texture,
                scalex=scalex,
                whitelevel=whitelevel,
                blacklevel=blacklevel,
                diffuse_white=diffuse_white,
                peak_white=peak_white,
                ae_gain=ae_gain,
            )

        self.ctx.finish()
        elapsed = (time.time() - t0) * 1000
        interval = (time.time() - self.tprev) * 1000
        self.fps[:-1] = self.fps[1:]
        self.fps[-1] = 1000 / interval
        self.tprev = time.time()
        self._vprint(f"rendering {w} x {h} pixels took {elapsed:.1f} ms, frame-to-frame interval was {interval:.1f} ms", log_level=2)
        return elapsed

    def release(self):
        """
        Explicitly release all GL resources while the context is still alive.
        Must be called before the window closes to avoid errors from the GC
        trying to call glDeleteBuffers after the context is gone.
        """
        if self.textures is not None:
            self.textures.release_all()
        if self.tile_target is not None:
            self.tile_target.release()
        for obj in [self.vao_post, self.vao, self.vbo, self.postprocess, self.prog]:
            if obj is not None:
                obj.release()
        # Don't release self.ctx: moderngl attaches to pyglet's context, so
        # pyglet owns the context lifecycle. Releasing it here causes pyglet's
        # own cleanup to fail (glDeleteBuffers MissingFunctionException).

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
        gamma_override = int(dtype == np.uint8)  # float32 => linear RGB
        self.redraw(fbo, gamma_override=gamma_override)
        self.ctx.screen.use()
        screenshot = fbo.read(components=3, dtype=dt, clamp=False)
        screenshot = np.frombuffer(screenshot, dtype=dtype)
        screenshot = screenshot.reshape(h, w, 3)
        screenshot = np.ascontiguousarray(screenshot[::-1])
        elapsed = (time.time() - t0) * 1000
        self._vprint(f"Taking a screenshot took {elapsed:.1f} ms")
        return screenshot

    def _prepare_tile_texture(self, imgidx: int, texture, snapshot, vpw: int, vph: int):
        gpu_texture = texture.texture
        gpu_texture.filter = self.filters[self.ui.config.texture_filter]
        if not texture.mipmaps_done:
            gpu_texture.filter = self.filters["NEAREST"]
            self._vprint(f"Texture #{imgidx} not fully uploaded yet, disabling mipmaps")
        gpu_texture.repeat_x = False
        gpu_texture.repeat_y = False
        gpu_texture.swizzle = 'RGB1'
        gpu_texture.use(location=0)
        orientation = snapshot.orientations[imgidx]
        texw, texh = gpu_texture.width, gpu_texture.height
        texw, texh = (texh, texw) if orientation in [90, 270] else (texw, texh)
        scalex, scaley = self._get_aspect_ratio(vpw, vph, texw, texh)
        return gpu_texture, orientation, scalex, scaley

    def _render_tile_scene(self, tileidx: int, imgidx: int, gpu_texture, orientation: int, scalex: float, scaley: float, snapshot):
        fbo = self.tile_target.fbo
        fbo.use()
        fbo.clear(*self._get_debug_tile_color(tileidx))
        self.prog['img'] = 0
        self.prog['mousepos'] = tuple(self.ui.state.mousepos[tileidx])
        self.prog['scale'] = self.ui.state.scale[tileidx]
        self.prog['aspect'] = (scalex, scaley)
        self.prog['orientation'] = orientation
        self.prog['grayscale'] = (gpu_texture.components == 1)
        self.prog['degamma'] = snapshot.linearize[imgidx]
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _render_postprocess_tile(
        self,
        tileidx: int,
        imgidx: int,
        target,
        gamma_override: int | None,
        vpw: int,
        vph: int,
        gpu_texture,
        scalex: float,
        whitelevel: float,
        blacklevel: float,
        diffuse_white: float,
        peak_white: float,
        ae_gain: float,
    ):
        target.use()
        target.viewport = self.ui.viewports[tileidx]
        target.clear(viewport=target.viewport)
        self.tile_target.fbo.color_attachments[0].use(location=0)
        uniforms = self._build_postprocess_uniforms(
            tileidx=tileidx,
            imgidx=imgidx,
            gamma_override=gamma_override,
            vpw=vpw,
            vph=vph,
            gpu_texture=gpu_texture,
            scalex=scalex,
            whitelevel=whitelevel,
            blacklevel=blacklevel,
            diffuse_white=diffuse_white,
            peak_white=peak_white,
            ae_gain=ae_gain,
        )
        for key, value in uniforms.items():
            self.postprocess[key] = value
        self.vao_post.render(moderngl.TRIANGLE_STRIP)

    def _normalization_levels(self, texture_obj):
        norm_maxvals = np.r_[1, texture_obj.maxval, texture_obj.maxval, texture_obj.percentiles, texture_obj.diffuse_white]
        norm_minvals = np.r_[0, 0, texture_obj.minval, 0, 0, 0, 0, 0]
        return norm_maxvals[self.ui.config.normalize], norm_minvals[self.ui.config.normalize]

    def _resolve_tile_exposure(self, tileidx: int, texture, ae_gain, tile_diffuse, tile_peak):
        if self.ui.state.ae_per_tile[tileidx]:
            if ae_gain is not None:
                if self.ui.state.ae_reset_per_tile[tileidx]:
                    self.ae_gain_per_tile[tileidx] = ae_gain
                    self.ui.state.ae_reset_per_tile[tileidx] = False
                else:
                    self.ae_gain_per_tile[tileidx] = ae_gain * 0.1 + self.ae_gain_per_tile[tileidx] * 0.9
                self.ae_converged[tileidx] = np.isclose(ae_gain, self.ae_gain_per_tile[tileidx], rtol=0.01)
        else:
            self.ae_gain_per_tile[tileidx] = 1.0
            self.ae_converged[tileidx] = True

        if tile_peak is None:
            tile_peak = texture.maxval
            tile_diffuse = texture.diffuse_white
        peak_white = tile_peak / tile_diffuse
        diffuse_white = tile_diffuse
        return self.ae_gain_per_tile[tileidx], diffuse_white, peak_white

    def _build_postprocess_uniforms(
        self,
        tileidx: int,
        imgidx: int,
        gamma_override: int | None,
        vpw: int,
        vph: int,
        gpu_texture,
        scalex: float,
        whitelevel: float,
        blacklevel: float,
        diffuse_white: float,
        peak_white: float,
        ae_gain: float,
    ):
        magnification = scalex * vpw / (gpu_texture.width / self.ui.state.scale[tileidx])
        kernel = self._sharpen(magnification)
        max_kernel_size = self.postprocess['kernel'].array_length
        return {
            'img': 0,
            'mousepos': (0.0, 0.0),
            'scale': 1.0,
            'aspect': (1.0, 1.0),
            'resolution': (vpw, vph),
            'magnification': magnification,
            'mirror': self.ui.state.mirror_per_tile[tileidx],
            'sharpen': self.ui.state.sharpen_per_tile[tileidx],
            'kernel': np.resize(kernel, max_kernel_size),
            'kernw': kernel.shape[0],
            'minval': blacklevel,
            'maxval': whitelevel,
            'diffuse_white': diffuse_white,
            'peak_white': peak_white,
            'autoexpose': self.ui.state.ae_per_tile[tileidx],
            'ae_gain': ae_gain,
            'ev': self.ui.config.ev,
            'cs_in': self.ui.config.cs_in,
            'cs_out': self.ui.config.cs_out,
            'tonemap': int(self.ui.state.tonemap_per_tile[tileidx]) * 3,
            'gamut.compress': self.ui.state.gamutmap_per_tile[tileidx],
            'gamut.power': self.ui.config.gamut_pow,
            'gamut.thr': self.ui.config.gamut_thr,
            'gamut.scale': self._gamut(imgidx),
            'contrast': 0.25 if self.ui.state.tonemap_per_tile[tileidx] else 0.0,
            'gamma': self.ui.config.gamma if gamma_override is None else gamma_override,
            'debug': self.ui.config.debug_mode * int(self.ui.config.debug_mode_on),
        }

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
        if (gamut_lim := self.ui.config.gamut_lim) is None:  # use global limits by default
            snapshot = self.files.snapshot()
            gamut_lim = snapshot.metadata[imgidx]['gamut_bounds']  # per-image limit
        gamut_lim = np.clip(gamut_lim, 1.01, np.inf)  # >1.01 to ensure no overflows
        scale = self._gamut_curve(self.ui.config.gamut_pow, self.ui.config.gamut_thr, gamut_lim)
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
