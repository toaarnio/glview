""" A tiled image renderer with zoom & pan support based on OpenGL. """

import os                      # built-in library
import time                    # built-in library
import struct                  # built-in library
import threading               # built-in library
import numpy as np             # pip install numpy
import moderngl                # pip install moderngl
import PIL.Image


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
            texture = self._load_texture(imgidx)
            texture.filter = self.filters[self.ui.texture_filter]
            texture.repeat_x = False
            texture.repeat_y = False
            texture.swizzle = 'RGB1'
            texture.use(location=0)
            orientation = self.files.orientations[imgidx]
            texw, texh = texture.width, texture.height
            texw, texh = (texh, texw) if orientation in [90, 270] else (texw, texh)
            _vpx, _vpy, vpw, vph = self.ui.viewports[i]
            maxval = self.files.metadata[imgidx]['maxval']
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
            self.prog['maxval'] = maxval if self.ui.normalize else 1.0
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
        w, h = self.ui.window.get_size()
        self.tprev = time.time()
        self._vprint(f"rendering {w} x {h} pixels took {elapsed:.1f} ms, frame-to-frame interval was {interval:.1f} ms", log_level=2)
        return elapsed

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

    def _create_texture(self, img):
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
        texture = self.ctx.texture((w, h), components, img.ravel(), dtype=dtype)
        texture.build_mipmaps()
        mipmap = img
        for lvl in range(1, int(np.log2(max(w, h)))):
            #import rawpipe
            mipw = int(np.floor(w / (2 ** lvl)))
            miph = int(np.floor(h / (2 ** lvl)))
            mipmap = self._downsample(img, 2 ** lvl)
            #mipmap = rawpipe.verbose.downsample(mipmap)
            texture.write(mipmap.ravel(), level=lvl)
        """
        mipmaps = []
        for lvl in range(int(np.log2(max(w, h)))):
            mipw = int(np.floor(w / (2 ** lvl)))
            miph = int(np.floor(h / (2 ** lvl)))
            print(f"Retrieving mipmap level {lvl}: {mipw} x {miph} x {components}...")
            mipmap = texture.read(level=lvl)
            mipmap = np.frombuffer(mipmap, dtype=img.dtype)
            print(mipmap.dtype)
            mipmap = mipmap.reshape(miph, mipw, components)
            print(mipmap.shape)
            mipmaps.append(mipmap)
        """
        return texture

    def _downsample(self, img, factor):
        h, w = np.floor(np.asarray(img.shape[:2]) / factor).astype(int)
        self._vprint(f"PIL (Lanczos) downscaling {img.dtype} texture by factor {factor}: {img.shape} => ({h}, {w}, {img.shape[2]})")
        if img.ndim == 3:
            planar = np.moveaxis(img, -1, 0).astype(np.float32)
            channels = []
            for channel in planar:
                channel_pil = PIL.Image.fromarray(channel, mode="F")
                channel_pil = channel_pil.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                channels.append(np.asarray(channel_pil))
            downscaled = np.dstack(channels)
            downscaled = np.rint(downscaled).astype(img.dtype)
        else:
            downscaled = PIL.Image.fromarray(img.astype(np.float32), mode="F")
            downscaled = downscaled.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
            downscaled = np.rint(downscaled).astype(img.dtype)
        return downscaled

    def _create_dummy_texture(self):
        dummy = self.ctx.texture((32, 32), 3, np.random.random((32, 32, 3)).astype(np.float32), dtype='f4')
        return dummy

    def _load_texture(self, idx):
        img = self.loader.get_image(idx)
        assert isinstance(img, (np.ndarray, str)), type(img)
        if isinstance(img, np.ndarray):
            maxval = 1.0 if img.dtype == np.uint8 else np.max(img)
            texture = self._create_texture(img)
            self.files.metadata[idx] = {}
            self.files.metadata[idx]['maxval'] = maxval
            self.files.metadata[idx]['gamut_bounds'] = None
            self.files.textures[idx] = texture
            self.loader.release_image(idx)
        else:  # PENDING | INVALID | RELEASED
            if self.files.textures[idx] is None:
                texture = self._create_dummy_texture()
                maxval = 1.0
                self.files.metadata[idx] = {}
                self.files.metadata[idx]['maxval'] = maxval
                self.files.metadata[idx]['gamut_bounds'] = None
                self.files.textures[idx] = texture
            else:  # RELEASED
                texture = self.files.textures[idx]
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
