""" A tiled image renderer with zoom & pan support based on OpenGL. """

import os                      # built-in library
import time                    # built-in library
import struct                  # built-in library
import threading               # built-in library
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
        self.texture_filter = None  # filter_nearest' or filter_linear
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
        self.prog['scale'].value = 1.0
        self.prog['orientation'].value = 0
        self.prog['mousepos'].value = (0.0, 0.0)
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
            texture.swizzle = 'RGB1'
            texture.use()
            orientation = self.files.orientations[imgidx]
            texw, texh = texture.width, texture.height
            texw, texh = (texh, texw) if orientation in [90, 270] else (texw, texh)
            _vpx, _vpy, vpw, vph = self.ui.viewports[i]
            self.ctx.viewport = self.ui.viewports[i]
            self.ctx.clear(*tile_colors[i], viewport=self.ctx.viewport)
            self.prog['mousepos'].value = tuple(self.ui.mousepos)
            self.prog['orientation'].value = orientation
            self.prog['aspect'].value = self._get_aspect_ratio(vpw, vph, texw, texh)
            self.prog['scale'].value = self.ui.scale
            self.prog['grayscale'].value = (texture.components == 1)
            self.prog['gamma'].value = self.ui.gamma
            self.prog['ev'].value = self.ui.ev
            self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.finish()
        elapsed = (time.time() - t0) * 1000
        interval = (time.time() - self.tprev) * 1000
        w, h = self.ui.window.get_size()
        self.tprev = time.time()
        self._vprint(f"rendering {w} x {h} pixels took {elapsed:.1f} ms, frame-to-frame interval was {interval:.1f} ms")
        return elapsed

    def _create_texture(self, img):
        # ModernGL texture dtypes that actually work:
        #   'f1': fixed-point [0, 1] internal format (GL_RGB8), uint8 input
        #   'f2': fixed-point [0, 1] internal format (GL_RGB16F), float16 input in [0, 1]
        #   'f4': fixed-point [0, 1] internal format (GL_RGB32F), float32 input in [0, 1]
        #
        # dtypes yielding constant zero in fragment shader (as of ModernGL 5.5.2):
        #   'u1': integer [0, 255] internal format (GL_RGB8UI), uint8 input
        #   'u2': integer [0, 65535] internal format (GL_RGB16UI), uint16 input
        #   'u4': integer [0, 2^32-1] internal format (GL_RGB32UI), uint32 input
        #
        h, w = img.shape[:2]
        dtype = f"f{img.itemsize}"  # uint8 => 'f1', float16 => 'f2', float32 => 'f4'
        components = img.shape[2] if img.ndim == 3 else 1  # RGB/RGBA/grayscale
        texture = self.ctx.texture((w, h), components, img.ravel(), dtype=dtype)
        return texture

    def _create_dummy_texture(self):
        dummy = self.ctx.texture((32, 32), 3, np.random.random((32, 32, 3)).astype(np.float32), dtype='f4')
        return dummy

    def _update_texture(self, texture, img):
        # TODO: take this into use
        texture.write(img.ravel())
        texture.build_mipmaps()
        return texture

    def _load_texture(self, idx):
        img = self.loader.get_image(idx)
        assert isinstance(img, (np.ndarray, str))
        if isinstance(img, np.ndarray):  # success
            texture = self._create_texture(img)
            self.files.textures[idx] = texture
            self.loader.release_image(idx)
        else:  # PENDING | INVALID | RELEASED
            texture = self.files.textures[idx]
            if texture is None:
                texture = self._create_dummy_texture()
                self.files.textures[idx] = texture
        if self.ui.texture_filter != "NEAREST":
            texture.build_mipmaps()
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

    def _vprint(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
