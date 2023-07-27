""" A graphical user interface for glview, based on Pyglet and ModernGL. """

import os                      # built-in library
import threading               # built-in library
import pprint                  # built-in library
import traceback               # built-in library
import pyglet                  # pip install pyglet
import piexif                  # pip install piexif
import numpy as np             # pip install numpy
import imsize                  # pip install imsize


class PygletUI:
    """ A graphical user interface for glview, based on Pyglet and ModernGL. """

    def __init__(self, files, debug, verbose=False):
        """ Create a new PygletUI with the given (hardcoded) FileList instance. """
        self.thread_name = "UIThread"
        self.debug_selected = debug  # selected debug rendering mode: 1|2|3|...
        self.debug_mode = 0  # start in normal mode, toggle on/off with space
        self.verbose = verbose
        self.files = files
        self.fullscreen = False
        self.numtiles = 1
        self.running = None
        self.need_redraw = True
        self.was_resized = True
        self.window = None
        self.key_state = None
        self.screensize = None
        self.winsize = None
        self.tileidx = 0
        self.scale = 1.0
        self.mousepos = np.zeros(2)  # always scaled & clipped to [0, 1] x [0, 1]
        self.mouse_speed = 4.0
        self.mouse_canvas_width = 1000
        self.keyboard_pan_speed = 100
        self.viewports = None
        self.ui_thread = None
        self.event_loop = None
        self.renderer = None
        self.texture_filter = "NEAREST"
        self.img_per_tile = [0, 1, 2, 3]
        self.gamma = False
        self.normalize = False
        self.ev_range = 2
        self.ev_linear = 0.0
        self.ev = 0.0
        self.gamut_fit = 0  # 0|1|2...
        self.gamut_lim = np.ones(3) * 1.1
        self.gamut_pow = np.ones(3) * 1.5
        self.gamut_thr = np.ones(3) * 0.8
        self.gamut_lin = 0.0

    def start(self, renderer):
        """ Start the UI thread. """
        self._vprint(f"spawning {self.thread_name}...")
        self.renderer = renderer
        self.running = True
        self.ui_thread = threading.Thread(target=lambda: self._try(self._pyglet_runner), name=self.thread_name)
        self.ui_thread.daemon = True  # terminate when main process ends
        self.ui_thread.start()

    def stop(self):
        """ Stop the UI thread. """
        self._vprint(f"killing {self.thread_name}...")
        self.running = False
        self.event_loop.has_exit = True
        self.ui_thread.join()
        self._vprint(f"{self.thread_name} killed")

    def _pyglet_runner(self):
        self._init_pyglet()
        self.renderer.init()
        self._vprint("starting Pyglet event loop...")
        self.event_loop = self._create_eventloop()
        self.event_loop.run()
        self._vprint("Pyglet event loop stopped")

    def _create_eventloop(self):
        parent = self

        class _EventLoop(pyglet.app.EventLoop):
            def idle(self):
                dt = self.clock.update_time()
                window = list(pyglet.app.windows)[0]
                parent.need_redraw |= dt > 0.9  # redraw at least once per second
                window.dispatch_event("on_draw")
                return 0.5  # call again after 0.5 seconds if no events until then

        return _EventLoop()

    def _init_pyglet(self):
        self._vprint("initializing Pyglet & native OpenGL...")
        pyglet.options['debug_lib'] = self.verbose
        pyglet.options['debug_gl'] = self.verbose
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        self.screensize = (screen.width, screen.height)
        self.winsize = (screen.width // 3, screen.height // 3)
        self.viewports = self._retile(self.numtiles, self.winsize)
        self.window = pyglet.window.Window(*self.winsize, resizable=True, vsync=True)
        self.window.set_caption(self._caption())
        self.window.set_fullscreen(self.fullscreen)
        self.window.set_mouse_visible(not self.fullscreen)
        self._setup_events()
        self.key_state = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.key_state)
        self._vprint("Pyglet & native OpenGL initialized")

    def _caption(self):
        fps = pyglet.clock.get_frequency()
        gamut = "off" if not self.gamut_fit else f"p = {self.gamut_pow[0]:.1f}"
        caption = f"glview [{self.ev:+1.2f}EV | gamut fit {gamut} | {fps:.1f} fps]"
        for tileidx in range(self.numtiles):
            imgidx = self.img_per_tile[tileidx]
            basename = self.files.filespecs[imgidx]
            caption = f"{caption} | {basename} [{imgidx+1}/{self.files.numfiles}]"
        return caption

    def _retile(self, numtiles, winsize):
        w, h = winsize
        viewports = {}
        if numtiles == 1:
            vpw, vph = (w, h)
            viewports[0] = (0, 0, vpw, vph)
        elif numtiles == 2:
            vpw, vph = (w // 2, h)
            viewports[0] = (0,   0, vpw, vph)
            viewports[1] = (vpw, 0, vpw, vph)
        elif numtiles == 3:
            vpw, vph = (w // 3, h)
            viewports[0] = (0,       0, vpw, vph)
            viewports[1] = (vpw,     0, vpw, vph)
            viewports[2] = (2 * vpw, 0, vpw, vph)
        elif numtiles == 4:
            vpw, vph = (w // 2, h // 2)
            viewports[0] = (0,   vph, vpw, vph)  # bottom left => top left
            viewports[1] = (vpw, vph, vpw, vph)  # bottom right => top right
            viewports[2] = (0,   0,   vpw, vph)  # top left => bottom left
            viewports[3] = (vpw, 0,   vpw, vph)  # top right => bottom right
        return viewports

    def _print_exif(self, filespec):
        try:
            exif_all = piexif.load(filespec)
            exif_tags = {tag: name for name, tag in piexif.ExifIFD.__dict__.items() if isinstance(tag, int)}
            image_tags = {tag: name for name, tag in piexif.ImageIFD.__dict__.items() if isinstance(tag, int)}
            exif_dict = {exif_tags[name]: val for name, val in exif_all.pop("Exif").items()}
            image_dict = {image_tags[name]: val for name, val in exif_all.pop("0th").items()}
            merged_dict = {**exif_dict, **image_dict}
            print(f"EXIF data for {filespec}:")
            pprint.pprint(merged_dict)
        except piexif.InvalidImageDataError as e:
            print(f"Failed to extract EXIF metadata from {filespec}: {e}")

    def _keyboard_zoom_pan(self):
        # this is invoked 50 times per second, so zooming/panning is pretty fast
        keys = pyglet.window.key
        ctrl_down = self.key_state[keys.LCTRL] or self.key_state[keys.RCTRL]
        if not ctrl_down:
            prev_scale = self.scale
            self.scale *= 1.0 + 0.1 * self.key_state[keys.PLUS]  # zoom in
            self.scale /= 1.0 + 0.1 * self.key_state[keys.MINUS]  # zoom out
            dx = self.key_state[keys.LEFT] - self.key_state[keys.RIGHT]
            dy = self.key_state[keys.DOWN] - self.key_state[keys.UP]
            dxdy = np.array((dx, dy))
            dxdy = dxdy * self.keyboard_pan_speed
            dxdy = dxdy / self.scale
            dxdy = dxdy / self.mouse_canvas_width
            self.mousepos = np.clip(self.mousepos + dxdy, -1.0, 1.0)
            if np.any(dxdy != 0.0) or self.scale != prev_scale:
                self.need_redraw = True

    def _triangle_wave(self, x, amplitude):
        # [0, 1] => [-amplitude, +amplitude]
        y = 4 * amplitude * np.abs((x - 0.25) % 1 - 0.5) - amplitude
        return y

    def _smooth_exposure(self):
        # this is invoked 50 times per second, so exposure control is pretty fast
        keys = pyglet.window.key
        self.ev_linear += 0.005 * self.key_state[keys.E]
        self.ev = self._triangle_wave(self.ev_linear, self.ev_range)
        self.need_redraw = True

    def _switch_gamut_curve(self):
        # cycle through a predefined selection of gamut compression modes:
        #  0 - off
        #  1 - steep curve, almost like clipping
        #  2 - shallow curve, strong desaturation
        presets = [None, (10.0, 1.1, 0.8), (3.0, 1.2, 0.8)]
        self.gamut_fit = (self.gamut_fit + 1) % len(presets)
        if (selection := presets[self.gamut_fit]) is not None:
            power, limit, threshold = selection
            self.gamut_pow = np.ones(3) * power
            self.gamut_lim = np.ones(3) * limit
            self.gamut_thr = np.ones(3) * threshold
            self._vprint(f"Gamut curve shape: pow = {power}, lim = {limit}, thr = {threshold}")
        else:
            self._vprint("Gamut compression off")

    def _setup_events(self):
        self._vprint("setting up Pyglet window event handlers...")

        @self.window.event
        def on_draw():
            self._keyboard_zoom_pan()
            self._smooth_exposure()
            if self.need_redraw:
                self.renderer.redraw()
                self.window.set_caption(self._caption())
                self.window.flip()
                if self.was_resized:
                    # ensure that both buffers (back & front) are filled
                    # with the same image after a resize event, or else
                    # the window may be left black until the next redraw
                    self.renderer.redraw()
                    self.window.flip()
                    self.was_resized = False
                self.need_redraw = False

        @self.window.event
        def on_resize(width, height):
            self.winsize = (width, height)
            self.viewports = self._retile(self.numtiles, self.winsize)
            self.need_redraw = True
            self.was_resized = True

        @self.window.event
        def on_mouse_press(_x, _y, button, _modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.window.set_mouse_visible(False)

        @self.window.event
        def on_mouse_release(_x, _y, button, _modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.window.set_mouse_visible(True)

        @self.window.event
        def on_mouse_drag(_x, _y, dx, dy, buttons, _modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                dxdy = np.array((dx, dy))
                dxdy = dxdy * self.mouse_speed
                dxdy = dxdy / self.scale
                dxdy = dxdy / self.mouse_canvas_width
                self.mousepos = np.clip(self.mousepos + dxdy, -1.0, 1.0)
                self.need_redraw = True

        @self.window.event
        def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
            scale_factor = 1.0 + 0.1 * scroll_y
            self.scale *= scale_factor
            self.need_redraw = True

        @self.window.event
        def on_close():
            self.running = False
            self.event_loop.has_exit = True

        @self.window.event
        def on_key_press(symbol, modifiers):
            keys = pyglet.window.key
            disallowed_keys = keys.MOD_CTRL | keys.MOD_ALT
            self._vprint(f"on_key_press({keys.symbol_string(symbol)}, modifiers={keys.modifiers_string(modifiers)})")
            if symbol == keys.C and modifiers == keys.MOD_CTRL:
                self.running = False
                self.event_loop.has_exit = True
            if (modifiers & disallowed_keys) == 0:  # ignore NumLock, ScrollLock, CapsLock, Shift
                if symbol in [keys.ESCAPE, keys.Q]:  # exit
                    self.running = False
                    self.event_loop.has_exit = True
                if symbol == keys.F:  # fullscreen
                    self.fullscreen = not self.fullscreen
                    self.window.set_fullscreen(self.fullscreen)
                    self.window.set_mouse_visible(not self.fullscreen)
                    self.need_redraw = True
                if symbol == keys.H:  # reset exposure + zoom & pan ("home")
                    self.scale = 1.0
                    self.mousepos = np.zeros(2)
                    self.ev_linear = 0.0
                    self.gamut_lin = 0.0
                    self.gamut_fit = 0
                    self.need_redraw = True
                if symbol == keys.G:  # gamma
                    self.gamma = not self.gamma
                    self.need_redraw = True
                if symbol == keys.B:  # toggle between narrow/wide (LDR/HDR) exposure control
                    self.ev_range = (self.ev_range + 6) % 12
                    self.need_redraw = True
                if symbol == keys.K: # cycle through gamut compression modes (off/hi/lo)
                    self._switch_gamut_curve()
                    self.need_redraw = True
                if symbol == keys.N:  # normalize
                    self.normalize = not self.normalize
                    self.need_redraw = True
                if symbol == keys.T:  # texture filtering
                    self.texture_filter = "LINEAR" if self.texture_filter == "NEAREST" else "NEAREST"
                    self.need_redraw = True
                if symbol == keys.S:  # split
                    self.numtiles = (self.numtiles % 4) + 1
                    self.tileidx = min(self.tileidx, self.numtiles - 1)
                    self.img_per_tile = np.clip(self.img_per_tile, 0, self.files.numfiles - 1)
                    self.viewports = self._retile(self.numtiles, self.winsize)
                    self.window.set_caption(self._caption())
                    self.need_redraw = True
                if symbol == keys.R:  # rotate
                    for imgidx in self.img_per_tile[:self.numtiles]:
                        self.files.orientations[imgidx] += 90
                        self.files.orientations[imgidx] %= 360
                    self.need_redraw = True
                if symbol == keys.I:  # image info
                    imgidx = self.img_per_tile[self.tileidx]
                    filespec = self.files.filespecs[imgidx]
                    fileinfo = imsize.read(filespec)
                    print(fileinfo)
                    self._print_exif(filespec)
                if symbol == keys.SPACE:  # toggle debug mode on/off
                    N = self.debug_selected
                    self.debug_mode = (self.debug_mode + N) % (N * 2)
                    self._vprint(f"debug rendering mode {self.debug_mode}")
                    self.need_redraw = True
                if symbol in [keys.D, keys.DELETE]:
                    if symbol == keys.D:  # drop
                        indices = self.img_per_tile[:self.numtiles]
                        self.files.drop(indices)
                    if symbol == keys.DELETE:  # delete
                        if self.numtiles == 1:  # only in single-tile mode
                            imgidx = self.img_per_tile[self.tileidx]
                            self.files.delete(imgidx)
                    if self.files.numfiles == 0:
                        self.running = False
                        self.event_loop.has_exit = True
                    else:
                        N = self.numtiles
                        visible_images = np.asarray(self.img_per_tile[:N]) - N
                        self.img_per_tile[:N] = visible_images % self.files.numfiles
                        self.window.set_caption(self._caption())
                        self.need_redraw = True
                # pylint: disable=protected-access
                if symbol in [keys._1, keys._2, keys._3, keys._4]:
                    tileidx = symbol - keys._1
                    self.tileidx = tileidx if tileidx < self.numtiles else self.tileidx
                    self.need_redraw = True

        @self.window.event
        def on_text_motion(motion):
            keys = pyglet.window.key
            if motion in [keys.MOTION_NEXT_PAGE, keys.MOTION_PREVIOUS_PAGE]:
                # PageUp / PageDown
                self._vprint(f"on_text_motion({keys.symbol_string(motion)})")
                incr = 1 if motion == keys.MOTION_NEXT_PAGE else -1
                imgidx = self.img_per_tile[self.tileidx]
                imgidx = (imgidx + incr) % self.files.numfiles
                self.img_per_tile[self.tileidx] = imgidx
                self.window.set_caption(self._caption())
                self.need_redraw = True
            if motion in [keys.MOTION_NEXT_WORD, keys.MOTION_PREVIOUS_WORD]:
                # Ctrl + Left / Right
                self._vprint(f"on_text_motion({keys.symbol_string(motion)})")
                incr = 1 if motion == keys.MOTION_NEXT_WORD else -1
                active_tiles = self.img_per_tile[:self.numtiles]
                consecutive = np.ptp(active_tiles) + 1 == self.numtiles
                stride = self.numtiles if consecutive else 1
                active_tiles = np.array(active_tiles) + incr * stride
                active_tiles = active_tiles % self.files.numfiles
                self.img_per_tile[:self.numtiles] = active_tiles
                self.window.set_caption(self._caption())
                self.need_redraw = True

    def _try(self, func):
        try:
            func()
        except Exception as e:
            self.running = False
            self.event_loop.has_exit = True
            if self.verbose:
                self._vprint(f"exception in {func.__name__}():")
                traceback.print_exc()
            else:
                print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {type(e).__name__}: {e}")

    def _vprint(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
