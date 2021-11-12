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

    # pylint: disable=too-many-instance-attributes

    def __init__(self, files, verbose=False):
        """ Create a new PygletUI with the given (hardcoded) FileList instance. """
        self.thread_name = "UIThread"
        self.verbose = verbose
        self.files = files
        self.fullscreen = False
        self.numtiles = 1
        self.running = None
        self.window = None
        self.screensize = None
        self.winsize = None
        self.tileidx = 0
        self.scale = 1.0
        self.mousepos = np.zeros(2)  # always scaled & clipped to [0, 1] x [0, 1]
        self.mouse_speed = 4.0
        self.mouse_canvas_width = 1000
        self.viewports = None
        self.ui_thread = None
        self.event_loop = None
        self.renderer = None
        self.texture_filter = "NEAREST"
        self.img_per_tile = [0, 1, 2, 3]
        self.gamma = False
        self.ev = 0

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
        pyglet.clock.schedule_interval(lambda t: None, 0.5)  # trigger on_draw every 0.5 seconds
        self.event_loop = pyglet.app.EventLoop()
        self.event_loop.run()
        self._vprint("Pyglet event loop stopped")

    def _init_pyglet(self):
        self._vprint("initializing Pyglet & native OpenGL...")
        pyglet.options['debug_lib'] = self.verbose
        pyglet.options['debug_gl'] = self.verbose
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        self.screensize = (screen.width, screen.height)
        self.winsize = (screen.width // 3, screen.height // 3)
        self.viewports = self._retile(self.numtiles, self.winsize)
        self.window = pyglet.window.Window(*self.winsize, resizable=True, vsync=False)
        self.window.set_caption(self._caption())
        self.window.set_fullscreen(self.fullscreen)
        self.window.set_mouse_visible(not self.fullscreen)
        self._setup_events()
        self._vprint("Pyglet & native OpenGL initialized")

    def _caption(self):
        fps = pyglet.clock.get_fps()
        caption = f"glview [{self.ev:+1.1f}EV | {fps:.1f} fps]"
        for tileidx in range(self.numtiles):
            imgidx = self.img_per_tile[tileidx]
            basename = os.path.basename(self.files.filespecs[imgidx])
            caption = f"{caption} | {basename} [{imgidx+1}/{self.files.numfiles}]"
        return caption

    def _retile(self, numtiles, winsize):
        # pylint: disable=bad-whitespace
        # pylint: disable=no-self-use
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
        # pylint: disable=no-self-use
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

    def _setup_events(self):
        # pylint: disable=too-many-statements
        # pylint: disable=unused-variable
        self._vprint("setting up Pyglet window event handlers...")

        @self.window.event
        def on_draw():
            self.renderer.redraw()
            self.window.set_caption(self._caption())

        @self.window.event
        def on_resize(width, height):
            self.winsize = (width, height)
            self.viewports = self._retile(self.numtiles, self.winsize)
            self.window.dispatch_event("on_draw")

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

        @self.window.event
        def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
            scale_factor = 1.0 + 0.1 * scroll_y
            self.scale *= scale_factor

        @self.window.event
        def on_close():
            self.running = False
            self.event_loop.has_exit = True

        @self.window.event
        def on_key_press(symbol, modifiers):
            # pylint: disable=too-many-branches
            # pylint: disable=too-many-statements
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
                if symbol == keys.H:  # reset zoom & pan
                    self.scale = 1.0
                    self.mousepos = np.zeros(2)
                if symbol == keys.G:  # gamma
                    self.gamma = not self.gamma
                if symbol == keys.B:  # brightness
                    ev = (self.ev * 2) + 4
                    ev = (ev + 1) % 9  # [0, 8] ==> [-2, +2] EV in 0.5-EV steps
                    self.ev = (ev - 4) / 2
                if symbol == keys.T:  # texture filtering
                    self.texture_filter = "LINEAR" if self.texture_filter == "NEAREST" else "NEAREST"
                if symbol == keys.S:  # split
                    self.numtiles = (self.numtiles % 4) + 1
                    self.tileidx = min(self.tileidx, self.numtiles - 1)
                    self.viewports = self._retile(self.numtiles, self.winsize)
                    self.window.set_caption(self._caption())
                if symbol == keys.R:  # rotate
                    imgidx = self.img_per_tile[self.tileidx]
                    self.files.orientations[imgidx] += 90
                    self.files.orientations[imgidx] %= 360
                if symbol == keys.I:  # image info
                    imgidx = self.img_per_tile[self.tileidx]
                    filespec = self.files.filespecs[imgidx]
                    fileinfo = imsize.read(filespec)
                    print(fileinfo)
                    self._print_exif(filespec)
                if symbol == keys.DELETE:  # delete file, but not in split-screen mode
                    if self.numtiles == 1:
                        imgidx = self.img_per_tile[self.tileidx]
                        self.files.remove(imgidx)
                        if self.files.numfiles == 0:
                            self.running = False
                            self.event_loop.has_exit = True
                        else:
                            self.img_per_tile[self.tileidx] = (imgidx - 1) % self.files.numfiles
                            self.window.set_caption(self._caption())
                # pylint: disable=protected-access
                if symbol in [keys._1, keys._2, keys._3, keys._4]:
                    tileidx = symbol - keys._1
                    self.tileidx = tileidx if tileidx < self.numtiles else self.tileidx

        @self.window.event
        def on_text_motion(motion):  # handle PageUp / PageDown
            keys = pyglet.window.key
            self._vprint(f"on_text_motion({keys.symbol_string(motion)})")
            if motion in [keys.MOTION_NEXT_PAGE, keys.MOTION_PREVIOUS_PAGE]:
                incr = 1 if motion == keys.MOTION_NEXT_PAGE else -1
                imgidx = self.img_per_tile[self.tileidx]
                imgidx = (imgidx + incr) % self.files.numfiles
                self.img_per_tile[self.tileidx] = imgidx
                self.window.set_caption(self._caption())
            if motion in [keys.MOTION_LEFT, keys.MOTION_RIGHT]:
                incr = 1 if motion == keys.MOTION_RIGHT else -1
                active_tiles = self.img_per_tile[:self.numtiles]
                stride = max(active_tiles) - min(active_tiles)
                is_consecutive = (stride + 1 == self.numtiles)
                incr *= self.numtiles if is_consecutive else 1
                active_tiles = np.array(active_tiles) + incr
                if np.amax(active_tiles) >= self.files.numfiles:
                    active_tiles -= np.amin(active_tiles)
                if np.amin(active_tiles) < 0:
                    active_tiles += self.files.numfiles
                    active_tiles -= stride * (1 - int(is_consecutive))
                self.img_per_tile[:self.numtiles] = active_tiles
                self.window.set_caption(self._caption())

    def _try(self, func):
        try:
            func()
        except Exception as e:  # pylint: disable=broad-except
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
