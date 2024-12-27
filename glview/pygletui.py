""" A graphical user interface for glview, based on Pyglet and ModernGL. """

import threading               # built-in library
import pprint                  # built-in library
import traceback               # built-in library
import pyglet                  # pip install pyglet
import piexif                  # pip install piexif
import numpy as np             # pip install numpy
import imsize                  # pip install imsize
import imgio                   # pip install imgio


class PygletUI:
    """ A graphical user interface for glview, based on Pyglet and ModernGL. """

    def __init__(self, files, debug, verbose=False):
        """ Create a new PygletUI with the given (hardcoded) FileList instance. """
        self.thread_name = "UIThread"
        self.debug_selected = debug  # selected debug rendering mode: 1|2|3|...
        self.debug_mode = 0  # start in normal mode, toggle on/off with space
        self.verbose = verbose
        self.files = files
        self.version = None
        self.fullscreen = False
        self.numtiles = 1
        self.running = None
        self.need_redraw = True
        self.was_resized = True
        self.window = None
        self.key_state = None
        self.winsize = None
        self.tileidx = 0
        self.scale = np.ones(4)  # per-tile scale
        self.mousepos = np.zeros((4, 2))  # per-tile (x, y); clipped to [0, 1]
        self.mouse_speed = 4.0
        self.mouse_canvas_width = 1000
        self.keyboard_pan_speed = 100
        self.viewports = None
        self.layout = "N x 1"  # N x 1 | 1 x N | 2 x 2
        self.ui_thread = None
        self.event_loop = None
        self.renderer = None
        self.texture_filter = "NEAREST"
        self.img_per_tile = [0, 1, 2, 3]
        self.tonemap_per_tile = [False, False, False, False]
        self.images_pending = True
        self.cs_in = 0
        self.cs_out = 0
        self.gamma = 1
        self.gtm_ymax = 0
        self.gtm_linear = 0
        self.normalize = 0  # 0|1|2|...
        self.ev_range = 2
        self.ev_linear = 0.0
        self.ev = 0.0
        self.gamut_fit = 0  # 0|1|2...
        self.gamut_lim = np.ones(3) * 1.1
        self.gamut_pow = np.ones(3) * 1.5
        self.gamut_thr = np.ones(3) * 0.8
        self.gamut_lin = 0.0
        self.ss_idx = 0

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
                parent._keyboard_zoom_pan()
                parent._smooth_exposure()
                parent._poll_loading()
                parent._upload_textures()
                window = next(iter(pyglet.app.windows))
                window.dispatch_event("on_draw")
                return 1/60  # pan/zoom at max 60 fps

        return _EventLoop()

    def _init_pyglet(self):
        self._vprint("initializing Pyglet & native OpenGL...")
        pyglet.options['debug_lib'] = self.verbose
        pyglet.options['debug_gl'] = self.verbose
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        self.winsize = (screen.width // 3, screen.height // 3)
        self.viewports = self._retile(self.numtiles, self.winsize, self.layout)
        self.window = pyglet.window.Window(*self.winsize, resizable=True, vsync=True)
        self.window.set_caption(self._caption())
        self.window.set_fullscreen(self.fullscreen)
        self.window.set_mouse_visible(not self.fullscreen)
        self.key_state = {k: False for k in pyglet.window.key._key_names}
        self._setup_events()
        self._vprint("Pyglet & native OpenGL initialized")

    def _poll_loading(self):
        """
        Trigger a redraw event when the currently visible image(s) have been
        loaded from disk. Otherwise, a placeholder dummy image would remain
        visible until the user performs some interaction.
        """
        for imgidx in self.img_per_tile[:self.numtiles]:
            img = self.files.images[imgidx]
            if isinstance(img, str) and img == "PENDING":
                self.images_pending = True
                break
        else:
            # if the for-loop completes without breaking, it means that
            # all currently visible images have been loaded from disk;
            # now trigger a redraw, but only if some images were pending
            # on the previous invocation
            if self.images_pending:
                self.images_pending = False
                self.need_redraw = True

    def _upload_textures(self):
        """
        Upload a slice of the first non-completed texture to OpenGL, but only if
        there are no pending redraw requests (to keep the UI responsive). Textures
        that are currently visible are prioritized, otherwise uploading proceeds
        in index order.
        """
        if not self.need_redraw:
            indices = self.img_per_tile[:self.numtiles]
            indices = list(indices) + list(range(self.files.numfiles))
            for imgidx in indices:
                if self.files.ready_to_upload(imgidx):
                    texture = self.files.textures[imgidx]
                    if texture is None or not texture.extra.done:
                        texture = self.renderer.upload_texture(imgidx, piecewise=True)
                        self.need_redraw = texture.extra.done
                        break  # upload only one slice of one texture per call

    def _caption(self):
        ver = self.version
        fps = np.median(self.renderer.fps)
        cspaces = ["sRGB", "DCI-P3", "Rec2020"]
        csc = f"{cspaces[self.cs_in]} => {cspaces[self.cs_out]}"
        norm = ["off", "max", "stretch", "99.5%", "98%", "95%", "90%", "mean"][self.normalize]
        gtm = np.asarray(["N", "Y"])[np.asarray(self.tonemap_per_tile).astype(int)]
        gtm = "".join(gtm)[:self.numtiles]  # [False, True, True, False] => "NYYN"
        gamma = ["off", "sRGB", "HLG", "HDR10"][self.gamma]
        gamut = "clip" if not self.gamut_fit else f"fit p = {self.gamut_pow[0]:.1f}"
        caption = f"glview {ver} | {self.ev:+1.2f}EV | norm {norm} | {csc} | "
        caption += f"gamut {gamut} | tonemap {gtm} | gamma {gamma} | {fps:.0f} fps"

        # Show filenames in the title bar such that the first name is displayed
        # in full, the others as deltas with respect to the first, common substrings
        # replaced with asterisks. For example, "foobar.jpg" and "foobar.png" would
        # be displayed as "foobar.jpg" and "*.png".

        basenames = []
        for tileidx in range(self.numtiles):
            imgidx = self.img_per_tile[tileidx]
            basename = self.files.filespecs[imgidx]
            basenames.append(basename)

        def max_substr(strings):
            """ Return the longest common substring. """
            subs = lambda x: {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}
            s = subs(strings[0])
            for val in strings[1:]:
                s.intersection_update(subs(val))
            return max(s, key=len)

        def shorten(ref, strings):
            """ Replace common substrings with an asterisk. """
            shortened = [ref]
            for string in strings[1:]:
                pair = [ref, string]
                substr = max_substr(pair)
                substr = substr.lstrip(".").rstrip(".")
                while len(substr) >= 8:
                    pair = [s.replace(substr, "*") for s in pair]
                    substr = max_substr(pair)
                shortened.append(pair[1])
            return shortened

        if len(basenames) > 1:
            basenames[1:] = shorten(basenames[0], basenames[1:])
        for tileidx in range(self.numtiles):
            imgidx = self.img_per_tile[tileidx]
            basename = basenames[tileidx]
            caption = f"{caption} | {basename} [{imgidx+1}/{self.files.numfiles}]"
        return caption

    def _retile(self, numtiles, winsize, layout):
        w, h = winsize
        viewports = {}
        if numtiles == 1:
            vpw, vph = (w, h)
            viewports[0] = (0, 0, vpw, vph)
        elif numtiles == 2:
            if layout == "N x 1":
                vpw, vph = (w // 2, h)
                viewports[0] = (0,   0, vpw, vph)
                viewports[1] = (vpw, 0, vpw, vph)
            elif layout == "1 x N":
                vpw, vph = (w, h // 2)
                viewports[0] = (0, vph, vpw, vph)
                viewports[1] = (0, 0, vpw, vph)
        elif numtiles == 3:
            vpw, vph = (w // 3, h)
            viewports[0] = (0 * vpw, 0, vpw, vph)
            viewports[1] = (1 * vpw, 0, vpw, vph)
            viewports[2] = (2 * vpw, 0, vpw, vph)
        elif numtiles == 4:
            if layout == "2 x 2":
                vpw, vph = (w // 2, h // 2)
                viewports[0] = (0,   vph, vpw, vph)  # bottom left => top left
                viewports[1] = (vpw, vph, vpw, vph)  # bottom right => top right
                viewports[2] = (0,   0,   vpw, vph)  # top left => bottom left
                viewports[3] = (vpw, 0,   vpw, vph)  # top right => bottom right
            elif layout == "N x 1":
                vpw, vph = (w // 4, h)
                viewports[0] = (0 * vpw, 0, vpw, vph)
                viewports[1] = (1 * vpw, 0, vpw, vph)
                viewports[2] = (2 * vpw, 0, vpw, vph)
                viewports[3] = (3 * vpw, 0, vpw, vph)
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
        win_down = self.key_state[keys.LWINDOWS] or self.key_state[keys.RWINDOWS]
        if win_down:
            # Clear arrow key state while the Windows key is pressed;
            # this is needed because Windows appears to be stealing
            # arrow key release events when the user is rearranging
            # windows with Win + up/down/left/right.
            self.key_state[keys.LEFT] = False
            self.key_state[keys.RIGHT] = False
            self.key_state[keys.UP] = False
            self.key_state[keys.DOWN] = False
        if not ctrl_down and not win_down:
            prev_scale = self.scale.copy()
            self.scale *= 1.0 + 0.1 * self.key_state[keys.PLUS]  # zoom in
            self.scale /= 1.0 + 0.1 * self.key_state[keys.MINUS]  # zoom out
            dx = self.key_state[keys.LEFT] - self.key_state[keys.RIGHT]
            dy = self.key_state[keys.DOWN] - self.key_state[keys.UP]
            dxdy = np.tile((dx, dy), (4, 1))
            dxdy = dxdy * self.keyboard_pan_speed
            dxdy = dxdy / self.scale[:, np.newaxis]
            dxdy = dxdy / self.mouse_canvas_width
            self.mousepos = np.clip(self.mousepos + dxdy, -1.0, 1.0)
            if np.any(dxdy != 0.0) or np.any(self.scale != prev_scale):
                self.need_redraw = True

    def _triangle_wave(self, x, amplitude):
        # [0, 1] => [-amplitude, +amplitude]
        y = 4 * amplitude * np.abs((x - 0.25) % 1 - 0.5) - amplitude
        return y

    def _sine_wave(self, x, amplitude):
        # [0, 1] => [-amplitude, +amplitude]
        y = np.sin(x * 2 * np.pi) * amplitude
        return y

    def _smooth_exposure(self):
        # this is typically invoked 60 times per second,
        # so exposure control is pretty fast
        keys = pyglet.window.key
        shift_down = self.key_state[keys.LSHIFT] or self.key_state[keys.RSHIFT]
        if not shift_down and self.key_state[keys.E]:
            self.ev_linear += 0.005 * self.key_state[keys.E]
            self.need_redraw = True
        self.ev = self._triangle_wave(self.ev_linear, self.ev_range)
        # tonemap y-limit control
        if shift_down and self.key_state[keys.E]:
            self.gtm_linear += 0.005 * self.key_state[keys.E]
            self.need_redraw = True
        self.gtm_ymax = 2 + self._sine_wave(self.gtm_linear, 1)  # [-1, 1] => [1, 3]

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

    def _crop_borders(self, img):
        nonzero = np.any(img != 0.0, axis=2)
        rowmask = np.any(nonzero, axis=1)
        img = img[rowmask, :]
        nonzero = np.any(img != 0.0, axis=2)
        colmask = np.any(nonzero, axis=0)
        img = img[:, colmask]
        return img

    def _setup_events(self):  # noqa: PLR0915, C901
        self._vprint("setting up Pyglet window event handlers...")

        @self.window.event
        def on_draw():
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
            self._vprint(f"on_resize({width}, {height})")
            self.winsize = (width, height)
            self.viewports = self._retile(self.numtiles, self.winsize, self.layout)
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
        def on_mouse_drag(_x, _y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                keys = pyglet.window.key
                shift_down = modifiers & keys.MOD_SHIFT
                tidx = self.tileidx if shift_down else np.s_[:]
                dxdy = np.tile((dx, dy), (4, 1))
                dxdy = dxdy * self.mouse_speed
                dxdy = dxdy / self.scale[tidx, np.newaxis]
                dxdy = dxdy / self.mouse_canvas_width
                mousepos = np.clip(self.mousepos[tidx] + dxdy[tidx], -1.0, 1.0)
                self.mousepos[tidx] = mousepos
                self.need_redraw = True

        @self.window.event
        def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
            keys = pyglet.window.key
            shift_down = self.key_state[keys.LSHIFT] or self.key_state[keys.RSHIFT]
            tidx = self.tileidx if shift_down else np.s_[:]
            scale_factor = 1.0 + 0.1 * scroll_y
            self.scale[tidx] *= scale_factor
            self.need_redraw = True

        @self.window.event
        def on_close():
            self.running = False
            self.event_loop.has_exit = True

        @self.window.event
        def on_key_release(symbol, modifiers):
            keys = pyglet.window.key
            if symbol in [keys.LWINDOWS, keys.RWINDOWS]:
                # Clear arrow key state when the Windows key is released;
                # this is needed because Windows appears to be stealing
                # arrow key release events when the user is rearranging
                # windows with Win + up/down/left/right.
                self.key_state[keys.LEFT] = False
                self.key_state[keys.RIGHT] = False
                self.key_state[keys.UP] = False
                self.key_state[keys.DOWN] = False
            self.key_state[symbol] = False
            self._vprint(f"on_key_release({keys.symbol_string(symbol)}, modifiers={keys.modifiers_string(modifiers)})")

        @self.window.event
        def on_key_press(symbol, modifiers):  # noqa: PLR0912, PLR0915, C901
            self.key_state[symbol] = True
            keys = pyglet.window.key
            self._vprint(f"on_key_press({keys.symbol_string(symbol)}, modifiers={keys.modifiers_string(modifiers)})")
            disallowed_keys = keys.MOD_CTRL | keys.MOD_ALT | keys.MOD_WINDOWS | keys.MOD_COMMAND
            if symbol == keys.C and modifiers == keys.MOD_CTRL:
                self.running = False
                self.event_loop.has_exit = True
            if (modifiers & disallowed_keys) == 0:  # ignore NumLock, ScrollLock, CapsLock, Shift
                match symbol:
                    case keys.ESCAPE | keys.Q:  # exit
                        self.running = False
                        self.event_loop.has_exit = True
                    case keys.F:  # fullscreen
                        self.fullscreen = not self.fullscreen
                        self.need_redraw = True
                        self.was_resized = True
                        self.window.set_fullscreen(self.fullscreen)
                        self.window.set_mouse_visible(not self.fullscreen)
                    case keys.H:  # reset exposure + zoom & pan + gtm + gamut (global)
                        self.scale = np.ones(4)
                        self.mousepos = np.zeros((4, 2))
                        self.ev_linear = 0.0
                        self.gtm_linear = 0.0
                        self.gamut_lin = 0.0
                        self.gamut_fit = 0
                        self.need_redraw = True
                    case keys.L:  # toggle linearization on/off (current image)
                        imgidx = self.img_per_tile[self.tileidx]
                        self.files.linearize[imgidx] = not self.files.linearize[imgidx]
                        self.need_redraw = True
                    case keys.G:  # cycle through gamma modes (global)
                        self.gamma = (self.gamma + 1) % 4
                        self.need_redraw = True
                    case keys.C:  # toggle tone mapping on/off (current tile)
                        self.tonemap_per_tile[self.tileidx] = not self.tonemap_per_tile[self.tileidx]
                        self.need_redraw = True
                    case keys.I:  # input color space (global)
                        self.cs_in = (self.cs_in + 1) % 3
                        self.need_redraw = True
                    case keys.O:  # output color space (global)
                        self.cs_out = (self.cs_out + 1) % 3
                        self.need_redraw = True
                    case keys.B:  # toggle narrow/wide exposure control (global)
                        self.ev_range = (self.ev_range + 6) % 12
                        self.need_redraw = True
                    case keys.K: # cycle through gamut compression modes (global)
                        self._switch_gamut_curve()
                        self.need_redraw = True
                    case keys.N:  # normalize off/max/... (global)
                        self.normalize = (self.normalize + 1) % 8
                        self.need_redraw = True
                    case keys.T:  # texture filtering (global)
                        self.texture_filter = "LINEAR" if self.texture_filter == "NEAREST" else "NEAREST"
                        self.need_redraw = True
                    case keys.S:  # split
                        if self.numtiles == 4 and self.layout == "N x 1":
                            self.layout = "2 x 2"
                        elif self.numtiles == 2 and self.layout == "N x 1":
                            self.layout = "1 x N"
                        else:
                            self.layout = "N x 1"
                            self.numtiles = (self.numtiles % 4) + 1
                            self.tileidx = min(self.tileidx, self.numtiles - 1)
                            self.img_per_tile = np.clip(self.img_per_tile, 0, self.files.numfiles - 1)
                        self.viewports = self._retile(self.numtiles, self.winsize, self.layout)
                        self.window.set_caption(self._caption())
                        self.need_redraw = True
                    case keys.P if self.numtiles == 2:  # flip image pair
                        self.img_per_tile[:2] = self.img_per_tile[:2][::-1]
                        self.window.set_caption(self._caption())
                        self.need_redraw = True
                    case keys.R:  # rotate (current image)
                        imgidx = self.img_per_tile[self.tileidx]
                        self.files.orientations[imgidx] += 90
                        self.files.orientations[imgidx] %= 360
                        self.need_redraw = True
                    case keys.U:  # reload currently visible images from disk
                        for imgidx in self.img_per_tile[:self.numtiles]:
                            self.files.images[imgidx] = "PENDING"
                    case keys.X:  # EXIF info (current image)
                        imgidx = self.img_per_tile[self.tileidx]
                        filespec = self.files.filespecs[imgidx]
                        fileinfo = imsize.read(filespec)
                        print(fileinfo)
                        self._print_exif(filespec)
                    case keys.W:  # take a screenshot
                        screenshot_uint8 = self.renderer.screenshot(np.uint8)
                        screenshot_fp32 = self.renderer.screenshot(np.float32)
                        screenshot_uint8 = self._crop_borders(screenshot_uint8)
                        screenshot_fp32 = self._crop_borders(screenshot_fp32)
                        imgio.imwrite(f"screenshot{self.ss_idx:02d}.jpg", screenshot_uint8, maxval=255, verbose=True)
                        imgio.imwrite(f"screenshot{self.ss_idx:02d}.pfm", screenshot_fp32, maxval=1.0, verbose=True)
                        self.ss_idx += 1
                    case keys.SPACE:  # toggle debug mode on/off
                        N = self.debug_selected
                        self.debug_mode = (self.debug_mode + N) % (N * 2)
                        self._vprint(f"debug rendering mode {self.debug_mode}")
                        self.need_redraw = True
                    case keys.D | keys.DELETE:
                        if not self.files.mutex.locked():
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
                    case keys._1 | keys._2 | keys._3 | keys._4:
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
