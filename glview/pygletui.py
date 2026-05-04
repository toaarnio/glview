""" A graphical user interface for glview, based on Pyglet and ModernGL. """

import threading               # built-in library
import traceback               # built-in library
from pathlib import Path       # built-in library

import pyglet                  # pip install pyglet
import numpy as np             # pip install numpy

from glview import uiops
from glview.imagestate import ImageStatus
from glview.viewconfig import ViewConfigState
from glview.viewerstate import ViewerState


class PygletUI:
    """ A graphical user interface for glview, based on Pyglet and ModernGL. """

    def __init__(self, files, debug, verbose=False):
        """ Create a new PygletUI with the given (hardcoded) FileList instance. """
        self.thread_name = "UIThread"
        self.config = ViewConfigState(debug_mode=debug)
        self.verbose = verbose
        self.files = files
        self.version = None
        self.fullscreen = False
        self.state = ViewerState()
        self.running = None
        self.need_redraw = True
        self.was_resized = True
        self.window = None
        self.ops = uiops.UIOperations(self)
        self.key_state = None
        self.winsize = None
        self.mouse_speed = 2.0
        self.mouse_canvas_width = 1000
        self.keyboard_pan_speed = 100
        self.viewports = None
        self.ui_thread = None
        self.event_loop = None
        self.loader = None
        self.renderer = None
        self.images_pending = True
        self.ss_idx = 0

    def start(self, renderer):
        """ Start the UI thread. """
        self._vprint(f"spawning {self.thread_name}...")
        self.renderer = renderer
        self.loader = renderer.loader
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
                removed = parent.loader.apply_updates()
                if removed:
                    parent.ops.finish_removal()
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
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        self.winsize = (screen.width // 3, screen.height // 3)
        self.viewports = self._retile(self.state.numtiles, self.winsize, self.state.layout)
        self.window = pyglet.window.Window(*self.winsize, resizable=True, vsync=True)
        self.window.set_caption(self._caption())
        self.window.set_fullscreen(self.fullscreen)
        self.window.set_mouse_visible(not self.fullscreen)
        self.key_state = dict.fromkeys(pyglet.window.key._key_names, False)
        self._patch_gl_cleanup()
        self._setup_events()
        self._vprint("Pyglet & native OpenGL initialized")

    @staticmethod
    def _patch_gl_cleanup():
        """
        On Windows, pyglet uses WGLFunctionProxy to lazily resolve OpenGL 1.2+
        function pointers via wglGetProcAddress on the first call. If the first
        call happens after the GL context is destroyed (e.g. from BufferObject.__del__
        via Context.delete_buffer, or from set_current's _delete_objects/_delete_objects_one_by_one),
        wglGetProcAddress returns NULL and pyglet raises MissingFunctionException.

        Wrap the affected GL functions in pyglet.gl with a silent exception handler.
        This covers all call paths without probing (probing caused an access violation).
        """
        def _make_safe(fn):
            def safe(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except pyglet.gl.lib.MissingFunctionException:
                    pass
            return safe

        for name in [
            "glDeleteBuffers",        # OpenGL 1.5 — called from BufferObject.__del__
            "glDeleteProgram",        # OpenGL 2.0 — called from shader program __del__
            "glDeleteShader",         # OpenGL 2.0
            "glDeleteVertexArrays",   # OpenGL 3.0
            "glDeleteFramebuffers",   # OpenGL 3.0
            "glDeleteRenderbuffers",  # OpenGL 3.0
        ]:
            fn = getattr(pyglet.gl, name, None)
            if fn is not None:
                setattr(pyglet.gl, name, _make_safe(fn))

    def _poll_loading(self):
        """
        Trigger a redraw event when the currently visible image(s) have been
        loaded from disk. Otherwise, a placeholder dummy image would remain
        visible until the user performs some interaction.
        """
        snapshot = self.files.snapshot()
        for imgidx in self.state.img_per_tile[:self.state.numtiles]:
            if snapshot.entries[imgidx].status == ImageStatus.PENDING:
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
            snapshot = self.files.snapshot()
            indices = self.state.img_per_tile[:self.state.numtiles]
            indices = list(indices) + list(range(snapshot.numfiles))
            for imgidx in indices:
                status = snapshot.entries[imgidx].status
                if status not in [ImageStatus.PENDING, ImageStatus.INVALID]:
                    slot_id = snapshot.entries[imgidx].slot_id
                    texture = self.renderer.textures.get_cached(slot_id)
                    if texture is None or not texture.done:
                        texture = self.renderer.textures.upload(imgidx, piecewise=True)
                        self.need_redraw = texture.done
                        break  # upload only one slice of one texture per call

    def _caption(self):
        snapshot = self.files.snapshot()
        ver = self.version
        fps = np.median(self.renderer.fps)
        cspaces = ["sRGB", "DCI-P3", "Rec2020", "XYZ"]
        csc = f"{cspaces[self.config.cs_in]} => {cspaces[self.config.cs_out]}"
        norm = ["off", "max", "stretch", "99.5%", "98%", "95%", "90%", "mean"][self.config.normalize]
        ae = np.asarray(["N", "Y"])[np.asarray(self.state.ae_per_tile).astype(int)]
        ae = "".join(ae)[:self.state.numtiles]  # [False, True, True, False] => "NYYN"
        gtm = np.asarray(["N", "Y"])[np.asarray(self.state.tonemap_per_tile).astype(int)]
        gtm = "".join(gtm)[:self.state.numtiles]  # [False, True, True, False] => "NYYN"
        gmap = np.asarray(["N", "Y"])[np.asarray(self.state.gamutmap_per_tile).astype(int)]
        gmap = "".join(gmap)[:self.state.numtiles]  # [False, True, True, False] => "NYYN"
        sharpen = np.asarray(["N", "Y"])[np.asarray(self.state.sharpen_per_tile).astype(int)]
        sharpen = "".join(sharpen)[:self.state.numtiles]  # [False, True, True, False] => "NYYN"
        gamma = ["off", "sRGB", "HLG", "HDR10"][self.config.gamma]
        caption = f"glview {ver} | {self.config.ev:+1.2f}EV | norm {norm} | {csc} | "
        caption += f"ae {ae} | tonemap {gtm} | gamut {gmap} | sharpen {sharpen} | gamma {gamma} | {fps:.0f} fps"

        # Filenames and paths are hard to fit into the title bar in multi-tile mode,
        # so we need to make a compromise: show filenames if all files are in the same
        # directory; otherwise show the directory name but not the filename

        dirnames = [Path(entry.filespec).parent for entry in snapshot.entries]
        basenames = [Path(entry.filespec).name for entry in snapshot.entries]
        hide_dirname = np.unique(dirnames).size == 1
        for tileidx in range(self.state.numtiles):
            imgidx = self.state.img_per_tile[tileidx]
            label = Path(snapshot.entries[imgidx].filespec)
            if self.state.numtiles > 1:  # show folder name or filename but not both
                label = basenames[imgidx] if hide_dirname else dirnames[imgidx]
            caption = f"{caption} | {label} [{imgidx+1}/{snapshot.numfiles}]"
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
            dx = self.key_state[keys.LEFT] - self.key_state[keys.RIGHT]
            dy = self.key_state[keys.DOWN] - self.key_state[keys.UP]
            changed = self.state.keyboard_pan_zoom(
                zoom_steps=(self.key_state[keys.PLUS], self.key_state[keys.MINUS]),
                pan_steps=(dx, dy),
                pan_speed=self.keyboard_pan_speed,
                canvas_width=self.mouse_canvas_width,
            )
            if changed:
                self.need_redraw = True

    def _triangle_wave(self, x, amplitude):
        # [0, 1] => [-amplitude, +amplitude]
        y = 4 * amplitude * np.abs((x - 0.25) % 1 - 0.5) - amplitude
        return y

    def _smooth_exposure(self):
        # this is typically invoked 60 times per second,
        # so exposure control is pretty fast
        keys = pyglet.window.key
        shift_down = self.key_state[keys.LSHIFT] or self.key_state[keys.RSHIFT]
        changed = self.config.update_exposure(
            increase=bool(not shift_down and self.key_state[keys.E]),
            triangle_wave=self._triangle_wave,
        )
        if changed:
            self.need_redraw = True

    def _reset_view_command(self):
        self.state.reset_view()
        self.config.reset_exposure()
        self.need_redraw = True

    def _toggle_linearize_current(self):
        imgidx = self.state.img_per_tile[self.state.tileidx]
        self.files.toggle_linearize(imgidx)
        self.need_redraw = True

    def _cycle_gamma(self):
        self.config.cycle_gamma()
        self.need_redraw = True

    def _cycle_input_colorspace(self):
        self.config.cycle_input_colorspace()
        self.need_redraw = True

    def _cycle_output_colorspace(self):
        self.config.cycle_output_colorspace()
        self.need_redraw = True

    def _toggle_exposure_range(self):
        self.config.toggle_exposure_range()
        self.need_redraw = True

    def _cycle_normalize(self):
        self.config.cycle_normalize()
        self.state.reset_ae()
        self.need_redraw = True

    def _toggle_texture_filter(self):
        self.config.toggle_texture_filter()
        self.need_redraw = True

    def _toggle_ae_command(self):
        self.state.toggle_ae()
        self.need_redraw = True

    def _toggle_tonemap_command(self):
        self.state.toggle_tonemap()
        self.need_redraw = True

    def _toggle_gamutmap_command(self):
        self.state.toggle_gamutmap()
        self.need_redraw = True

    def _toggle_sharpen_command(self):
        self.state.toggle_sharpen()
        self.need_redraw = True

    def _cycle_split_command(self):
        self.state.cycle_split(self.files.numfiles)
        self.viewports = self._retile(self.state.numtiles, self.winsize, self.state.layout)
        self.window.set_caption(self._caption())
        self.need_redraw = True

    def _flip_pair_command(self):
        self.state.flip_pair()
        self.window.set_caption(self._caption())
        self.need_redraw = True

    def _rotate_current_image(self):
        imgidx = self.state.img_per_tile[self.state.tileidx]
        self.files.rotate_orientation(imgidx)
        self.need_redraw = True

    def _cycle_mirror_command(self):
        self.state.cycle_mirror()
        self.need_redraw = True

    def _simple_key_actions(self, keys):
        return {
            keys.ESCAPE: self.ops.request_exit,
            keys.Q: self.ops.request_exit,
            keys.F: self.ops.toggle_fullscreen,
            keys.H: self._reset_view_command,
            keys.L: self._toggle_linearize_current,
            keys.G: self._cycle_gamma,
            keys.A: self._toggle_ae_command,
            keys.C: self._toggle_tonemap_command,
            keys.K: self._toggle_gamutmap_command,
            keys.Z: self._toggle_sharpen_command,
            keys.I: self._cycle_input_colorspace,
            keys.O: self._cycle_output_colorspace,
            keys.B: self._toggle_exposure_range,
            keys.N: self._cycle_normalize,
            keys.T: self._toggle_texture_filter,
            keys.S: self._cycle_split_command,
            keys.R: self._rotate_current_image,
            keys.M: self._cycle_mirror_command,
            keys.U: self.ops.reload_visible_images,
            keys.SPACE: self.ops.toggle_debug_mode,
        }

    def _handle_key_press(self, symbol, modifiers):
        keys = pyglet.window.key
        self._vprint(f"on_key_press({keys.symbol_string(symbol)}, modifiers={keys.modifiers_string(modifiers)})")
        disallowed_keys = keys.MOD_CTRL | keys.MOD_ALT | keys.MOD_WINDOWS | keys.MOD_COMMAND
        if symbol == keys.C and modifiers == keys.MOD_CTRL:
            self.ops.request_exit()
        if (modifiers & disallowed_keys) == 0:
            self._dispatch_key_press(symbol, keys)

    def _dispatch_key_press(self, symbol, keys):
        action = self._simple_key_actions(keys).get(symbol)
        if action is not None:
            action()
        elif symbol == keys.P and self.state.numtiles == 2:
            self._flip_pair_command()
        elif symbol == keys.X:
            self.ops.show_exif_for_current()
        elif symbol == keys.W:
            self.ops.take_screenshot()
        elif symbol == keys.D:
            self.ops.remove_visible_images()
        elif symbol == keys.DELETE:
            self.ops.delete_current_image()
        elif symbol in [keys._1, keys._2, keys._3, keys._4]:
            tileidx = symbol - keys._1
            self.ops.select_tile(tileidx)

    def _setup_events(self):
        self._vprint("setting up Pyglet window event handlers...")
        self._setup_draw_event()
        self._setup_resize_event()
        self._setup_close_event()
        self._setup_mouse_events()
        self._setup_keyboard_events()

    def _setup_draw_event(self):
        @self.window.event
        def on_draw():
            if self.need_redraw or not np.all(self.renderer.ae_converged):
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

    def _setup_resize_event(self):
        @self.window.event
        def on_resize(width, height):
            self._vprint(f"on_resize({width}, {height})")
            self.winsize = (width, height)
            self.viewports = self._retile(self.state.numtiles, self.winsize, self.state.layout)
            self.need_redraw = True
            self.was_resized = True

    def _setup_close_event(self):
        @self.window.event
        def on_close():
            self.renderer.release()
            self.running = False
            self.event_loop.has_exit = True

    def _setup_mouse_events(self):
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
                self.state.drag_mouse(
                    dx,
                    dy,
                    self.mouse_speed,
                    self.mouse_canvas_width,
                    bool(shift_down),
                )
                self.need_redraw = True

        @self.window.event
        def on_mouse_scroll(_x, _y, _scroll_x, scroll_y):
            keys = pyglet.window.key
            shift_down = self.key_state[keys.LSHIFT] or self.key_state[keys.RSHIFT]
            self.state.scroll_zoom(scroll_y, shift_down)
            self.need_redraw = True

    def _setup_keyboard_events(self):
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
        def on_key_press(symbol, modifiers):
            self.key_state[symbol] = True
            self._handle_key_press(symbol, modifiers)

        @self.window.event
        def on_text_motion(motion):
            keys = pyglet.window.key
            if motion in [keys.MOTION_NEXT_PAGE, keys.MOTION_PREVIOUS_PAGE]:
                # PageUp / PageDown
                self._vprint(f"on_text_motion({keys.symbol_string(motion)})")
                incr = 1 if motion == keys.MOTION_NEXT_PAGE else -1
                self.ops.step_active_tile(incr)
            if motion in [keys.MOTION_NEXT_WORD, keys.MOTION_PREVIOUS_WORD]:
                # Ctrl + Left / Right
                self._vprint(f"on_text_motion({keys.symbol_string(motion)})")
                incr = 1 if motion == keys.MOTION_NEXT_WORD else -1
                self.ops.step_all_tiles(incr)

    def _try(self, func):
        try:
            func()
        except Exception as e:
            self.running = False
            if self.event_loop is not None:
                self.event_loop.has_exit = True
            if self.verbose:
                self._vprint(f"exception in {func.__name__}():")
                traceback.print_exc()
            else:
                print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {type(e).__name__}: {e}")

    def _vprint(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}/{threading.current_thread().name}] {message}")
