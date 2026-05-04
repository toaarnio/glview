import unittest
from pathlib import Path

import numpy as np

from glview.glview import FileList
from glview.imagestate import ImageStatus
from glview.pygletui import PygletUI


class _FakeUploadedTexture:

    def __init__(self, done):
        self.done = done


class _FakeRenderer:

    def __init__(self, upload_results=None, loader=None):
        self.fps = np.array([58.0, 60.0, 62.0])
        self.loader = loader
        self.textures = _FakeTextureManager(upload_results or {})


class _FakeTextureManager:

    def __init__(self, upload_results):
        self.upload_results = upload_results
        self.upload_calls = []
        self.cached_textures = {}

    def upload(self, imgidx, piecewise, snapshot=None):
        self.upload_calls.append((imgidx, piecewise))
        return self.upload_results[imgidx]

    def get_cached(self, slot_id):
        return self.cached_textures.get(slot_id)


class _FakeLoader:

    def __init__(self):
        self.reload_calls = []
        self.apply_updates_calls = 0
        self.apply_updates_result = False

    def apply_updates(self):
        self.apply_updates_calls += 1
        return self.apply_updates_result

    def reload_image(self, imgidx):
        self.reload_calls.append(imgidx)


class _FakeWindow:

    def __init__(self):
        self.caption = None
        self.fullscreen = None
        self.mouse_visible = None

    def set_caption(self, caption):
        self.caption = caption

    def set_fullscreen(self, fullscreen):
        self.fullscreen = fullscreen

    def set_mouse_visible(self, visible):
        self.mouse_visible = visible


class PygletUISmokeTests(unittest.TestCase):

    def _ui(self, filespecs):
        files = FileList(filespecs)
        ui = PygletUI(files, debug=1, verbose=False)
        ui.version = "test"
        ui.loader = _FakeLoader()
        ui.renderer = _FakeRenderer(loader=ui.loader)
        ui.window = _FakeWindow()
        ui.winsize = (120, 80)
        ui.viewports = ui._retile(ui.state.numtiles, ui.winsize, ui.state.layout)
        ui.event_loop = type("EventLoop", (), {"has_exit": False})()
        return ui

    def test_poll_loading_requests_redraw_when_visible_images_finish_loading(self):
        ui = self._ui(["a.png", "b.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)
        ui.images_pending = True
        ui.need_redraw = False
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.files.mark_loaded(1, np.zeros((1, 1, 3), dtype=np.uint8))

        ui._poll_loading()

        self.assertFalse(ui.images_pending)
        self.assertTrue(ui.need_redraw)

    def test_poll_loading_keeps_waiting_when_any_visible_image_is_pending(self):
        ui = self._ui(["a.png", "b.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)
        ui.images_pending = False
        ui.need_redraw = False
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))

        ui._poll_loading()

        self.assertTrue(ui.images_pending)
        self.assertFalse(ui.need_redraw)

    def test_upload_textures_prioritizes_visible_images_and_stops_after_one_upload(self):
        ui = self._ui(["a.png", "b.png", "c.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([2, 1, 0, 3], dtype=int)
        ui.need_redraw = False
        ui.files.mark_released(0)
        ui.files.mark_loaded(1, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.files.mark_loaded(2, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.renderer = _FakeRenderer(
            upload_results={
                1: _FakeUploadedTexture(done=True),
                2: _FakeUploadedTexture(done=False),
            },
            loader=ui.loader,
        )

        ui._upload_textures()

        self.assertEqual(ui.renderer.textures.upload_calls, [(2, True)])
        self.assertFalse(ui.need_redraw)

    def test_upload_textures_sets_redraw_when_uploaded_texture_completes(self):
        ui = self._ui(["a.png"])
        ui.need_redraw = False
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.renderer = _FakeRenderer(upload_results={0: _FakeUploadedTexture(done=True)}, loader=ui.loader)

        ui._upload_textures()

        self.assertEqual(ui.renderer.textures.upload_calls, [(0, True)])
        self.assertTrue(ui.need_redraw)

    def test_upload_textures_is_skipped_while_redraw_is_pending(self):
        ui = self._ui(["a.png"])
        ui.need_redraw = True
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.renderer = _FakeRenderer(upload_results={0: _FakeUploadedTexture(done=True)}, loader=ui.loader)

        ui._upload_textures()

        self.assertEqual(ui.renderer.textures.upload_calls, [])

    def test_upload_textures_skips_when_cached_texture_is_already_done(self):
        ui = self._ui(["a.png"])
        ui.need_redraw = False
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        slot_id = ui.files.image_slot_id(0)
        ui.renderer.textures.cached_textures[slot_id] = _FakeUploadedTexture(done=True)

        ui._upload_textures()

        self.assertEqual(ui.renderer.textures.upload_calls, [])

    def test_caption_uses_filenames_when_all_files_share_directory(self):
        ui = self._ui(["/tmp/set/a.png", "/tmp/set/b.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)

        caption = ui._caption()

        self.assertIn("a.png [1/2]", caption)
        self.assertIn("b.png [2/2]", caption)

    def test_caption_uses_directory_names_in_multi_tile_mixed_directory_mode(self):
        ui = self._ui(["/tmp/set1/a.png", "/var/set2/b.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)

        caption = ui._caption()

        self.assertIn(str(Path("/tmp/set1")), caption)
        self.assertIn(str(Path("/var/set2")), caption)
        self.assertNotIn("a.png [1/2]", caption)

    def test_retile_returns_expected_viewports_for_two_tile_vertical_layout(self):
        ui = self._ui(["a.png", "b.png"])

        viewports = ui._retile(2, (120, 80), "1 x N")

        self.assertEqual(viewports[0], (0, 40, 120, 40))
        self.assertEqual(viewports[1], (0, 0, 120, 40))

    def test_reload_marks_visible_images_pending_and_clears_payloads(self):
        ui = self._ui(["a.png", "b.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)
        ui.files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        ui.files.mark_loaded(1, np.ones((1, 1, 3), dtype=np.uint8))
        ui.files.consume_image(0, np.full((1, 1, 3), 2, dtype=np.uint8))
        ui.files.consume_image(1, np.full((1, 1, 3), 3, dtype=np.uint8))

        ui.ops.reload_visible_images()

        self.assertEqual(ui.files.image_status(0), ImageStatus.PENDING)
        self.assertEqual(ui.files.image_status(1), ImageStatus.PENDING)
        self.assertIsNone(ui.files.loaded_images[0])
        self.assertIsNone(ui.files.loaded_images[1])
        self.assertIsNone(ui.files.images[0])
        self.assertIsNone(ui.files.images[1])
        self.assertEqual(ui.loader.reload_calls, [0, 1])

    def test_cycle_split_command_updates_state_and_viewports(self):
        ui = self._ui(["a.png", "b.png", "c.png"])

        ui._cycle_split_command()

        self.assertEqual(ui.state.numtiles, 2)
        self.assertEqual(ui.state.layout, "N x 1")
        self.assertEqual(ui.viewports[1], (60, 0, 60, 80))
        self.assertTrue(ui.need_redraw)
        self.assertIsNotNone(ui.window.caption)

    def test_remove_visible_images_repairs_view_and_requests_redraw(self):
        ui = self._ui(["a.png", "b.png", "c.png", "d.png"])
        ui.state.numtiles = 2
        ui.state.img_per_tile = np.array([2, 3, 0, 1], dtype=int)
        ui.need_redraw = False

        ui.ops.remove_visible_images()

        self.assertEqual(ui.files.numfiles, 2)
        np.testing.assert_array_equal(ui.state.img_per_tile, np.array([0, 1, 0, 1]))
        self.assertTrue(ui.need_redraw)
        self.assertIsNotNone(ui.window.caption)

    def test_event_loop_idle_repairs_view_after_loader_removal(self):
        ui = self._ui(["a.png", "b.png"])
        ui.loader.apply_updates_result = True
        ui.state.numtiles = 1
        ui.state.img_per_tile = np.array([0, 1, 2, 3], dtype=int)
        ui.files.drop([0])
        ui.need_redraw = False
        ui._keyboard_zoom_pan = lambda: None
        ui._smooth_exposure = lambda: None
        ui._poll_loading = lambda: None
        ui._upload_textures = lambda: None
        class _DummyWindow:
            def dispatch_event(self, _name):
                return None
        import pyglet
        original_windows = pyglet.app.windows
        pyglet.app.windows = [_DummyWindow()]
        try:
            loop = ui._create_eventloop()
            delay = loop.idle()
        finally:
            pyglet.app.windows = original_windows

        self.assertEqual(delay, 1/60)
        self.assertEqual(ui.loader.apply_updates_calls, 1)
        self.assertTrue(ui.need_redraw)
        self.assertEqual(ui.state.img_per_tile[0], 0)


if __name__ == "__main__":
    unittest.main()
