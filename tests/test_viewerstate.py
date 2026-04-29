import unittest

import numpy as np

from glview.viewerstate import ViewerState


class ViewerStateTests(unittest.TestCase):

    def test_reset_view_restores_zoom_pan_and_marks_ae_reset(self):
        state = ViewerState(
            scale=np.array([2.0, 3.0, 4.0, 5.0]),
            mousepos=np.array([[0.5, -0.5], [0.1, 0.2], [0.3, 0.4], [0.6, 0.7]]),
            ae_reset_per_tile=[False, False, False, False],
        )

        state.reset_view()

        np.testing.assert_array_equal(state.scale, np.ones(4))
        np.testing.assert_array_equal(state.mousepos, np.zeros((4, 2)))
        self.assertEqual(state.ae_reset_per_tile, [True, True, True, True])

    def test_cycle_split_updates_layout_and_tile_count(self):
        state = ViewerState()

        state.cycle_split(numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (2, "N x 1", 0))

        state.cycle_split(numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (2, "1 x N", 0))

    def test_flip_pair_reorders_tile_data_and_marks_ae_reset(self):
        state = ViewerState(
            numtiles=2,
            img_per_tile=np.array([10, 20, 30, 40]),
            ae_per_tile=[True, False, False, False],
            ae_reset_per_tile=[False, False, False, False],
            tonemap_per_tile=[True, False, False, False],
            sharpen_per_tile=[True, False, False, False],
        )

        state.flip_pair()

        np.testing.assert_array_equal(state.img_per_tile, np.array([20, 10, 30, 40]))
        self.assertEqual(state.ae_per_tile, [False, True, False, False])
        self.assertEqual(state.ae_reset_per_tile[:2], [True, True])
        self.assertEqual(state.tonemap_per_tile, [False, True, False, False])
        self.assertEqual(state.sharpen_per_tile, [False, True, False, False])

    def test_step_active_tile_marks_only_active_tile_for_ae_reset(self):
        state = ViewerState(tileidx=1, img_per_tile=np.array([0, 1, 2, 3]))

        state.step_active_tile(incr=-1, numfiles=4)

        np.testing.assert_array_equal(state.img_per_tile, np.array([0, 0, 2, 3]))
        self.assertEqual(state.ae_reset_per_tile, [False, True, False, False])

    def test_step_all_tiles_marks_all_tiles_for_ae_reset(self):
        state = ViewerState(numtiles=3, img_per_tile=np.array([4, 5, 6, 9]))

        state.step_all_tiles(incr=1, numfiles=10)

        np.testing.assert_array_equal(state.img_per_tile, np.array([7, 8, 9, 9]))
        self.assertEqual(state.ae_reset_per_tile, [True, True, True, True])

    def test_toggle_methods_mutate_active_tile(self):
        state = ViewerState(tileidx=2)

        state.toggle_ae()
        state.toggle_tonemap()
        state.toggle_gamutmap()
        state.toggle_sharpen()
        state.cycle_mirror()

        self.assertTrue(state.ae_per_tile[2])
        self.assertTrue(state.ae_reset_per_tile[2])
        self.assertTrue(state.tonemap_per_tile[2])
        self.assertTrue(state.gamutmap_per_tile[2])
        self.assertTrue(state.sharpen_per_tile[2])
        self.assertEqual(state.mirror_per_tile[2], 1)

    def test_keyboard_pan_zoom_updates_all_tiles_and_reports_change(self):
        state = ViewerState()

        changed = state.keyboard_pan_zoom(
            key_zoom_in=1,
            key_zoom_out=0,
            dx=1,
            dy=-1,
            pan_speed=100,
            canvas_width=1000,
        )

        self.assertTrue(changed)
        np.testing.assert_allclose(state.scale, np.ones(4) * 1.1)
        np.testing.assert_allclose(state.mousepos[0], np.array([0.09090909, -0.09090909]))

    def test_drag_mouse_updates_only_active_tile_when_requested(self):
        state = ViewerState(tileidx=1)

        state.drag_mouse(dx=10, dy=-10, pan_speed=2.0, canvas_width=1000, active_only=True)

        np.testing.assert_allclose(state.mousepos[1], np.array([0.02, -0.02]))
        np.testing.assert_array_equal(state.mousepos[0], np.zeros(2))
        np.testing.assert_array_equal(state.mousepos[2], np.zeros(2))

    def test_scroll_zoom_updates_only_active_tile_when_requested(self):
        state = ViewerState(tileidx=2)

        state.scroll_zoom(scroll_y=1, active_only=True)

        np.testing.assert_allclose(state.scale, np.array([1.0, 1.0, 1.1, 1.0]))

    def test_repair_after_removal_preserves_filled_view(self):
        state = ViewerState(numtiles=3, img_per_tile=np.array([5, 6, 7, 8]))

        state.repair_after_removal(numfiles=10)

        np.testing.assert_array_equal(state.img_per_tile, np.array([2, 3, 4, 8]))


if __name__ == "__main__":
    unittest.main()
