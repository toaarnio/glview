import unittest

import numpy as np

from glview import uistate


class UIStateTests(unittest.TestCase):

    def test_cycle_split_state_rotates_layouts_and_tile_count(self):
        state = uistate.SplitState(
            numtiles=1,
            layout="N x 1",
            tileidx=0,
            img_per_tile=np.array([0, 1, 2, 3]),
        )

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (2, "N x 1", 0))

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (2, "1 x N", 0))

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (3, "N x 1", 0))

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (4, "N x 1", 0))

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (4, "2 x 2", 0))

        state = uistate.cycle_split_state(state, numfiles=4)
        self.assertEqual((state.numtiles, state.layout, state.tileidx), (1, "N x 1", 0))

    def test_cycle_split_state_clamps_tile_index_and_visible_images(self):
        state = uistate.SplitState(
            numtiles=4,
            layout="2 x 2",
            tileidx=3,
            img_per_tile=np.array([7, 8, 9, 10]),
        )

        state = uistate.cycle_split_state(state, numfiles=3)

        self.assertEqual(state.numtiles, 1)
        self.assertEqual(state.layout, "N x 1")
        self.assertEqual(state.tileidx, 0)
        np.testing.assert_array_equal(state.img_per_tile, np.array([2, 2, 2, 2]))

    def test_flip_pair_reverses_first_two_entries_only(self):
        flipped = uistate.flip_pair([10, 20, 30, 40])

        self.assertEqual(flipped, [20, 10, 30, 40])

    def test_step_active_tile_wraps_single_tile(self):
        stepped = uistate.step_active_tile([0, 1, 2, 3], tileidx=1, incr=-1, numfiles=4)

        np.testing.assert_array_equal(stepped, np.array([0, 0, 2, 3]))

    def test_step_all_tiles_preserves_stride_for_consecutive_tiles(self):
        stepped = uistate.step_all_tiles([4, 5, 6, 9], numtiles=3, incr=1, numfiles=10)

        np.testing.assert_array_equal(stepped, np.array([7, 8, 9, 9]))

    def test_step_all_tiles_uses_unit_stride_for_non_consecutive_tiles(self):
        stepped = uistate.step_all_tiles([1, 3, 4, 9], numtiles=3, incr=-1, numfiles=10)

        np.testing.assert_array_equal(stepped, np.array([0, 2, 3, 9]))

    def test_repair_visible_images_after_removal_keeps_view_filled(self):
        repaired = uistate.repair_visible_images_after_removal([5, 6, 7, 8], numtiles=3, numfiles=10)

        np.testing.assert_array_equal(repaired, np.array([2, 3, 4, 8]))


if __name__ == "__main__":
    unittest.main()
