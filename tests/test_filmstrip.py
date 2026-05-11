"""Unit tests for the thumbnail filmstrip layout, hit-test, and cache logic."""

import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np

from glview.filmstrip import (
    Filmstrip,
    _CellLayout,
    _letterbox_to_cell,
    _to_rgba_uint8,
    compute_layout,
    hit_test_cells,
)
from glview.imagestate import ImageSlot, ImageStatus


@dataclass(frozen=True)
class _StubEntry:
    slot_id: int
    revision: int = 0
    status: ImageStatus = ImageStatus.LOADED
    image_slot: ImageSlot = None  # noqa: RUF013

    @classmethod
    def make(cls, slot_id, revision=0, status=ImageStatus.LOADED):
        return cls(slot_id=slot_id, revision=revision, status=status,
                   image_slot=ImageSlot(slot_id=slot_id, status=status, revision=revision))


@dataclass(frozen=True)
class _StubSnapshot:
    entries: tuple
    numfiles: int

    @classmethod
    def of(cls, entries):
        entries = tuple(entries)
        return cls(entries=entries, numfiles=len(entries))


class _StubState:
    def __init__(self, numtiles=1, tileidx=0, img_per_tile=(0,)):
        self.numtiles = numtiles
        self.tileidx = tileidx
        self.img_per_tile = np.array(list(img_per_tile) + [0] * (4 - len(img_per_tile)), dtype=int)


class _StubFiles:
    def __init__(self, images=None):
        self.images = images or {}

    def get_consumed_image(self, idx):
        return self.images.get(idx)


class ComputeLayoutTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        cells, sy, sh = compute_layout(0, (800, 600), 96, 80, 4, 0)
        self.assertEqual(cells, ())
        self.assertEqual((sy, sh), (0, 96))

    def test_centered_block_when_everything_fits(self):
        # 3 cells, stride 84, total = 3*84 - 4 = 248. Win w 800 -> first_x = (800-248)/2 = 276.
        cells, _, _ = compute_layout(3, (800, 96), 96, 80, 4, 1)
        self.assertEqual(cells[0].x, 276)
        self.assertEqual(cells[1].x, 276 + 84)
        self.assertEqual(cells[2].x, 276 + 168)
        self.assertEqual(cells[0].w, 80)
        self.assertEqual(cells[0].y, (96 - 80) // 2)

    def test_auto_centers_active_when_strip_overflows(self):
        # 50 cells of stride 84 = 4196 px wide; window 400 px. Center idx 25.
        cells, _, _ = compute_layout(50, (400, 96), 96, 80, 4, 25)
        center_cell = cells[25]
        midpoint = center_cell.x + center_cell.w // 2
        self.assertAlmostEqual(midpoint, 200, delta=1)

    def test_clamps_at_left_edge(self):
        cells, _, _ = compute_layout(50, (400, 96), 96, 80, 4, 0)
        self.assertEqual(cells[0].x, 0)

    def test_clamps_at_right_edge(self):
        cells, _, _ = compute_layout(50, (400, 96), 96, 80, 4, 49)
        last = cells[49]
        self.assertEqual(last.x + last.w, 400)


class HitTestTests(unittest.TestCase):

    def test_hits_correct_cell(self):
        cells, sy, sh = compute_layout(5, (800, 96), 96, 80, 4, 2)
        target = cells[3]
        idx = hit_test_cells(cells, sy, sh, target.x + 5, target.y + 5)
        self.assertEqual(idx, 3)

    def test_misses_outside_strip(self):
        cells, sy, sh = compute_layout(5, (800, 96), 96, 80, 4, 2)
        self.assertIsNone(hit_test_cells(cells, sy, sh, 10, 500))

    def test_misses_in_gap_between_cells(self):
        cells, sy, sh = compute_layout(5, (800, 96), 96, 80, 4, 2)
        gap_x = cells[0].x + cells[0].w + 1  # in the 4 px pad
        self.assertIsNone(hit_test_cells(cells, sy, sh, gap_x, cells[0].y + 5))


class HelperTests(unittest.TestCase):

    def test_to_rgba_uint8_handles_uint8_rgb(self):
        img = np.full((4, 6, 3), 128, dtype=np.uint8)
        out = _to_rgba_uint8(img)
        self.assertEqual(out.shape, (4, 6, 4))
        self.assertTrue(np.all(out[..., :3] == 128))
        self.assertTrue(np.all(out[..., 3] == 255))

    def test_to_rgba_uint8_handles_grayscale(self):
        img = np.full((4, 6), 200, dtype=np.uint8)
        out = _to_rgba_uint8(img)
        self.assertEqual(out.shape, (4, 6, 4))
        self.assertTrue(np.all(out[..., 0] == out[..., 1]))
        self.assertTrue(np.all(out[..., 1] == out[..., 2]))

    def test_to_rgba_uint8_normalizes_floats(self):
        img = np.full((2, 2, 3), 2.0, dtype=np.float32)
        out = _to_rgba_uint8(img, normalize_max=4.0)
        self.assertTrue(np.all(out[..., :3] == 127) or np.all(out[..., :3] == 128))

    def test_letterbox_centers_landscape_in_square_cell(self):
        thumb = np.full((4, 16, 4), 200, dtype=np.uint8)
        thumb[..., 3] = 255
        canvas = _letterbox_to_cell(thumb, 32)
        self.assertEqual(canvas.shape, (32, 32, 4))
        # Centered horizontally, with padding rows top and bottom (filled with bg).
        # Resized image is 32 wide, height 8.
        self.assertTrue(np.any(canvas[16, :, 0] > 100))


class FilmstripTests(unittest.TestCase):

    def test_toggle_flips_visible(self):
        fs = Filmstrip()
        self.assertFalse(fs.visible)
        fs.toggle()
        self.assertTrue(fs.visible)
        fs.toggle()
        self.assertFalse(fs.visible)

    def test_update_skips_when_hidden(self):
        fs = Filmstrip()
        snapshot = _StubSnapshot.of([_StubEntry.make(0)])
        state = _StubState()
        fs.update((800, 600), state, snapshot, _StubFiles())
        self.assertEqual(fs._cells, ())

    def test_update_caches_cpu_thumbnail(self):
        fs = Filmstrip(cell_size=16)
        fs.visible = True
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        snapshot = _StubSnapshot.of([_StubEntry.make(slot_id=7, revision=0)])
        state = _StubState()
        files = _StubFiles({0: img})
        with mock.patch.object(fs, "_rebuild_draw_resources"):
            fs.update((800, 600), state, snapshot, files)
        self.assertIn((7, 0), fs._thumb_cache)
        thumb = fs._thumb_cache[(7, 0)]
        self.assertEqual(thumb.shape, (16, 16, 4))

    def test_update_marks_invalid_as_placeholder(self):
        fs = Filmstrip(cell_size=16)
        fs.visible = True
        snapshot = _StubSnapshot.of([_StubEntry.make(slot_id=1, status=ImageStatus.INVALID)])
        state = _StubState()
        with mock.patch.object(fs, "_rebuild_draw_resources"):
            fs.update((800, 600), state, snapshot, _StubFiles())
        self.assertIn((1, 0), fs._thumb_cache)
        self.assertIsNone(fs._thumb_cache[(1, 0)])

    def test_update_evicts_stale_revisions(self):
        fs = Filmstrip(cell_size=16)
        fs.visible = True
        # Pre-populate a stale entry.
        fs._thumb_cache[(7, 0)] = np.zeros((16, 16, 4), dtype=np.uint8)
        snapshot = _StubSnapshot.of([_StubEntry.make(slot_id=7, revision=1, status=ImageStatus.PENDING)])
        state = _StubState()
        with mock.patch.object(fs, "_rebuild_draw_resources"):
            fs.update((800, 600), state, snapshot, _StubFiles())
        self.assertNotIn((7, 0), fs._thumb_cache)

    def test_hit_test_returns_imgidx_when_visible(self):
        fs = Filmstrip(cell_size=16, cell_pad=0, strip_height=20)
        fs.visible = True
        entries = [_StubEntry.make(i) for i in range(3)]
        snapshot = _StubSnapshot.of(entries)
        state = _StubState()
        with mock.patch.object(fs, "_rebuild_draw_resources"):
            fs.update((800, 600), state, snapshot, _StubFiles())
        target = fs._cells[1]
        self.assertEqual(fs.hit_test(target.x + 1, target.y + 1), 1)
        self.assertIsNone(fs.hit_test(target.x + 1, fs._strip_y + fs._strip_h + 5))

    def test_preload_caches_even_when_hidden(self):
        fs = Filmstrip(cell_size=16)
        # visible stays False
        img = np.full((24, 24, 3), 50, dtype=np.uint8)
        snapshot = _StubSnapshot.of([_StubEntry.make(slot_id=3, revision=0)])
        files = _StubFiles({0: img})
        fs.preload(snapshot, files)
        self.assertIn((3, 0), fs._thumb_cache)

    def test_preload_persists_after_source_disappears(self):
        fs = Filmstrip(cell_size=16)
        img = np.full((24, 24, 3), 200, dtype=np.uint8)
        snapshot = _StubSnapshot.of([_StubEntry.make(slot_id=9, revision=0)])
        files = _StubFiles({0: img})
        fs.preload(snapshot, files)
        # Simulate the renderer releasing the CPU image after upload.
        files.images.clear()
        fs.preload(snapshot, files)
        self.assertIn((9, 0), fs._thumb_cache)
        self.assertEqual(fs._thumb_cache[(9, 0)].shape, (16, 16, 4))


if __name__ == "__main__":
    unittest.main()
