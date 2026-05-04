import tempfile
import unittest
from pathlib import Path

import numpy as np

from glview.glview import FileList
from glview.imagestate import ImageStatus


class FileListTests(unittest.TestCase):

    def test_ready_to_upload_distinguishes_pending_and_invalid(self):
        files = FileList(["a.png", "b.png", "c.png", "d.png"])
        files.mark_invalid(1)
        files.mark_released(2)
        files.mark_loaded(3, np.zeros((1, 1, 3), dtype=np.uint8))

        self.assertFalse(files.ready_to_upload(0))
        self.assertFalse(files.ready_to_upload(1))
        self.assertTrue(files.ready_to_upload(2))
        self.assertTrue(files.ready_to_upload(3))

    def test_drop_reindexes_metadata_and_keeps_remaining_slot_state(self):
        files = FileList(["a.png", "b.png", "c.png"])
        files.mark_released(0)
        files.mark_invalid(1)
        files.mark_loaded(2, np.ones((2, 2, 3), dtype=np.uint8))
        files.consume_image(2, np.full((2, 2, 3), 9, dtype=np.uint8))
        files.entry(0).orientation = 90
        files.entry(1).orientation = 180
        files.entry(2).orientation = 270
        files.set_linearize(0, True)
        files.set_linearize(1, False)
        files.set_linearize(2, True)
        files.entry(0).metadata = {"name": "a"}
        files.entry(1).metadata = {"name": "b"}
        files.entry(2).metadata = {"name": "c"}
        files.reindexed = False

        files.drop([0, 2])

        self.assertEqual(files.numfiles, 1)
        self.assertEqual([entry.filespec for entry in files.entries], ["b.png"])
        self.assertEqual(files.image_status(0), ImageStatus.INVALID)
        self.assertIsNone(files.entry(0).loaded_image)
        self.assertIsNone(files.entry(0).image)
        self.assertEqual(files.entry(0).orientation, 180)
        self.assertFalse(files.entry(0).linearize)
        self.assertEqual(files.entry(0).metadata, {"name": "b"})
        self.assertTrue(files.reindexed)

    def test_delete_removes_file_and_keeps_remaining_slot_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.png"
            file_b = Path(tmpdir) / "b.png"
            file_a.write_bytes(b"a")
            file_b.write_bytes(b"b")

            files = FileList([str(file_a), str(file_b)])
            files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
            files.consume_image(0, np.ones((1, 1, 3), dtype=np.uint8))
            files.entry(0).metadata = {"name": "a"}
            files.entry(1).metadata = {"name": "b"}
            files.reindexed = False

            files.delete(0)

            self.assertFalse(file_a.exists())
            self.assertTrue(file_b.exists())
            self.assertEqual([entry.filespec for entry in files.entries], [str(file_b)])
            self.assertEqual(files.image_status(0), ImageStatus.PENDING)
            self.assertIsNone(files.entry(0).loaded_image)
            self.assertIsNone(files.entry(0).image)
            self.assertEqual(files.entry(0).metadata, {"name": "b"})
            self.assertEqual(files.numfiles, 1)
            self.assertTrue(files.reindexed)

    def test_mark_pending_clears_loaded_and_consumed_payloads(self):
        files = FileList(["a.png"])
        loaded = np.zeros((1, 1, 3), dtype=np.uint8)
        consumed = np.ones((1, 1, 3), dtype=np.uint8)
        files.mark_loaded(0, loaded)
        files.consume_image(0, consumed)
        rev_before = files.image_revision(0)

        files.mark_pending(0)

        self.assertEqual(files.image_status(0), ImageStatus.PENDING)
        self.assertEqual(files.image_revision(0), rev_before + 1)
        self.assertIsNone(files.entry(0).loaded_image)
        self.assertIsNone(files.entry(0).image)

    def test_snapshot_returns_consistent_read_only_view(self):
        files = FileList(["a.png", "b.png"])
        files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
        files.set_linearize(1, True)
        files.entry(1).metadata = {"name": "b"}

        snapshot = files.snapshot()
        files.drop([0])

        self.assertEqual(snapshot.numfiles, 2)
        self.assertEqual(tuple(entry.filespec for entry in snapshot.entries), ("a.png", "b.png"))
        self.assertEqual(snapshot.entries[0].status, ImageStatus.LOADED)
        self.assertTrue(snapshot.entries[1].linearize)
        self.assertEqual(snapshot.entries[1].metadata, {"name": "b"})


if __name__ == "__main__":
    unittest.main()
