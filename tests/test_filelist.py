import tempfile
import unittest
from pathlib import Path

import numpy as np

from glview.glview import FileList
from glview.imagestate import ImageStatus


class _FakeTexture:

    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1


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

    def test_drop_releases_textures_and_reindexes_metadata(self):
        files = FileList(["a.png", "https://example.com/b.png", "c.png"])
        tex0 = _FakeTexture()
        tex2 = _FakeTexture()
        files.textures[0] = tex0
        files.textures[2] = tex2
        files.mark_released(0)
        files.mark_invalid(1)
        files.mark_loaded(2, np.ones((2, 2, 3), dtype=np.uint8))
        files.consume_image(2, np.full((2, 2, 3), 9, dtype=np.uint8))
        files.orientations = [90, 180, 270]
        files.linearize = [True, False, True]
        files.metadata = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        files.reindexed = False

        files.drop([0, 2])

        self.assertEqual(tex0.release_count, 1)
        self.assertEqual(tex2.release_count, 1)
        self.assertEqual(files.numfiles, 1)
        self.assertEqual(files.filespecs, ["https://example.com/b.png"])
        self.assertEqual(files.image_status(0), ImageStatus.INVALID)
        self.assertEqual(files.loaded_images, [None])
        self.assertEqual(files.images, [None])
        self.assertEqual(files.orientations, [180])
        self.assertEqual(files.linearize, [False])
        self.assertEqual(files.metadata, [{"name": "b"}])
        self.assertEqual(files.is_url, [True])
        self.assertTrue(files.reindexed)

    def test_delete_releases_texture_and_removes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.png"
            file_b = Path(tmpdir) / "b.png"
            file_a.write_bytes(b"a")
            file_b.write_bytes(b"b")

            files = FileList([str(file_a), str(file_b)])
            tex = _FakeTexture()
            files.textures[0] = tex
            files.mark_loaded(0, np.zeros((1, 1, 3), dtype=np.uint8))
            files.consume_image(0, np.ones((1, 1, 3), dtype=np.uint8))
            files.metadata = [{"name": "a"}, {"name": "b"}]
            files.reindexed = False

            files.delete(0)

            self.assertEqual(tex.release_count, 1)
            self.assertFalse(file_a.exists())
            self.assertTrue(file_b.exists())
            self.assertEqual(files.filespecs, [str(file_b)])
            self.assertEqual(files.image_status(0), ImageStatus.PENDING)
            self.assertEqual(files.loaded_images, [None])
            self.assertEqual(files.images, [None])
            self.assertEqual(files.metadata, [{"name": "b"}])
            self.assertEqual(files.numfiles, 1)
            self.assertEqual(files.is_url, [False])
            self.assertTrue(files.reindexed)

    def test_release_textures_ignores_none_entries(self):
        files = FileList(["a.png", "b.png", "c.png"])
        tex = _FakeTexture()
        files.textures = [None, tex, None]

        files.release_textures([0, 1, 2])

        self.assertEqual(tex.release_count, 1)

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
        self.assertIsNone(files.loaded_images[0])
        self.assertIsNone(files.images[0])


if __name__ == "__main__":
    unittest.main()
