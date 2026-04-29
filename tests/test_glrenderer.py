import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from glview.glrenderer import GLRenderer
from glview.glview import FileList


class _FakeLoader:

    def __init__(self, img, token):
        self.img = img
        self.token = token
        self.release_calls = []

    def get_image_record(self, idx):
        return self.img, self.token

    def release_image(self, idx, token=None):
        self.release_calls.append((idx, token))


class _FakeTextureObject:

    def __init__(self, _ctx, img, idx, verbose):
        self.img = img
        self.idx = idx
        self.verbose = verbose
        self.texture = SimpleNamespace(
            width=img.shape[1] if isinstance(img, np.ndarray) else 1,
            height=img.shape[0] if isinstance(img, np.ndarray) else 1,
            components=img.shape[2] if isinstance(img, np.ndarray) and img.ndim == 3 else 3,
        )
        self.done = False
        self.reuse_calls = []
        self.upload_calls = []
        self.release_count = 0

    def reuse(self, img):
        self.reuse_calls.append(img)

    def upload(self, piecewise):
        self.upload_calls.append(piecewise)

    def release(self):
        self.release_count += 1


class GLRendererTextureCacheTests(unittest.TestCase):

    def _renderer(self, files, loader):
        ui = SimpleNamespace()
        return GLRenderer(ui, files, loader, verbose=False)

    def test_upload_texture_caches_by_slot_id_and_releases_with_token(self):
        files = FileList(["a.png"])
        img = np.ones((2, 3, 3), dtype=np.uint8)
        token = files.image_token(0)
        loader = _FakeLoader(img, token)
        renderer = self._renderer(files, loader)
        renderer.ctx = object()

        with mock.patch("glview.glrenderer.texture.Texture", _FakeTextureObject):
            tex = renderer.upload_texture(0, piecewise=False)

        slot_id = files.image_slot_id(0)
        self.assertIs(renderer.texture_cache.get(slot_id), tex)
        self.assertEqual(loader.release_calls, [(0, token)])
        self.assertEqual(tex.upload_calls, [False])

    def test_prune_texture_cache_releases_removed_slot_ids(self):
        files = FileList(["a.png", "b.png"])
        renderer = self._renderer(files, loader=SimpleNamespace())
        tex0 = _FakeTextureObject(None, np.zeros((1, 1, 3), dtype=np.uint8), 0, False)
        tex1 = _FakeTextureObject(None, np.zeros((1, 1, 3), dtype=np.uint8), 1, False)
        slot0 = files.image_slot_id(0)
        slot1 = files.image_slot_id(1)
        renderer.texture_cache.store(slot0, tex0)
        renderer.texture_cache.store(slot1, tex1)

        files.drop([0])
        renderer._prune_texture_cache(files.snapshot())

        self.assertEqual(tex0.release_count, 1)
        self.assertEqual(tex1.release_count, 0)
        self.assertEqual(set(renderer.texture_cache.keys()), {slot1})


if __name__ == "__main__":
    unittest.main()
