import unittest

import numpy as np

from glview import ae
from glview.texture import Texture


class _FakeTexture:

    def __init__(self, size, components, dtype):
        self.size = size
        self.width, self.height = size
        self.components = components
        self.dtype = dtype
        self.released = False
        self.writes = []
        self.mipmaps_built = 0

    def release(self):
        self.released = True

    def write(self, data, viewport=None):
        self.writes.append((np.array(data), viewport))

    def build_mipmaps(self):
        self.mipmaps_built += 1


class _FakeContext:

    def __init__(self):
        self.created = []

    def texture(self, size, components, data=None, dtype=None):
        texture = _FakeTexture(size, components, dtype)
        self.created.append(texture)
        return texture


class TextureReuseTests(unittest.TestCase):

    def test_reuse_recreates_texture_when_channel_count_changes(self):
        ctx = _FakeContext()
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        gray = np.zeros((4, 5, 1), dtype=np.uint8)

        tex = Texture(ctx, rgb, idx=0, verbose=False)
        original = tex.texture

        tex.reuse(gray)

        self.assertTrue(original.released)
        self.assertIsNot(tex.texture, original)
        self.assertEqual(tex.texture.size, (5, 4))
        self.assertEqual(tex.texture.components, 1)
        self.assertEqual(tex.components, 1)

    def test_reuse_recreates_texture_when_dtype_changes(self):
        ctx = _FakeContext()
        u8 = np.zeros((4, 5, 3), dtype=np.uint8)
        f16 = np.zeros((4, 5, 3), dtype=np.float16)

        tex = Texture(ctx, u8, idx=0, verbose=False)
        original = tex.texture

        tex.reuse(f16)

        self.assertTrue(original.released)
        self.assertIsNot(tex.texture, original)
        self.assertEqual(tex.texture.dtype, "f2")
        self.assertEqual(tex.dtype, np.float16)

    def test_compute_stats_marks_all_black_image_done_without_warnings(self):
        ctx = _FakeContext()
        black = np.zeros((12, 12, 3), dtype=np.uint8)

        tex = Texture(ctx, black, idx=0, verbose=False)

        self.assertTrue(tex.stats_done)
        self.assertEqual(tex.minval, 0.0)
        self.assertEqual(tex.maxval, 1.0)
        self.assertEqual(tex.meanval, 0.5)
        np.testing.assert_array_equal(tex.percentiles, np.ones(4))

    def test_upload_piecewise_completes_over_multiple_calls(self):
        ctx = _FakeContext()
        tall = np.arange(105 * 2 * 3, dtype=np.uint8).reshape(105, 2, 3)
        tex = Texture(ctx, tall.copy(), idx=1, verbose=False)

        tex.upload(piecewise=True)
        self.assertFalse(tex.upload_done)
        self.assertEqual(tex._rows_uploaded, 100)
        self.assertEqual(tex.texture.writes[0][1], (0, 0, 2, 100))
        np.testing.assert_array_equal(tex.texture.writes[0][0], tall[:100].ravel())

        tex.upload(piecewise=True)
        self.assertTrue(tex.upload_done)
        self.assertTrue(tex.mipmaps_done)
        self.assertIsNone(tex.img)
        self.assertEqual(tex._rows_uploaded, 105)
        self.assertEqual(tex.texture.writes[1][1], (0, 100, 2, 5))
        np.testing.assert_array_equal(tex.texture.writes[1][0], tall[100:].ravel())
        self.assertEqual(tex.texture.mipmaps_built, 1)

    def test_reuse_same_shape_dtype_keeps_existing_texture(self):
        ctx = _FakeContext()
        img1 = np.ones((4, 5, 3), dtype=np.uint8)
        img2 = np.full((4, 5, 3), 2, dtype=np.uint8)

        tex = Texture(ctx, img1, idx=0, verbose=False)
        original = tex.texture

        tex.reuse(img2)

        self.assertIs(tex.texture, original)
        self.assertFalse(original.released)
        self.assertFalse(tex.upload_done)
        self.assertFalse(tex.mipmaps_done)
        self.assertTrue(tex.stats_done)


class AutoExposureTests(unittest.TestCase):

    def test_estimate_diffuse_white_handles_black_image(self):
        black = np.zeros((8, 8), dtype=np.float32)

        diffuse_white = ae.estimate_diffuse_white(black)

        self.assertEqual(diffuse_white, 1.0)


if __name__ == "__main__":
    unittest.main()
