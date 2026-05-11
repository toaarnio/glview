import unittest
from unittest import mock

import numpy as np

from glview import ae
from glview.texture import Texture, _DEGAMMA_LUT


class _FakeTexture:

    def __init__(self, size, components, dtype, data=None):
        self.size = size
        self.width, self.height = size
        self.components = components
        self.dtype = dtype
        self.data = None if data is None else np.array(data)
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
        texture = _FakeTexture(size, components, dtype, data=data)
        self.created.append(texture)
        return texture


class TextureReuseTests(unittest.TestCase):

    def test_create_dummy_uses_full_uint8_color_range(self):
        ctx = _FakeContext()
        dummy = np.full((32, 32, 3), 127, dtype=np.uint8)

        class _FakeRng:
            def integers(self, low, high, size=None, dtype=None):
                self.args = (low, high, size, dtype)
                return dummy

        rng = _FakeRng()
        with mock.patch("glview.texture.np.random.default_rng", return_value=rng):
            tex = Texture(ctx, None, idx=0, verbose=False)

        self.assertEqual(rng.args, (0, 256, (32, 32, 3), np.uint8))
        self.assertEqual(tex.texture.dtype, "f1")
        np.testing.assert_array_equal(tex.texture.data, dummy)

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


class LinearizeTests(unittest.TestCase):

    def _make_texture(self, img, linearize, rawmax=65535):
        ctx = _FakeContext()
        return Texture(ctx, img.copy(), idx=0, verbose=False, linearize=linearize, rawmax=rawmax), ctx

    def test_linearize_uint8_creates_float16_texture(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, ctx = self._make_texture(img, linearize=True)
        self.assertEqual(ctx.created[0].dtype, "f2")
        self.assertEqual(tex.dtype, np.float16)

    def test_linearize_uint16_creates_float32_texture(self):
        img = np.full((4, 4, 3), 32768, dtype=np.uint16)
        tex, ctx = self._make_texture(img, linearize=True)
        self.assertEqual(ctx.created[0].dtype, "f4")
        self.assertEqual(tex.dtype, np.float32)

    def test_no_linearize_uint8_creates_fixed_point_texture(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, ctx = self._make_texture(img, linearize=False)
        self.assertEqual(ctx.created[0].dtype, "f1")
        self.assertEqual(tex.dtype, np.uint8)

    def test_no_linearize_uint16_creates_nu2_texture(self):
        """Linear uint16 (RAW/TIFF) uses nu2 (GL_RGB16); GPU normalizes by 65535."""
        img = np.full((4, 4, 3), 32768, dtype=np.uint16)
        tex, ctx = self._make_texture(img, linearize=False)
        self.assertEqual(ctx.created[0].dtype, "nu2")
        self.assertEqual(tex.dtype, np.uint16)

    def test_no_linearize_uint16_uploads_raw_data(self):
        """Raw uint16 bytes are written as-is; nu2 texture normalizes on the GPU."""
        img = np.full((4, 4, 3), 32768, dtype=np.uint16)
        tex, _ = self._make_texture(img, linearize=False)
        tex.upload(piecewise=False)
        written = tex.texture.writes[0][0]
        np.testing.assert_array_equal(written, img.ravel())

    def test_no_linearize_uint16_stats_normalized_by_65535(self):
        """maxval for raw uint16 is in [0, 1], consistent with nu2 GPU normalization."""
        img = np.full((12, 12, 3), 32768, dtype=np.uint16)
        tex, _ = self._make_texture(img, linearize=False)
        self.assertAlmostEqual(tex.maxval, 32768 / 65535.0, places=4)

    def test_linearize_uint8_uploads_degamma_converted_data(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=True)
        tex.upload(piecewise=False)
        written = tex.texture.writes[0][0]
        expected = _DEGAMMA_LUT[np.dtype("uint8")][img].ravel()
        np.testing.assert_array_equal(written, expected)

    def test_linearize_uint16_uploads_degamma_converted_data(self):
        img = np.full((4, 4, 3), 32768, dtype=np.uint16)
        tex, _ = self._make_texture(img, linearize=True)
        tex.upload(piecewise=False)
        written = tex.texture.writes[0][0]
        expected = _DEGAMMA_LUT[np.dtype("uint16")][img].ravel()
        np.testing.assert_array_equal(written, expected)

    def test_no_linearize_uint8_uploads_raw_data(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=False)
        tex.upload(piecewise=False)
        written = tex.texture.writes[0][0]
        np.testing.assert_array_equal(written, img.ravel())

    def test_linearize_uint8_stats_reflect_linear_values(self):
        img = np.full((12, 12, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=True)
        expected_max = float(_DEGAMMA_LUT[np.dtype("uint8")][128])
        self.assertAlmostEqual(tex.maxval, expected_max, places=4)

    def test_no_linearize_uint8_stats_reflect_gamma_values(self):
        img = np.full((12, 12, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=False)
        self.assertAlmostEqual(tex.maxval, 128 / 255.0, places=4)

    def test_linearize_does_not_affect_non_srgb_float32_images(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        tex, ctx = self._make_texture(img, linearize=True)
        self.assertEqual(ctx.created[0].dtype, "f4")
        tex.upload(piecewise=False)
        written = tex.texture.writes[0][0]
        np.testing.assert_array_equal(written, img.ravel())

    def test_reuse_with_linearize_keeps_existing_float16_texture(self):
        img1 = np.full((4, 4, 3), 100, dtype=np.uint8)
        img2 = np.full((4, 4, 3), 200, dtype=np.uint8)
        tex, _ = self._make_texture(img1, linearize=True)
        original = tex.texture
        tex.reuse(img2)
        self.assertIs(tex.texture, original)
        self.assertFalse(original.released)

    def test_no_linearize_uint16_bitdepth_scale_maps_bitdepth_max_to_one(self):
        """For 12-bit RAW (rawmax=4095), value 4095 must map to 1.0 in texture.fs."""
        img = np.full((12, 12, 3), 4095, dtype=np.uint16)
        tex, ctx = self._make_texture(img, linearize=False, rawmax=4095)
        self.assertAlmostEqual(tex.bitdepth_scale, 65535.0 / 4095, places=3)
        # GPU normalizes by 65535; texture.fs then scales by bitdepth_scale:
        self.assertAlmostEqual(4095 / 65535.0 * tex.bitdepth_scale, 1.0, places=5)

    def test_no_linearize_uint16_10bit_bitdepth_scale(self):
        """For 10-bit RAW (rawmax=1023), value 1023 must map to 1.0 in texture.fs."""
        img = np.full((12, 12, 3), 1023, dtype=np.uint16)
        tex, ctx = self._make_texture(img, linearize=False, rawmax=1023)
        self.assertAlmostEqual(4095 / 65535.0 * tex.bitdepth_scale, 4095.0 / 1023.0 / 65535.0 * 65535.0, places=3)
        self.assertAlmostEqual(1023 / 65535.0 * tex.bitdepth_scale, 1.0, places=5)

    def test_no_linearize_uint16_fullrange_bitdepth_scale_is_one(self):
        """For full 16-bit data (rawmax=65535), bitdepth_scale must be 1.0."""
        img = np.full((4, 4, 3), 32768, dtype=np.uint16)
        tex, _ = self._make_texture(img, linearize=False, rawmax=65535)
        self.assertEqual(tex.bitdepth_scale, 1.0)

    def test_linearize_uint8_bitdepth_scale_is_one(self):
        """sRGB uint8 is already linearized by the LUT; no bitdepth rescaling needed."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=True)
        self.assertEqual(tex.bitdepth_scale, 1.0)

    def test_no_linearize_uint8_bitdepth_scale_is_one(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        tex, _ = self._make_texture(img, linearize=False)
        self.assertEqual(tex.bitdepth_scale, 1.0)



    def test_estimate_diffuse_white_handles_black_image(self):
        black = np.zeros((8, 8), dtype=np.float32)

        diffuse_white = ae.estimate_diffuse_white(black)

        self.assertEqual(diffuse_white, 1.0)

    def test_autoexposure_handles_thin_mipmap_levels_without_zero_height(self):
        class _ThinTexture:
            size = (492, 1)

            def build_mipmaps(self, max_level):
                self.max_level = max_level

            def read(self, level):
                self.level = level
                stats = np.ones((1, 246, 3), dtype=np.float32)
                return stats.tobytes()

        texture = _ThinTexture()

        ae_gain, diffuse_white, peak_white = ae.autoexposure(texture, whitelevel=1.0, clip_pct=1.0)

        self.assertEqual(texture.max_level, 1)
        self.assertEqual(texture.level, 1)
        self.assertIsNotNone(ae_gain)
        self.assertIsNotNone(diffuse_white)
        self.assertIsNotNone(peak_white)


if __name__ == "__main__":
    unittest.main()
