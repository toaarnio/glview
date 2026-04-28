import unittest
from types import SimpleNamespace
from unittest import mock

import imgio
import imsize
import numpy as np

from glview.glview import FileList
from glview.imageprovider import ImageProvider


def _config(**overrides):
    config = SimpleNamespace(
        verbose=False,
        downsample=1,
        width=None,
        height=None,
        bpp=None,
        stride=None,
        packing=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class ImageProviderTests(unittest.TestCase):

    def _provider(self, filespecs, **config_overrides):
        with mock.patch.object(ImageProvider, "validate_files", autospec=True):
            provider = ImageProvider(FileList(filespecs), _config(**config_overrides))
        return provider

    def test_get_and_release_image_round_trip(self):
        provider = self._provider(["a.png"])
        provider.files.images[0] = np.zeros((1, 1, 3), dtype=np.uint8)

        self.assertIsInstance(provider.get_image(0), np.ndarray)

        provider.release_image(0)

        self.assertEqual(provider.get_image(0), "RELEASED")

    def test_load_single_returns_none_for_non_pending_slot(self):
        provider = self._provider(["a.png"])
        provider.files.images[0] = "RELEASED"

        result = provider._load_single(0, downsample=1, verbose=False)

        self.assertIsNone(result)

    def test_load_single_converts_grayscale_to_single_channel_3d_array(self):
        provider = self._provider(["a.tif"])
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", return_value=(gray, 255)),
        ):
            result = provider._load_single(0, downsample=2, verbose=False)

        self.assertEqual(result.shape, (2, 2, 1))
        np.testing.assert_array_equal(result[:, :, 0], gray[::2, ::2])
        self.assertFalse(provider.files.linearize[0])

    def test_load_single_drops_alpha_channel(self):
        provider = self._provider(["a.tif"])
        rgba = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4)
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", return_value=(rgba, 255)),
        ):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result.shape, (4, 5, 3))
        np.testing.assert_array_equal(result, rgba[:, :, :3])

    def test_load_single_applies_degamma_for_srgb_like_formats(self):
        provider = self._provider(["a.jpg"])
        rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        degamma = np.full((3, 4, 3), 0.25, dtype=np.float16)
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", return_value=(rgb, 255)),
            mock.patch.object(provider, "_degamma", return_value=degamma) as degamma_mock,
        ):
            result = provider._load_single(0, downsample=1, verbose=False)

        degamma_mock.assert_called_once()
        np.testing.assert_array_equal(result, degamma)
        self.assertFalse(provider.files.linearize[0])

    def test_load_single_converts_uint16_to_float32_using_max_of_image_and_header(self):
        provider = self._provider(["a.tif"])
        img = np.array([[[0], [512]], [[1024], [2048]]], dtype=np.uint16)
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", return_value=(img, 1023)),
        ):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[:, :, 0], img[:, :, 0].astype(np.float32) / 2048.0)

    def test_load_single_converts_float64_to_float32(self):
        provider = self._provider(["a.exr"])
        img = np.linspace(0, 1, 12, dtype=np.float64).reshape(2, 2, 3)
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", return_value=(img, 1.0)),
        ):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, img.astype(np.float32))

    def test_load_single_uses_rawread_for_cfa_raw_with_config_overrides(self):
        provider = self._provider(
            ["a.raw"],
            width=640,
            height=480,
            bpp=10,
            stride=2048,
            packing="mipi",
        )
        raw = np.arange(4 * 4, dtype=np.uint16).reshape(4, 4)
        info = SimpleNamespace(
            cfa_raw=True,
            packed_raw=False,
            mipi_raw=False,
            width=100,
            height=50,
            bitdepth=12,
            stride=512,
        )

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.rawread", return_value=(raw, 1023)) as rawread_mock,
        ):
            result = provider._load_single(0, downsample=1, verbose=True)

        rawread_mock.assert_called_once_with("a.raw", 640, 480, 10, 2048, "mipi", verbose=True)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (4, 4, 1))

    def test_load_single_uses_detected_raw_packing_when_not_overridden(self):
        provider = self._provider(["a.raw"])
        raw = np.arange(9, dtype=np.uint16).reshape(3, 3)
        info = SimpleNamespace(
            cfa_raw=True,
            packed_raw=True,
            mipi_raw=False,
            width=320,
            height=240,
            bitdepth=12,
            stride=1024,
        )

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.rawread", return_value=(raw, 4095)) as rawread_mock,
        ):
            provider._load_single(0, downsample=1, verbose=False)

        rawread_mock.assert_called_once_with("a.raw", 320, 240, 12, 1024, "plain", verbose=False)

    def test_load_single_marks_invalid_images(self):
        provider = self._provider(["a.png"])
        info = SimpleNamespace(cfa_raw=False)

        with (
            mock.patch("glview.imageprovider.imsize.read", return_value=info),
            mock.patch("glview.imageprovider.imgio.imread", side_effect=imgio.ImageIOError("bad image")),
        ):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result, "INVALID")

    def test_load_single_marks_invalid_metadata_reads(self):
        provider = self._provider(["a.png"])

        with mock.patch("glview.imageprovider.imsize.read", side_effect=imsize.ImageFileError("bad metadata")):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result, "INVALID")


if __name__ == "__main__":
    unittest.main()
