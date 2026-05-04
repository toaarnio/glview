import unittest
from types import SimpleNamespace
from unittest import mock

import imgio
import imsize
import numpy as np

from glview.glview import FileList
from glview.imagestate import ImageStatus
from glview.imageprovider import ImageProvider, LoadRequest


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
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        provider.files.mark_loaded(0, img)
        provider.files.consume_image(0, img)

        self.assertIsInstance(provider.get_image(0), np.ndarray)

        provider.release_image(0)

        self.assertEqual(provider.get_image(0), ImageStatus.RELEASED.value)

    def test_apply_updates_moves_loaded_payloads_to_ui_owned_store(self):
        provider = self._provider(["a.png"])
        img = np.ones((2, 2, 3), dtype=np.uint8)
        slot_id, revision = provider.files.image_token(0)
        provider._update_queue.put(("loaded", 0, slot_id, revision, img))

        provider.apply_updates()

        self.assertEqual(provider.files.image_status(0), ImageStatus.LOADED)
        np.testing.assert_array_equal(provider.files.entry(0).loaded_image, img)
        np.testing.assert_array_equal(provider.files.entry(0).image, img)
        np.testing.assert_array_equal(provider.get_image(0), img)

    def test_apply_updates_drops_invalid_slots(self):
        provider = self._provider(["a.png", "b.png"])
        slot_id, revision = provider.files.image_token(0)
        provider._update_queue.put(("invalid", 0, slot_id, revision))

        removed = provider.apply_updates()

        self.assertTrue(removed)
        self.assertEqual([entry.filespec for entry in provider.files.entries], ["b.png"])
        self.assertEqual(provider.files.numfiles, 1)
        self.assertTrue(provider.files.reindexed)

    def test_apply_updates_ignores_stale_tokens(self):
        provider = self._provider(["a.png"])
        img = np.ones((2, 2, 3), dtype=np.uint8)
        stale_token = provider.files.image_token(0)
        provider.files.mark_pending(0)
        provider._update_queue.put(("loaded", 0, stale_token[0], stale_token[1], img))

        provider.apply_updates()

        self.assertEqual(provider.files.image_status(0), ImageStatus.PENDING)
        self.assertIsNone(provider.files.entry(0).image)

    def test_apply_updates_ignores_stale_indices_after_drop(self):
        provider = self._provider(["a.png", "b.png"])
        img = np.ones((2, 2, 3), dtype=np.uint8)
        stale_token = provider.files.image_token(0)
        provider.files.drop([0])
        provider._loader_statuses = [provider.files.image_status(0)]
        provider._loader_tokens = [provider.files.image_token(0)]
        provider._update_queue.put(("loaded", 0, stale_token[0], stale_token[1], img))

        provider.apply_updates()

        self.assertEqual([entry.filespec for entry in provider.files.entries], ["b.png"])
        self.assertEqual(provider.files.image_status(0), ImageStatus.PENDING)
        self.assertIsNone(provider.files.entry(0).image)

    def test_apply_updates_drops_multiple_invalid_slots(self):
        provider = self._provider(["a.png", "b.png", "c.png"])
        token0 = provider.files.image_token(0)
        token2 = provider.files.image_token(2)
        provider._update_queue.put(("invalid", 0, token0[0], token0[1]))
        provider._update_queue.put(("invalid", 2, token2[0], token2[1]))

        removed = provider.apply_updates()

        self.assertTrue(removed)
        self.assertEqual([entry.filespec for entry in provider.files.entries], ["b.png"])

    def test_apply_updates_returns_false_when_no_invalid_slots_are_removed(self):
        provider = self._provider(["a.png"])
        img = np.ones((2, 2, 3), dtype=np.uint8)
        slot_id, revision = provider.files.image_token(0)
        provider._update_queue.put(("loaded", 0, slot_id, revision, img))

        removed = provider.apply_updates()

        self.assertFalse(removed)

    def test_release_request_is_queued_for_loader_thread(self):
        provider = self._provider(["a.png"])
        provider.files.mark_loaded(0, np.ones((1, 1, 3), dtype=np.uint8))
        provider.files.consume_image(0, np.ones((1, 1, 3), dtype=np.uint8))
        token = provider.files.image_token(0)

        provider.release_image(0, token=token)
        with provider.files.mutex:
            provider._drain_requests_locked()

        self.assertEqual(provider.files.image_status(0), ImageStatus.RELEASED)
        self.assertEqual(provider._loader_statuses[0], ImageStatus.RELEASED)
        self.assertIsNone(provider.files.entry(0).image)

    def test_stale_release_request_is_ignored(self):
        provider = self._provider(["a.png"])
        stale_token = provider.files.image_token(0)
        provider.reload_image(0)
        with provider.files.mutex:
            provider._drain_requests_locked()

        provider.release_image(0, token=stale_token)
        with provider.files.mutex:
            provider._drain_requests_locked()

        self.assertEqual(provider._loader_statuses[0], ImageStatus.PENDING)

    def test_reload_request_is_queued_for_loader_thread(self):
        provider = self._provider(["a.png"])
        provider._loader_statuses[0] = ImageStatus.RELEASED

        provider.reload_image(0)
        with provider.files.mutex:
            provider._drain_requests_locked()

        self.assertEqual(provider.files.image_status(0), ImageStatus.PENDING)
        self.assertEqual(provider._loader_statuses[0], ImageStatus.PENDING)

    def test_publish_result_ignores_decode_finished_after_release(self):
        provider = self._provider(["a.png"])
        request = LoadRequest(
            idx=0,
            token=provider.files.image_token(0),
            filespec="a.png",
            linearize=False,
        )
        provider.release_image(0, token=request.token)
        with provider.files.mutex:
            provider._drain_requests_locked()
            published = provider._publish_result_locked(request, np.ones((1, 1, 3), dtype=np.uint8))

        self.assertFalse(published)
        self.assertEqual(provider._loader_statuses[0], ImageStatus.RELEASED)

    def test_load_single_returns_none_for_non_pending_slot(self):
        provider = self._provider(["a.png"])
        provider.files.mark_released(0)
        provider._loader_statuses[0] = ImageStatus.RELEASED

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
        self.assertFalse(provider.files.entry(0).linearize)

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
        self.assertFalse(provider.files.entry(0).linearize)

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

        self.assertEqual(result, ImageStatus.INVALID.value)

    def test_load_single_marks_invalid_metadata_reads(self):
        provider = self._provider(["a.png"])

        with mock.patch("glview.imageprovider.imsize.read", side_effect=imsize.ImageFileError("bad metadata")):
            result = provider._load_single(0, downsample=1, verbose=False)

        self.assertEqual(result, ImageStatus.INVALID.value)


if __name__ == "__main__":
    unittest.main()
