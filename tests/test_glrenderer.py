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


class GLRendererParameterTests(unittest.TestCase):

    def _renderer(self, ui=None, files=None):
        if files is None:
            files = FileList(["a.png"])
        if ui is None:
            ui = SimpleNamespace(
                normalize=0,
                ae_per_tile=[False, False, False, False],
                ae_reset_per_tile=[False, False, False, False],
                mirror_per_tile=[0, 0, 0, 0],
                sharpen_per_tile=[False, False, False, False],
                tonemap_per_tile=[False, False, False, False],
                gamutmap_per_tile=[False, False, False, False],
                gamut_pow=np.ones(3) * 5.0,
                gamut_thr=np.ones(3) * 0.8,
                gamma=1,
                ev=0.0,
                cs_in=0,
                cs_out=0,
                scale=np.ones(4),
                debug_mode=1,
                debug_mode_on=False,
                gamut_lim=np.ones(3) * 1.1,
            )
        renderer = GLRenderer(ui, files, loader=SimpleNamespace(), verbose=False)
        renderer.postprocess = {"kernel": SimpleNamespace(array_length=49)}
        return renderer

    def test_normalization_levels_follow_selected_mode(self):
        renderer = self._renderer(ui=SimpleNamespace(normalize=3))
        texture_obj = SimpleNamespace(
            maxval=10.0,
            minval=0.25,
            percentiles=np.array([8.0, 7.0, 6.0, 5.0, 4.0]),
            diffuse_white=3.0,
        )

        whitelevel, blacklevel = renderer._normalization_levels(texture_obj)

        self.assertEqual(whitelevel, 8.0)
        self.assertEqual(blacklevel, 0.0)

    def test_resolve_tile_exposure_resets_or_falls_back_as_needed(self):
        ui = SimpleNamespace(
            ae_per_tile=[True, False, False, False],
            ae_reset_per_tile=[True, False, False, False],
        )
        renderer = self._renderer(ui=ui)
        renderer.ae_gain_per_tile = np.ones(4)
        renderer.ae_converged = [True, True, True, True]
        texture_obj = SimpleNamespace(maxval=12.0, diffuse_white=3.0)

        ae_gain, diffuse_white, peak_white = renderer._resolve_tile_exposure(
            tileidx=0,
            texture=texture_obj,
            ae_gain=2.0,
            tile_diffuse=None,
            tile_peak=None,
        )

        self.assertEqual(ae_gain, 2.0)
        self.assertEqual(diffuse_white, 3.0)
        self.assertEqual(peak_white, 4.0)
        self.assertFalse(renderer.ui.ae_reset_per_tile[0])
        self.assertTrue(renderer.ae_converged[0])

        ae_gain, diffuse_white, peak_white = renderer._resolve_tile_exposure(
            tileidx=1,
            texture=texture_obj,
            ae_gain=None,
            tile_diffuse=None,
            tile_peak=None,
        )

        self.assertEqual(ae_gain, 1.0)
        self.assertEqual(diffuse_white, 3.0)
        self.assertEqual(peak_white, 4.0)
        self.assertTrue(renderer.ae_converged[1])

    def test_build_postprocess_uniforms_reflects_ui_flags(self):
        ui = SimpleNamespace(
            normalize=0,
            ae_per_tile=[True, False, False, False],
            ae_reset_per_tile=[False, False, False, False],
            mirror_per_tile=[2, 0, 0, 0],
            sharpen_per_tile=[True, False, False, False],
            tonemap_per_tile=[True, False, False, False],
            gamutmap_per_tile=[True, False, False, False],
            gamut_pow=np.array([5.0, 6.0, 7.0]),
            gamut_thr=np.array([0.8, 0.7, 0.6]),
            gamma=3,
            ev=1.25,
            cs_in=2,
            cs_out=1,
            scale=np.array([2.0, 1.0, 1.0, 1.0]),
            debug_mode=4,
            debug_mode_on=True,
            gamut_lim=np.array([1.1, 1.2, 1.3]),
        )
        renderer = self._renderer(ui=ui)
        gpu_texture = SimpleNamespace(width=100)

        with mock.patch.object(renderer, "_sharpen", return_value=np.ones((3, 3))) as sharpen_mock:
            with mock.patch.object(renderer, "_gamut", return_value=np.array([0.9, 0.8, 0.7])) as gamut_mock:
                uniforms = renderer._build_postprocess_uniforms(
                    tileidx=0,
                    imgidx=0,
                    vpw=200,
                    vph=100,
                    gpu_texture=gpu_texture,
                    scalex=0.5,
                    whitelevel=2.0,
                    blacklevel=0.1,
                    diffuse_white=1.5,
                    peak_white=4.0,
                    ae_gain=1.75,
                )

        sharpen_mock.assert_called_once_with(2.0)
        gamut_mock.assert_called_once_with(0)
        self.assertEqual(uniforms["magnification"], 2.0)
        self.assertEqual(uniforms["mirror"], 2)
        self.assertTrue(uniforms["sharpen"])
        self.assertEqual(uniforms["kernw"], 3)
        self.assertEqual(uniforms["minval"], 0.1)
        self.assertEqual(uniforms["maxval"], 2.0)
        self.assertEqual(uniforms["diffuse_white"], 1.5)
        self.assertEqual(uniforms["peak_white"], 4.0)
        self.assertTrue(uniforms["autoexpose"])
        self.assertEqual(uniforms["ae_gain"], 1.75)
        self.assertEqual(uniforms["ev"], 1.25)
        self.assertEqual(uniforms["cs_in"], 2)
        self.assertEqual(uniforms["cs_out"], 1)
        self.assertEqual(uniforms["tonemap"], 3)
        self.assertTrue(uniforms["gamut.compress"])
        np.testing.assert_array_equal(uniforms["gamut.power"], np.array([5.0, 6.0, 7.0]))
        np.testing.assert_array_equal(uniforms["gamut.thr"], np.array([0.8, 0.7, 0.6]))
        np.testing.assert_array_equal(uniforms["gamut.scale"], np.array([0.9, 0.8, 0.7]))
        self.assertEqual(uniforms["contrast"], 0.25)
        self.assertEqual(uniforms["gamma"], 3)
        self.assertEqual(uniforms["debug"], 4)


if __name__ == "__main__":
    unittest.main()
