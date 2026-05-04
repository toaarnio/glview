import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from glview.glrenderer import GLRenderer
from glview.rendertargets import TileRenderTarget
from glview.rendertextures import RenderTextureManager
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


class _FakeFramebuffer:

    def __init__(self, size, attachment):
        self.size = size
        self.color_attachments = [attachment]
        self.use = mock.Mock()
        self.clear = mock.Mock()
        self.release = mock.Mock()


class _FakeContext:

    def __init__(self):
        self.textures = []
        self.framebuffers = []

    def texture(self, size, components, dtype):
        texture = SimpleNamespace(
            size=size,
            components=components,
            dtype=dtype,
            repeat_x=None,
            repeat_y=None,
            filter=None,
            use=mock.Mock(),
        )
        self.textures.append(texture)
        return texture

    def framebuffer(self, attachments):
        fbo = _FakeFramebuffer(size=attachments[0].size, attachment=attachments[0])
        self.framebuffers.append(fbo)
        return fbo


class TileRenderTargetTests(unittest.TestCase):

    def test_ensure_reuses_matching_framebuffer_and_recreates_on_resize(self):
        ctx = _FakeContext()
        target = TileRenderTarget(ctx, {"NEAREST": ("near", "near")})

        fbo0 = target.ensure(64, 32)
        fbo1 = target.ensure(64, 32)
        fbo2 = target.ensure(80, 40)

        self.assertIs(fbo0, fbo1)
        self.assertIsNot(fbo0, fbo2)
        self.assertEqual(len(ctx.textures), 2)
        self.assertEqual(ctx.textures[0].filter, ("near", "near"))
        self.assertFalse(ctx.textures[0].repeat_x)
        self.assertFalse(ctx.textures[0].repeat_y)

    def test_release_releases_existing_framebuffer(self):
        ctx = _FakeContext()
        target = TileRenderTarget(ctx, {"NEAREST": ("near", "near")})
        fbo = target.ensure(64, 32)

        target.release()

        fbo.release.assert_called_once_with()
        self.assertIsNone(target.fbo)


class RenderTextureManagerTests(unittest.TestCase):

    def test_upload_caches_by_slot_id_and_releases_with_token(self):
        files = FileList(["a.png"])
        img = np.ones((2, 3, 3), dtype=np.uint8)
        token = files.image_token(0)
        loader = _FakeLoader(img, token)
        manager = RenderTextureManager(ctx=object(), files=files, loader=loader, verbose=False)

        with mock.patch("glview.rendertextures.texture.Texture", _FakeTextureObject):
            tex = manager.upload(0, piecewise=False)

        slot_id = files.image_slot_id(0)
        self.assertIs(manager.get_cached(slot_id), tex)
        self.assertEqual(loader.release_calls, [(0, token)])
        self.assertEqual(tex.upload_calls, [False])

    def test_prune_releases_removed_slot_ids(self):
        files = FileList(["a.png", "b.png"])
        manager = RenderTextureManager(ctx=None, files=files, loader=SimpleNamespace(), verbose=False)
        tex0 = _FakeTextureObject(None, np.zeros((1, 1, 3), dtype=np.uint8), 0, False)
        tex1 = _FakeTextureObject(None, np.zeros((1, 1, 3), dtype=np.uint8), 1, False)
        slot0 = files.image_slot_id(0)
        slot1 = files.image_slot_id(1)
        manager.cache.store(slot0, tex0)
        manager.cache.store(slot1, tex1)

        files.drop([0])
        manager.prune(files.snapshot())

        self.assertEqual(tex0.release_count, 1)
        self.assertEqual(tex1.release_count, 0)
        self.assertEqual(set(manager.cache.keys()), {slot1})


class GLRendererParameterTests(unittest.TestCase):

    def _renderer(self, ui=None, files=None):
        if files is None:
            files = FileList(["a.png"])
        if ui is None:
            ui = SimpleNamespace(
                state=SimpleNamespace(
                    ae_per_tile=[False, False, False, False],
                    ae_reset_per_tile=[False, False, False, False],
                    mirror_per_tile=[0, 0, 0, 0],
                    sharpen_per_tile=[False, False, False, False],
                    tonemap_per_tile=[False, False, False, False],
                    gamutmap_per_tile=[False, False, False, False],
                    scale=np.ones(4),
                    mousepos=np.zeros((4, 2)),
                    img_per_tile=np.array([0, 1, 2, 3], dtype=int),
                    numtiles=1,
                ),
                config=SimpleNamespace(
                    normalize=0,
                    gamut_pow=np.ones(3) * 5.0,
                    gamut_thr=np.ones(3) * 0.8,
                    gamma=1,
                    ev=0.0,
                    cs_in=0,
                    cs_out=0,
                    debug_mode=1,
                    debug_mode_on=False,
                    gamut_lim=np.ones(3) * 1.1,
                    texture_filter="NEAREST",
                ),
            )
        renderer = GLRenderer(ui, files, loader=SimpleNamespace(), verbose=False)
        renderer.postprocess = {"kernel": SimpleNamespace(array_length=49)}
        return renderer

    def test_normalization_levels_follow_selected_mode(self):
        renderer = self._renderer(ui=SimpleNamespace(config=SimpleNamespace(normalize=3)))
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
            state=SimpleNamespace(
                ae_per_tile=[True, False, False, False],
                ae_reset_per_tile=[True, False, False, False],
            ),
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
        self.assertFalse(renderer.ui.state.ae_reset_per_tile[0])
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
            state=SimpleNamespace(
                ae_per_tile=[True, False, False, False],
                ae_reset_per_tile=[False, False, False, False],
                mirror_per_tile=[2, 0, 0, 0],
                sharpen_per_tile=[True, False, False, False],
                tonemap_per_tile=[True, False, False, False],
                gamutmap_per_tile=[True, False, False, False],
                scale=np.array([2.0, 1.0, 1.0, 1.0]),
            ),
            config=SimpleNamespace(
                normalize=0,
                gamut_pow=np.array([5.0, 6.0, 7.0]),
                gamut_thr=np.array([0.8, 0.7, 0.6]),
                gamma=3,
                ev=1.25,
                cs_in=2,
                cs_out=1,
                debug_mode=4,
                debug_mode_on=True,
                gamut_lim=np.array([1.1, 1.2, 1.3]),
            ),
        )
        renderer = self._renderer(ui=ui)
        gpu_texture = SimpleNamespace(width=100)

        with mock.patch.object(renderer, "_sharpen", return_value=np.ones((3, 3))) as sharpen_mock:
            with mock.patch.object(renderer, "_gamut", return_value=np.array([0.9, 0.8, 0.7])) as gamut_mock:
                uniforms = renderer._build_postprocess_uniforms(
                    tileidx=0,
                    imgidx=0,
                    gamma_override=None,
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

    def test_render_tile_scene_populates_first_pass_uniforms(self):
        ui = SimpleNamespace(
            state=SimpleNamespace(
                mousepos=np.array([[0.25, -0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                scale=np.array([1.5, 1.0, 1.0, 1.0]),
            ),
        )
        renderer = self._renderer(ui=ui)
        renderer.tile_target = SimpleNamespace(fbo=SimpleNamespace(use=mock.Mock(), clear=mock.Mock()))
        renderer.prog = {}
        renderer.vao = SimpleNamespace(render=mock.Mock())
        snapshot = SimpleNamespace(entries=(SimpleNamespace(linearize=True),))
        gpu_texture = SimpleNamespace(components=1)

        renderer._render_tile_scene(
            tileidx=0,
            imgidx=0,
            gpu_texture=gpu_texture,
            orientation=90,
            scalex=0.8,
            scaley=0.6,
            snapshot=snapshot,
        )

        renderer.tile_target.fbo.use.assert_called_once_with()
        renderer.tile_target.fbo.clear.assert_called_once()
        self.assertEqual(renderer.prog["img"], 0)
        self.assertEqual(renderer.prog["mousepos"], (0.25, -0.5))
        self.assertEqual(renderer.prog["scale"], 1.5)
        self.assertEqual(renderer.prog["aspect"], (0.8, 0.6))
        self.assertEqual(renderer.prog["orientation"], 90)
        self.assertTrue(renderer.prog["grayscale"])
        self.assertTrue(renderer.prog["degamma"])
        renderer.vao.render.assert_called_once()

    def test_render_postprocess_tile_applies_uniforms_and_renders(self):
        ui = SimpleNamespace(viewports={0: (1, 2, 3, 4)})
        renderer = self._renderer(ui=ui)
        renderer.tile_target = SimpleNamespace(fbo=SimpleNamespace(color_attachments=[SimpleNamespace(use=mock.Mock())]))
        renderer.postprocess = {"kernel": SimpleNamespace(array_length=9)}
        renderer.vao_post = SimpleNamespace(render=mock.Mock())
        target = SimpleNamespace(use=mock.Mock(), clear=mock.Mock(), viewport=None)

        with mock.patch.object(renderer, "_build_postprocess_uniforms", return_value={"img": 0, "gamma": 3}) as build_uniforms:
            renderer._render_postprocess_tile(
                tileidx=0,
                imgidx=5,
                target=target,
                gamma_override=1,
                vpw=10,
                vph=20,
                gpu_texture=SimpleNamespace(width=100),
                scalex=0.5,
                whitelevel=2.0,
                blacklevel=0.1,
                diffuse_white=1.5,
                peak_white=4.0,
                ae_gain=1.75,
            )

        target.use.assert_called_once_with()
        self.assertEqual(target.viewport, (1, 2, 3, 4))
        target.clear.assert_called_once_with(viewport=(1, 2, 3, 4))
        renderer.tile_target.fbo.color_attachments[0].use.assert_called_once_with(location=0)
        build_uniforms.assert_called_once_with(
            0,
            5,
            1,
            10,
            20,
            mock.ANY,
            0.5,
            2.0,
            0.1,
            1.5,
            4.0,
            1.75,
        )
        self.assertEqual(renderer.postprocess["img"], 0)
        self.assertEqual(renderer.postprocess["gamma"], 3)
        renderer.vao_post.render.assert_called_once()

    def test_build_postprocess_uniforms_prefers_gamma_override(self):
        ui = SimpleNamespace(
            state=SimpleNamespace(
                mirror_per_tile=[0, 0, 0, 0],
                sharpen_per_tile=[False, False, False, False],
                ae_per_tile=[False, False, False, False],
                tonemap_per_tile=[False, False, False, False],
                gamutmap_per_tile=[False, False, False, False],
                scale=np.ones(4),
            ),
            config=SimpleNamespace(
                gamut_pow=np.ones(3) * 5.0,
                gamut_thr=np.ones(3) * 0.8,
                gamma=3,
                ev=0.0,
                cs_in=0,
                cs_out=0,
                debug_mode=1,
                debug_mode_on=False,
                gamut_lim=np.ones(3) * 1.1,
            ),
        )
        renderer = self._renderer(ui=ui)

        with mock.patch.object(renderer, "_sharpen", return_value=np.ones((3, 3))):
            with mock.patch.object(renderer, "_gamut", return_value=np.array([1.0, 1.0, 1.0])):
                uniforms = renderer._build_postprocess_uniforms(
                    tileidx=0,
                    imgidx=0,
                    gamma_override=0,
                    vpw=200,
                    vph=100,
                    gpu_texture=SimpleNamespace(width=100),
                    scalex=0.5,
                    whitelevel=2.0,
                    blacklevel=0.1,
                    diffuse_white=1.5,
                    peak_white=4.0,
                    ae_gain=1.75,
                )

        self.assertEqual(uniforms["gamma"], 0)

    def test_screenshot_uses_gamma_override_without_mutating_ui(self):
        ui = SimpleNamespace(window=SimpleNamespace(get_size=lambda: (20, 10)), config=SimpleNamespace(gamma=3))
        renderer = GLRenderer(ui, files=SimpleNamespace(), loader=SimpleNamespace(), verbose=False)
        renderer.ctx = SimpleNamespace(
            simple_framebuffer=lambda size, components, dtype: SimpleNamespace(
                read=lambda components, dtype, clamp: bytes(20 * 10 * 3),
            ),
            screen=SimpleNamespace(use=mock.Mock()),
        )

        with mock.patch.object(renderer, "redraw") as redraw_mock:
            screenshot = renderer.screenshot(np.uint8)

        redraw_mock.assert_called_once_with(mock.ANY, 1)
        self.assertEqual(renderer.ui.config.gamma, 3)
        self.assertEqual(screenshot.shape, (10, 20, 3))


if __name__ == "__main__":
    unittest.main()
