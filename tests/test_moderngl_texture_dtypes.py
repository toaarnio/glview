"""
Round-trip tests for ModernGL texture dtypes used by glview.

Each test uploads a known pixel value into a 1x1 texture, renders it to
a float32 framebuffer via a passthrough shader, and verifies the sampled
value matches what was written.

These tests require a GPU-capable environment.  They are skipped automatically
when a standalone OpenGL context cannot be created (e.g. CI without a GPU).
"""

import unittest

import numpy as np

try:
    import moderngl
    _ctx = moderngl.create_standalone_context()
except Exception:  # noqa: BLE001
    _ctx = None


_VS = """
#version 310 es
in vec2 vert;
out vec2 tc;
void main() {
    gl_Position = vec4(vert, 0.0, 1.0);
    tc = vert * 0.5 + 0.5;
}
"""

_FS_FLOAT = """
#version 310 es
precision highp float;
uniform sampler2D img;
in vec2 tc;
out vec4 color;
void main() { color = texture(img, tc); }
"""

_FS_UINT = """
#version 310 es
precision mediump float;
precision mediump usampler2D;
uniform usampler2D img;
in vec2 tc;
out vec4 color;
void main() { uvec4 s = texture(img, tc); color = vec4(s) / 65535.0; }
"""

_QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)


def _render(ctx, tex, fs_src) -> np.ndarray:
    """Render a 1x1 texture to a float32 FBO and return the RGB pixel as float32."""
    prog = ctx.program(vertex_shader=_VS, fragment_shader=fs_src)
    vbo = ctx.buffer(_QUAD)
    vao = ctx.simple_vertex_array(prog, vbo, "vert")
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((1, 1), 4, dtype="f4")])
    fbo.use()
    prog["img"] = 0
    tex.use(0)
    vao.render(moderngl.TRIANGLE_STRIP)
    raw = fbo.read(components=3, dtype="f4")
    return np.frombuffer(raw, dtype=np.float32).copy()


@unittest.skipIf(_ctx is None, "no OpenGL context available")
class TextureDtypeRoundTripTests(unittest.TestCase):
    """Verify that each texture dtype actually returns correct values in a shader."""

    def _roundtrip(self, tex, expected, fs_src=_FS_FLOAT, atol=1e-4):
        out = _render(_ctx, tex, fs_src)
        np.testing.assert_allclose(out, expected, atol=atol,
                                   err_msg=f"expected {expected}, got {out}")

    def test_f1_uint8_fixed_point(self):
        """f1 (GL_RGB8, fixed-point): the standard non-linearized path."""
        data = np.array([200, 100, 50], dtype=np.uint8)
        tex = _ctx.texture((1, 1), 3, data, dtype="f1")
        expected = np.float16(data / 255.0).astype(np.float32)
        self._roundtrip(tex, expected, atol=1e-4)

    def test_f2_float16_linearized_uint8(self):
        """f2 (GL_RGB16F): used for linearized uint8 images (JPEG/PNG 8-bit)."""
        data = np.array([0.5, 0.25, 0.125], dtype=np.float16)
        tex = _ctx.texture((1, 1), 3, data.tobytes(), dtype="f2")
        self._roundtrip(tex, data.astype(np.float32), atol=1e-4)

    def test_f4_float32_linearized_uint16(self):
        """f4 (GL_RGB32F): used for linearized uint16 PNGs and HDR images."""
        data = np.array([0.5, 0.25, 0.125], dtype=np.float32)
        tex = _ctx.texture((1, 1), 3, data.tobytes(), dtype="f4")
        self._roundtrip(tex, data, atol=1e-6)

    def test_u2_uint16_integer_returns_zero(self):
        """
        u2 (GL_RGB16UI): still broken in ModernGL — returns constant zero when
        sampled in a fragment shader.  This test documents the known limitation
        and will fail if the bug is ever fixed, signalling that the uint16→float32
        conversion in texture.py can be removed.
        """
        data = np.array([32768, 16384, 8192], dtype=np.uint16)
        tex = _ctx.texture((1, 1), 3, data.tobytes(), dtype="u2")
        out = _render(_ctx, tex, _FS_UINT)
        self.assertTrue(
            np.all(out == 0.0),
            msg=f"u2 texture now returns non-zero ({out}): the uint16 workaround in "
                "texture.py may no longer be necessary — consider removing it.",
        )

    def test_nu2_uint16_normalized_works_for_linear_data(self):
        """
        nu2 (GL_RGB16, normalized): works correctly with sampler2D for linear
        uint16 data (e.g. linear RAW sensor output). NOT suitable for sRGB uint16
        PNGs: GPU-side mipmap construction averages gamma-encoded values, producing
        incorrect (too dark) results. Degamma must happen before upload.
        """
        data = np.array([32768, 16384, 8192], dtype=np.uint16)
        tex = _ctx.texture((1, 1), 3, data.tobytes(), dtype="nu2")
        self._roundtrip(tex, data / 65535.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
