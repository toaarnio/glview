"""On-screen HUD overlay for glview."""

from pathlib import Path

import numpy as np
import pyglet


_FONT = ["Consolas", "Menlo", "DejaVu Sans Mono", "monospace"]
_FONT_SIZE = 11
_PAD = 8        # gap between label and viewport edge
_BG_PAD = 4     # padding inside the dark background box
_COLOR = (230, 230, 230, 255)
_BG_COLOR = (0, 0, 0, 180)

_DTYPE_SHORT = {
    "float16": "f16",
    "float32": "f32",
    "uint8":   "u8",
    "uint16":  "u16",
}

_CSPACES = ["sRGB", "DCI-P3", "Rec2020", "XYZ"]
_GAMMAS  = ["off", "sRGB", "HLG", "HDR10"]
_NORMS   = ["off", "max", "stretch", "99.5%", "98%", "95%", "90%", "mean"]


class HUD:
    """
    A toggleable on-screen overlay that shows viewer state using Pyglet text labels.
    Drawn after the ModernGL render passes, before window.flip().
    """

    def __init__(self):
        self.visible = True
        self._bg_batch = pyglet.graphics.Batch()
        self._text_batch = pyglet.graphics.Batch()
        self._labels = []
        self._boxes = []

    def toggle(self):
        self.visible = not self.visible

    def update(self, _winsize, state, config, snapshot, viewports, textures):
        """Rebuild labels from current viewer state. No-op when hidden."""
        if not self.visible:
            return
        self._labels.clear()
        self._boxes.clear()
        self._bg_batch = pyglet.graphics.Batch()
        self._text_batch = pyglet.graphics.Batch()

        # Per-tile labels: top-left corner of each viewport
        for tileidx in range(state.numtiles):
            imgidx = state.img_per_tile[tileidx]
            vx, vy, vw, vh = viewports[tileidx]
            x = vx + _PAD
            y = vy + vh - _PAD
            text = self._tile_label(tileidx, imgidx, state, snapshot, textures)
            self._make_label(text, x, y, anchor_y="top", max_width=vw - 2 * _PAD)

        # Global status bar: bottom-left of the window
        self._make_label(self._status_bar(state, config), _PAD, _PAD, anchor_y="bottom")

    def draw(self):
        """Draw all HUD labels. Must be called with the screen framebuffer active."""
        self._bg_batch.draw()
        self._text_batch.draw()

    # ------------------------------------------------------------------

    def _tile_label(self, tileidx, imgidx, state, snapshot, textures):
        if snapshot.numfiles == 0:
            return ""
        entry = snapshot.entries[imgidx]
        name = Path(entry.filespec).name
        index_str = f"[{imgidx + 1}/{snapshot.numfiles}]"
        zoom_pct = state.scale[tileidx] * 100.0

        slot_id = entry.slot_id
        tex = textures.get_cached(slot_id)
        if tex is not None and tex.upload_done:
            tw, th = tex.texture.width, tex.texture.height
            if entry.orientation in [90, 270]:
                tw, th = th, tw
            dtype_str = _DTYPE_SHORT.get(np.dtype(tex.dtype).name, np.dtype(tex.dtype).name)
            dims = f"{tw}x{th} {dtype_str}"
        else:
            dims = "loading..."

        ae = "Y" if state.ae_per_tile[tileidx] else "N"
        tm = "Y" if state.tonemap_per_tile[tileidx] else "N"
        gm = "Y" if state.gamutmap_per_tile[tileidx] else "N"
        sh = "Y" if state.sharpen_per_tile[tileidx] else "N"
        line1 = f"{index_str} {name} {dims} {zoom_pct:.0f}%"
        line2 = f"ae:{ae} tmap:{tm} gmap:{gm} sharpen:{sh}"
        return f"{line1}\n{line2}"

    def _status_bar(self, state, config):
        def flags(per_tile):
            return "".join("Y" if per_tile[i] else "N" for i in range(state.numtiles))

        csc   = f"{_CSPACES[config.cs_in]} => {_CSPACES[config.cs_out]}"
        gamma = _GAMMAS[config.gamma]
        norm  = _NORMS[config.normalize]
        debug = f"| debug {config.debug_mode}" if config.debug_mode_on else ""
        return f"{config.ev:+.2f}EV | {csc} | gamma {gamma} | norm {norm} {debug}"

    def _make_label(self, text, x, y, anchor_y="baseline", max_width=None):
        label = pyglet.text.Label(
            text,
            font_name=_FONT,
            font_size=_FONT_SIZE,
            x=x,
            y=y,
            width=max_width,
            multiline=max_width is not None,
            anchor_x="left",
            anchor_y=anchor_y,
            color=_COLOR,
            batch=self._text_batch,
        )
        self._labels.append(label)

        # Size the background box to snugly fit the label, with _BG_PAD on all sides.
        from pyglet import shapes as _shapes  # deferred: importing shapes triggers pyglet.gl on Windows  # noqa: PLC0415
        cw, ch = label.content_width, label.content_height
        bx = x - _BG_PAD
        by = (y - ch - _BG_PAD) if anchor_y == "top" else (y - _BG_PAD)
        box = _shapes.Rectangle(
            bx, by,
            cw + 2 * _BG_PAD,
            ch + 2 * _BG_PAD,
            color=_BG_COLOR,
            batch=self._bg_batch,
        )
        self._boxes.append(box)
