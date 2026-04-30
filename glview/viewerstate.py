"""Viewer-owned state for tiling, navigation, and per-tile display toggles."""

from dataclasses import dataclass, field

import numpy as np

from glview import uistate


@dataclass
class ViewerState:
    numtiles: int = 1
    layout: str = "N x 1"
    tileidx: int = 0
    scale: np.ndarray = field(default_factory=lambda: np.ones(4))
    mousepos: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    img_per_tile: np.ndarray = field(default_factory=lambda: np.array([0, 1, 2, 3], dtype=int))
    ae_per_tile: list[bool] = field(default_factory=lambda: [False, False, False, False])
    ae_reset_per_tile: list[bool] = field(default_factory=lambda: [False, False, False, False])
    tonemap_per_tile: list[bool] = field(default_factory=lambda: [False, False, False, False])
    gamutmap_per_tile: list[bool] = field(default_factory=lambda: [False, False, False, False])
    sharpen_per_tile: list[bool] = field(default_factory=lambda: [False, False, False, False])
    mirror_per_tile: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def reset_view(self):
        self.scale = np.ones(4)
        self.mousepos = np.zeros((4, 2))
        self.reset_ae()

    def split_state(self) -> uistate.SplitState:
        return uistate.SplitState(
            numtiles=self.numtiles,
            layout=self.layout,
            tileidx=self.tileidx,
            img_per_tile=np.asarray(self.img_per_tile),
        )

    def cycle_split(self, numfiles: int):
        state = uistate.cycle_split_state(self.split_state(), numfiles)
        self.numtiles = state.numtiles
        self.layout = state.layout
        self.tileidx = state.tileidx
        self.img_per_tile = state.img_per_tile

    def flip_pair(self):
        self.img_per_tile = np.asarray(uistate.flip_pair(self.img_per_tile), dtype=int)
        self.ae_per_tile = uistate.flip_pair(self.ae_per_tile)
        self.ae_reset_per_tile[:2] = [True, True]
        self.tonemap_per_tile = uistate.flip_pair(self.tonemap_per_tile)
        self.sharpen_per_tile = uistate.flip_pair(self.sharpen_per_tile)

    def reset_ae(self):
        self.ae_reset_per_tile = [True, True, True, True]

    def toggle_ae(self):
        self.ae_per_tile[self.tileidx] = not self.ae_per_tile[self.tileidx]
        self.ae_reset_per_tile[self.tileidx] = True

    def toggle_tonemap(self):
        self.tonemap_per_tile[self.tileidx] = not self.tonemap_per_tile[self.tileidx]
        self.ae_reset_per_tile[self.tileidx] = True

    def toggle_gamutmap(self):
        self.gamutmap_per_tile[self.tileidx] = not self.gamutmap_per_tile[self.tileidx]

    def toggle_sharpen(self):
        self.sharpen_per_tile[self.tileidx] = not self.sharpen_per_tile[self.tileidx]

    def cycle_mirror(self):
        self.mirror_per_tile[self.tileidx] = (self.mirror_per_tile[self.tileidx] + 1) % 4

    def repair_after_removal(self, numfiles: int):
        self.img_per_tile = uistate.repair_visible_images_after_removal(self.img_per_tile, self.numtiles, numfiles)

    def select_tile(self, tileidx: int):
        if tileidx < self.numtiles:
            self.tileidx = tileidx

    def step_active_tile(self, incr: int, numfiles: int):
        self.img_per_tile = uistate.step_active_tile(self.img_per_tile, self.tileidx, incr, numfiles)
        self.ae_reset_per_tile[self.tileidx] = True

    def step_all_tiles(self, incr: int, numfiles: int):
        self.img_per_tile = uistate.step_all_tiles(self.img_per_tile, self.numtiles, incr, numfiles)
        self.reset_ae()

    def keyboard_pan_zoom(self, zoom_steps: tuple[int, int], pan_steps: tuple[int, int], pan_speed: float, canvas_width: float):
        prev_scale = self.scale.copy()
        key_zoom_in, key_zoom_out = zoom_steps
        self.scale *= 1.0 + 0.1 * key_zoom_in
        self.scale /= 1.0 + 0.1 * key_zoom_out
        dxdy = np.tile(pan_steps, (4, 1))
        dxdy = dxdy * pan_speed
        dxdy = dxdy / self.scale[:, np.newaxis]
        dxdy = dxdy / canvas_width
        self.mousepos = np.clip(self.mousepos + dxdy, -1.0, 1.0)
        return np.any(dxdy != 0.0) or np.any(self.scale != prev_scale)

    def drag_mouse(self, dx: float, dy: float, pan_speed: float, canvas_width: float, active_only: bool):
        tidx = self.tileidx if active_only else np.s_[:]
        dxdy = np.tile((dx, dy), (4, 1))
        dxdy = dxdy * pan_speed
        dxdy = dxdy / self.scale[tidx, np.newaxis]
        dxdy = dxdy / canvas_width
        mousepos = np.clip(self.mousepos[tidx] + dxdy[tidx], -1.0, 1.0)
        self.mousepos[tidx] = mousepos

    def scroll_zoom(self, scroll_y: float, active_only: bool):
        tidx = self.tileidx if active_only else np.s_[:]
        scale_factor = 1.0 + 0.1 * scroll_y
        self.scale[tidx] *= scale_factor
