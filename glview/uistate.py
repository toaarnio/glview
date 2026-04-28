"""Pure UI state transition helpers for tile and image navigation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitState:
    numtiles: int
    layout: str
    tileidx: int
    img_per_tile: np.ndarray


def cycle_split_state(state: SplitState, numfiles: int) -> SplitState:
    """Advance to the next split/layout state used by the UI."""
    if state.numtiles == 4 and state.layout == "N x 1":
        return SplitState(
            numtiles=state.numtiles,
            layout="2 x 2",
            tileidx=state.tileidx,
            img_per_tile=state.img_per_tile.copy(),
        )
    if state.numtiles == 2 and state.layout == "N x 1":
        return SplitState(
            numtiles=state.numtiles,
            layout="1 x N",
            tileidx=state.tileidx,
            img_per_tile=state.img_per_tile.copy(),
        )
    img_per_tile = np.clip(state.img_per_tile, 0, numfiles - 1)
    numtiles = (state.numtiles % 4) + 1
    tileidx = min(state.tileidx, numtiles - 1)
    return SplitState(
        numtiles=numtiles,
        layout="N x 1",
        tileidx=tileidx,
        img_per_tile=img_per_tile,
    )


def flip_pair(values):
    """Reverse the first two entries of a per-tile state vector."""
    flipped = list(values)
    flipped[:2] = flipped[:2][::-1]
    return flipped


def step_active_tile(img_per_tile, tileidx: int, incr: int, numfiles: int):
    """Move the active tile's image index by one step with wraparound."""
    result = np.array(img_per_tile, copy=True)
    result[tileidx] = (result[tileidx] + incr) % numfiles
    return result


def step_all_tiles(img_per_tile, numtiles: int, incr: int, numfiles: int):
    """Move all visible tiles, preserving stride when they are consecutive."""
    result = np.array(img_per_tile, copy=True)
    active_tiles = result[:numtiles]
    consecutive = np.ptp(active_tiles) + 1 == numtiles
    stride = numtiles if consecutive else 1
    result[:numtiles] = (active_tiles + incr * stride) % numfiles
    return result


def repair_visible_images_after_removal(img_per_tile, numtiles: int, numfiles: int):
    """Shift currently visible indices after drop/delete to keep the view filled."""
    result = np.array(img_per_tile, copy=True)
    visible_images = result[:numtiles] - numtiles
    result[:numtiles] = visible_images % numfiles
    return result
