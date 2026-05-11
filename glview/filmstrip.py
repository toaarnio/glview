"""Toggleable thumbnail filmstrip overlay for glview."""

from dataclasses import dataclass

import numpy as np

from glview.imagestate import ImageStatus


_FONT = ["Consolas", "Menlo", "DejaVu Sans Mono", "monospace"]
_FONT_SIZE = 10
_BG_COLOR = (0, 0, 0, 200)
_CELL_BG_COLOR = (40, 40, 40, 255)
_PLACEHOLDER_COLOR = (90, 90, 90, 255)
_PLACEHOLDER_INVALID_COLOR = (110, 50, 50, 255)
_LABEL_COLOR = (220, 220, 220, 255)
_ACTIVE_BORDER_COLOR = (255, 196, 0, 255)
_TILE_BORDER_COLOR = (140, 140, 200, 255)


@dataclass(frozen=True)
class _CellLayout:
    imgidx: int
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px < self.x + self.w) and (self.y <= py < self.y + self.h)


def compute_layout(numfiles: int, winsize: tuple[int, int], strip_height: int,
                   cell_size: int, cell_pad: int, center_imgidx: int) -> tuple[tuple[_CellLayout, ...], int, int]:
    """Compute filmstrip cell rects, the strip y origin, and the strip height.

    Pure function: no Pyglet/GL dependencies. Auto-centers the strip horizontally
    on `center_imgidx` and clamps so the strip stays within the window when the
    total cell width exceeds the window width.
    """
    if numfiles <= 0:
        return ((), 0, strip_height)
    win_w, _ = winsize
    stride = cell_size + cell_pad
    total_w = numfiles * stride - cell_pad
    if total_w <= win_w:
        # Centered as a block when everything fits.
        first_x = (win_w - total_w) // 2
    else:
        center_imgidx = max(0, min(numfiles - 1, center_imgidx))
        center_cell_mid = center_imgidx * stride + cell_size // 2
        first_x = win_w // 2 - center_cell_mid
        # Clamp so we don't reveal empty space at either edge.
        min_first_x = win_w - total_w
        first_x = max(min_first_x, min(0, first_x))
    cell_y = (strip_height - cell_size) // 2
    cells = tuple(
        _CellLayout(imgidx=i, x=first_x + i * stride, y=cell_y, w=cell_size, h=cell_size)
        for i in range(numfiles)
    )
    return cells, 0, strip_height


def hit_test_cells(cells: tuple[_CellLayout, ...], strip_y: int, strip_h: int,
                   px: float, py: float) -> int | None:
    """Return the imgidx whose cell contains (px, py), or None."""
    if not cells:
        return None
    if not (strip_y <= py < strip_y + strip_h):
        return None
    for cell in cells:
        if cell.contains(px, py):
            return cell.imgidx
    return None


def _letterbox_to_cell(thumb_rgba: np.ndarray, cell_size: int, bg_rgba=(40, 40, 40, 255)) -> np.ndarray:
    """Place an arbitrary-aspect RGBA thumbnail into a square cell with letterboxing."""
    th, tw = thumb_rgba.shape[:2]
    if th == 0 or tw == 0:
        canvas = np.zeros((cell_size, cell_size, 4), dtype=np.uint8)
        canvas[:] = bg_rgba
        return canvas
    scale = min(cell_size / tw, cell_size / th)
    new_w = max(1, round(tw * scale))
    new_h = max(1, round(th * scale))
    # Nearest-neighbor downsample via slicing (cheap & dependency-free).
    ys = (np.arange(new_h) * th / new_h).astype(np.int64)
    xs = (np.arange(new_w) * tw / new_w).astype(np.int64)
    resized = thumb_rgba[ys][:, xs]
    canvas = np.zeros((cell_size, cell_size, 4), dtype=np.uint8)
    canvas[:] = bg_rgba
    oy = (cell_size - new_h) // 2
    ox = (cell_size - new_w) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def _to_rgba_uint8(img: np.ndarray, normalize_max: float = 1.0) -> np.ndarray:
    """Convert a CPU image (any of the supported dtypes) to RGBA uint8 for display."""
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    h, w, c = img.shape
    if img.dtype == np.uint8:
        flt = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        flt = img.astype(np.float32) / 65535.0
    else:
        flt = img.astype(np.float32)
        if normalize_max > 0:
            flt = flt / max(normalize_max, 1e-6)
    flt = np.clip(flt, 0.0, 1.0)
    if c == 1:
        flt = np.repeat(flt, 3, axis=2)
    elif c >= 3:
        flt = flt[:, :, :3]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = (flt * 255.0 + 0.5).astype(np.uint8)
    out[:, :, 3] = 255
    return out


class Filmstrip:
    """A toggleable bottom-of-window thumbnail strip with click-to-jump navigation."""

    def __init__(self, strip_height: int = 96, cell_size: int = 80, cell_pad: int = 4):
        self.visible = False
        self.strip_height = strip_height
        self.cell_size = cell_size
        self.cell_pad = cell_pad
        self._cells: tuple[_CellLayout, ...] = ()
        self._strip_y = 0
        self._strip_h = strip_height
        self._winsize = (0, 0)
        self._active_imgidx: int | None = None
        self._tile_imgidxs: tuple[int, ...] = ()
        # Thumbnail cache keyed by (slot_id, revision); value is a uint8 RGBA array
        # of shape (cell_size, cell_size, 4), or None to mark "invalid -> placeholder".
        self._thumb_cache: dict[tuple[int, int], np.ndarray | None] = {}
        # Pyglet draw resources, lazily initialized.
        self._bg_batch = None
        self._fg_batch = None
        self._sprites = []
        self._shapes = []
        self._labels = []

    def toggle(self):
        self.visible = not self.visible

    def strip_rect(self, winsize: tuple[int, int]) -> tuple[int, int, int, int]:
        return (0, 0, winsize[0], self.strip_height)

    def hit_test(self, px: float, py: float) -> int | None:
        return hit_test_cells(self._cells, self._strip_y, self._strip_h, px, py)

    def update(self, winsize, state, snapshot, files):
        """Rebuild layout, refresh draw resources. Cache is filled by `preload`."""
        self._winsize = winsize
        if not self.visible:
            return
        numfiles = snapshot.numfiles
        active_imgidx = int(state.img_per_tile[state.tileidx]) if numfiles > 0 else 0
        cells, strip_y, strip_h = compute_layout(
            numfiles, winsize, self.strip_height, self.cell_size, self.cell_pad, active_imgidx,
        )
        self._cells = cells
        self._strip_y = strip_y
        self._strip_h = strip_h
        self._active_imgidx = active_imgidx if numfiles > 0 else None
        self._tile_imgidxs = tuple(int(state.img_per_tile[t]) for t in range(state.numtiles))
        self.preload(snapshot, files)
        self._rebuild_draw_resources(snapshot)

    def preload(self, snapshot, files):
        """
        Eagerly populate the thumbnail cache for every image whose CPU array is
        currently in RAM. Safe to call every UI tick regardless of visibility:
        once a thumb is cached we never need the source pixels again, so we can
        capture them before the renderer's `upload()` releases the CPU copy.
        """
        for imgidx in range(snapshot.numfiles):
            entry = snapshot.entries[imgidx]
            key = (entry.slot_id, entry.revision)
            if key in self._thumb_cache:
                continue
            if entry.status == ImageStatus.INVALID:
                self._thumb_cache[key] = None
                continue
            thumb = self._build_thumb(imgidx, files)
            if thumb is not None:
                self._thumb_cache[key] = thumb
        self._invalidate_stale_cache(snapshot)

    def draw(self):
        if not self.visible or self._bg_batch is None:
            return
        self._bg_batch.draw()
        for sprite in self._sprites:
            sprite.draw()
        self._fg_batch.draw()

    def release(self):
        """Free Pyglet/GL resources held by the filmstrip."""
        self._sprites = []
        self._shapes = []
        self._labels = []
        self._bg_batch = None
        self._fg_batch = None
        self._thumb_cache.clear()

    # ------------------------------------------------------------------

    def _build_thumb(self, imgidx, files):
        if files is None:
            return None
        try:
            img = files.get_consumed_image(imgidx)
        except Exception:
            return None
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None
        stride = max(1, min(img.shape[:2]) // max(1, self.cell_size))
        small = img[::stride, ::stride]
        rgba = _to_rgba_uint8(small)
        return _letterbox_to_cell(rgba, self.cell_size)

    def _invalidate_stale_cache(self, snapshot):
        live_keys = {(e.slot_id, e.revision) for e in snapshot.entries}
        for stale in [k for k in self._thumb_cache if k not in live_keys]:
            del self._thumb_cache[stale]

    def _rebuild_draw_resources(self, snapshot):
        # Deferred imports: importing pyglet at module load can fail in headless test envs.
        import pyglet  # noqa: PLC0415
        from pyglet import shapes as _shapes  # noqa: PLC0415

        self._bg_batch = pyglet.graphics.Batch()
        self._fg_batch = pyglet.graphics.Batch()
        self._sprites = []
        self._shapes = []
        self._labels = []

        win_w, _ = self._winsize
        # Background bar across the full strip width.
        bg = _shapes.Rectangle(0, self._strip_y, win_w, self._strip_h, color=_BG_COLOR, batch=self._bg_batch)
        self._shapes.append(bg)

        for cell in self._cells:
            entry = snapshot.entries[cell.imgidx]
            key = (entry.slot_id, entry.revision)
            thumb = self._thumb_cache.get(key, None)
            self._draw_cell(pyglet, _shapes, cell, entry, thumb)

        # Borders for active tile (last so it draws on top).
        for tileidx, imgidx in enumerate(self._tile_imgidxs):
            cell = self._find_cell(imgidx)
            if cell is None:
                continue
            is_active = (imgidx == self._active_imgidx and tileidx == self._tile_active_index())
            color = _ACTIVE_BORDER_COLOR if is_active else _TILE_BORDER_COLOR
            thickness = 3 if is_active else 2
            self._add_border(_shapes, cell, color, thickness)

    def _tile_active_index(self) -> int:
        return self._tile_imgidxs.index(self._active_imgidx) if self._active_imgidx in self._tile_imgidxs else -1

    def _find_cell(self, imgidx: int) -> _CellLayout | None:
        for cell in self._cells:
            if cell.imgidx == imgidx:
                return cell
        return None

    def _draw_cell(self, pyglet, shapes, cell, entry, thumb):
        if thumb is None:
            color = _PLACEHOLDER_INVALID_COLOR if entry.status == ImageStatus.INVALID else _PLACEHOLDER_COLOR
            rect = shapes.Rectangle(cell.x, cell.y, cell.w, cell.h, color=color, batch=self._bg_batch)
            self._shapes.append(rect)
            label = pyglet.text.Label(
                str(cell.imgidx + 1),
                font_name=_FONT, font_size=_FONT_SIZE,
                x=cell.x + cell.w // 2, y=cell.y + cell.h // 2,
                anchor_x="center", anchor_y="center",
                color=_LABEL_COLOR, batch=self._fg_batch,
            )
            self._labels.append(label)
            return
        # Pyglet expects bottom-up rows; flip vertically once.
        flipped = np.ascontiguousarray(thumb[::-1])
        image_data = pyglet.image.ImageData(
            cell.w, cell.h, "RGBA", flipped.tobytes(), pitch=cell.w * 4,
        )
        sprite = pyglet.sprite.Sprite(image_data, x=cell.x, y=cell.y)
        self._sprites.append(sprite)

    def _add_border(self, shapes, cell, color, thickness: int):
        x, y, w, h = cell.x, cell.y, cell.w, cell.h
        sides = [
            (x, y, w, thickness),                      # bottom
            (x, y + h - thickness, w, thickness),      # top
            (x, y, thickness, h),                      # left
            (x + w - thickness, y, thickness, h),      # right
        ]
        for sx, sy, sw, sh in sides:
            self._shapes.append(shapes.Rectangle(sx, sy, sw, sh, color=color, batch=self._fg_batch))
