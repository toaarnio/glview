"""Renderer-side offscreen target lifecycle management."""


class TileRenderTarget:
    """Own the offscreen framebuffer used for per-tile first-pass rendering."""

    def __init__(self, ctx, filters):
        self.ctx = ctx
        self.filters = filters
        self.fbo = None

    def ensure(self, vpw: int, vph: int):
        """Create or resize the offscreen tile framebuffer as needed."""
        if self.fbo is None or self.fbo.size != (vpw, vph):
            offscreen_tile = self.ctx.texture((vpw, vph), components=3, dtype="f4")
            offscreen_tile.repeat_x = False
            offscreen_tile.repeat_y = False
            offscreen_tile.filter = self.filters["NEAREST"]
            self.fbo = self.ctx.framebuffer([offscreen_tile])
        return self.fbo

    def release(self):
        """Release the offscreen framebuffer if present."""
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
