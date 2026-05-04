"""Renderer-side offscreen target lifecycle management."""


class TileRenderTarget:
    """Own the offscreen framebuffer used for per-tile first-pass rendering."""

    def __init__(self, ctx, filters):
        self.ctx = ctx
        self.filters = filters
        self.fbo = None
        self.texture = None

    def ensure(self, vpw: int, vph: int):
        """Create or resize the offscreen tile framebuffer as needed."""
        if self.fbo is None or self.fbo.size != (vpw, vph):
            self.release()
            self.texture = self.ctx.texture((vpw, vph), components=3, dtype="f4")
            self.texture.repeat_x = False
            self.texture.repeat_y = False
            self.texture.filter = self.filters["NEAREST"]
            self.fbo = self.ctx.framebuffer([self.texture])
        return self.fbo

    def release(self):
        """Release the offscreen framebuffer if present."""
        if self.fbo is not None:
            self.fbo.release()
            self.fbo = None
        if self.texture is not None:
            self.texture.release()
            self.texture = None
