"""Renderer-side texture upload and cache coordination."""

import numpy as np

try:
    from glview import texture
    from glview import texturecache
except ImportError:
    import texture
    import texturecache


class RenderTextureManager:
    """Own renderer texture cache, upload, and loader-release coordination."""

    def __init__(self, ctx, files, loader, verbose=False):
        self.ctx = ctx
        self.files = files
        self.loader = loader
        self.verbose = verbose
        self.cache = texturecache.TextureCache()

    def upload(self, idx: int, piecewise: bool, snapshot=None) -> texture.Texture:
        """
        Upload the image at the given index to GPU memory, either all at once
        or piecewise, to avoid freezing the user interface. Create a new texture
        object on the GPU if necessary, otherwise use an existing one.
        """
        snapshot = snapshot or self.files.snapshot()
        slot_id = snapshot.image_slots[idx].slot_id
        tex = self.cache.get(slot_id)
        img, token = self.loader.get_image_record(idx)
        if not tex:
            img = img if isinstance(img, np.ndarray) else None
            tex = texture.Texture(self.ctx, img, idx, self.verbose)
            self.cache.store(slot_id, tex)
        elif isinstance(img, np.ndarray):
            tex.reuse(img)
        if isinstance(img, np.ndarray):
            self.files.consume_image(idx, img)
            self.loader.release_image(idx, token=token)
        tex.upload(piecewise)
        return tex

    def get_cached(self, slot_id: int):
        """Return the cached texture for the given slot id, if any."""
        return self.cache.get(slot_id)

    def prune(self, snapshot):
        """Release textures whose slot ids are no longer present in the catalog."""
        active_slot_ids = {slot.slot_id for slot in snapshot.image_slots}
        self.cache.prune(active_slot_ids)

    def release_all(self):
        self.cache.release_all()
