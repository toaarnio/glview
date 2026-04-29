"""Renderer-owned cache for GPU textures keyed by stable image slot ids."""


class TextureCache:

    def __init__(self):
        self._cache = {}

    def get(self, slot_id):
        return self._cache.get(slot_id)

    def store(self, slot_id, tex):
        self._cache[slot_id] = tex
        return tex

    def prune(self, active_slot_ids):
        stale_slot_ids = [slot_id for slot_id in self._cache if slot_id not in active_slot_ids]
        for slot_id in stale_slot_ids:
            self._cache.pop(slot_id).release()

    def release_all(self):
        for tex in self._cache.values():
            tex.release()
        self._cache.clear()

    def keys(self):
        return self._cache.keys()
