"""Indexed catalog of image files plus their per-image state and snapshots.

`FileList` is the single source of truth for which files glview is showing,
their per-image toggles (orientation, linearize), and their loader state
(via `ImageSlot`). It is mutated from the UI thread but read from the loader
and renderer threads, so all mutations and snapshots are guarded by `mutex`.

`FileEntrySnapshot` / `FileListSnapshot` are immutable read-only views
returned by `FileList.snapshot()`; consumers should prefer them over
touching `FileList` directly to avoid races.
"""

import os
import pathlib
import threading
from dataclasses import dataclass

import numpy as np

from glview.imagestate import ImageSlot, ImageStatus


@dataclass
class FileEntry:
    filespec: str
    orientation: int
    linearize: bool
    image_slot: ImageSlot
    loaded_image: object = None
    image: object = None
    metadata: object = None
    rawmax: int = 65535  # bit-depth maximum (e.g., 1023 for 10-bit, 4095 for 12-bit, 65535 for 16-bit)

    @property
    def slot_id(self) -> int:
        return self.image_slot.slot_id

    @property
    def status(self) -> ImageStatus:
        return self.image_slot.status

    @property
    def revision(self) -> int:
        return self.image_slot.revision

    @property
    def token(self):
        return (self.slot_id, self.revision)


@dataclass(frozen=True)
class FileEntrySnapshot:
    filespec: str
    orientation: int
    linearize: bool
    image_slot: ImageSlot
    metadata: object = None
    rawmax: int = 65535  # bit-depth maximum from the loader (e.g., 1023 for 10-bit RAW)

    @classmethod
    def from_entry(cls, entry: FileEntry):
        return cls(
            filespec=entry.filespec,
            orientation=entry.orientation,
            linearize=entry.linearize,
            image_slot=ImageSlot(
                slot_id=entry.slot_id,
                status=entry.status,
                revision=entry.revision,
            ),
            metadata=entry.metadata,
            rawmax=entry.rawmax,
        )

    @property
    def slot_id(self) -> int:
        return self.image_slot.slot_id

    @property
    def status(self) -> ImageStatus:
        return self.image_slot.status

    @property
    def revision(self) -> int:
        return self.image_slot.revision

    @property
    def token(self):
        return (self.slot_id, self.revision)

    @property
    def auto_linearize(self) -> bool:
        """True for file types that are gamma-encoded and need sRGB degamma before display."""
        return pathlib.Path(self.filespec).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


@dataclass(frozen=True)
class FileListSnapshot:
    entries: tuple
    numfiles: int


class FileList:
    """ An indexed container for images and their source filenames. """

    def __init__(self, filespecs):
        """
        Create a new FileList with the given list of filenames. Files can only
        be removed from the list, not added or reordered.
        """
        self.mutex = threading.Lock()
        self._next_slot_id = 0
        self.entries = [
            FileEntry(
                filespec=filespec,
                orientation=0,
                linearize=False,
                image_slot=self._new_slot(),
            )
            for filespec in filespecs
        ]
        self._update()
        self.reindexed = False

    def ready_to_upload(self, idx):
        """
        Return True if the given image is ready to be uploaded to OpenGL,
        or has already been uploaded.
        """
        return self.image_status(idx) not in [ImageStatus.PENDING, ImageStatus.INVALID]

    def entry(self, idx) -> FileEntry:
        return self.entries[idx]

    def image_status(self, idx) -> ImageStatus:
        return self.entries[idx].status

    def image_revision(self, idx) -> int:
        return self.entries[idx].revision

    def image_slot_id(self, idx) -> int:
        return self.entries[idx].slot_id

    def image_token(self, idx):
        return self.entries[idx].token

    def get_loaded_image(self, idx):
        return self.entries[idx].loaded_image

    def get_consumed_image(self, idx):
        return self.entries[idx].image

    def filespec(self, idx) -> str:
        return self.entries[idx].filespec

    def set_linearize(self, idx, linearize: bool):
        self.entries[idx].linearize = linearize

    def toggle_linearize(self, idx) -> bool:
        entry = self.entries[idx]
        entry.linearize = not entry.linearize
        return entry.linearize

    def rotate_orientation(self, idx, degrees: int = 90) -> int:
        entry = self.entries[idx]
        entry.orientation = (entry.orientation + degrees) % 360
        return entry.orientation

    def mark_pending(self, idx):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.PENDING
        entry.image_slot.revision += 1
        entry.loaded_image = None
        entry.image = None

    def mark_loaded(self, idx, img, rawmax=65535):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.LOADED
        entry.loaded_image = img
        entry.rawmax = rawmax

    def mark_released(self, idx):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.RELEASED
        entry.loaded_image = None

    def mark_invalid(self, idx):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.INVALID
        entry.loaded_image = None
        entry.image = None

    def consume_image(self, idx, img):
        self.entries[idx].image = img

    def clear_consumed_image(self, idx):
        self.entries[idx].image = None

    def snapshot(self) -> FileListSnapshot:
        """Return a consistent read-only view of the catalog state."""
        with self.mutex:
            entries = tuple(FileEntrySnapshot.from_entry(entry) for entry in self.entries)
            return FileListSnapshot(
                entries=entries,
                numfiles=self.numfiles,
            )

    def drop(self, indices):
        """ Drop the given images from this FileList, do not delete the files. """
        with self.mutex:
            try:
                self.entries = self._drop(self.entries, indices)
                self._update()
                print(f"[FileList] Dropped images {indices}")
            except IndexError:
                pass

    def delete(self, idx):
        """ Remove the given image from this FileList and delete the file from disk. """
        with self.mutex:
            try:
                filespec = self.entries[idx].filespec
                self.entries = self._drop(self.entries, [idx])
                self._update()
                os.remove(filespec)
                print(f"[FileList] Deleted {filespec}")
            except IndexError:
                pass

    def _drop(self, arr, indices):
        arr = np.asarray(arr, dtype=object)
        arr = np.delete(arr, indices)
        arr = arr.tolist()
        return arr

    def _update(self):
        self.numfiles = len(self.entries)
        self.reindexed = True

    def _new_slot(self):
        slot = ImageSlot(slot_id=self._next_slot_id)
        self._next_slot_id += 1
        return slot
