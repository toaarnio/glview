#!/usr/bin/python3 -B

"""
The 'glview' command-line application.
"""

import sys                     # built-in library
import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
import pathlib                 # built-in library
import types                   # built-in library
from dataclasses import dataclass  # built-in library
from collections.abc import MutableSequence  # built-in library
import numpy as np             # pip install numpy
import natsort                 # pip install natsort
import psutil                  # pip install psutil
import imgio                   # pip install imgio

from glview import argv
from glview import glrenderer
from glview import imageprovider
from glview import pygletui
from glview import version
from glview.imagestate import ImageSlot, ImageStatus


IMAGE_TYPES = imgio.RO_FORMATS

COLORSPACES = {"sRGB": 0, "P3": 1, "Rec2020": 2, "XYZ": 3}

NORMS = {"off": 0, "max": 1, "stretch": 2, "99": 3, "98": 4, "95": 5, "90": 6, "mean": 7}


@dataclass
class FileEntry:
    filespec: str
    orientation: int
    linearize: bool
    image_slot: ImageSlot
    loaded_image: object = None
    image: object = None
    metadata: object = None


@dataclass(frozen=True)
class FileListSnapshot:
    entries: tuple
    filespecs: tuple
    numfiles: int
    orientations: tuple
    linearize: tuple
    image_slots: tuple
    metadata: tuple


class FileListFieldLengthError(ValueError):
    """Assigned FileList field data must match the number of entries."""


class ImmutableFieldViewError(TypeError):
    """FileList field views do not support structural edits."""


class _EntryFieldView(MutableSequence):
    """List-like view over a specific field of FileList entries."""

    __hash__ = None

    def __init__(self, file_list, field_name):
        self._file_list = file_list
        self._field_name = field_name

    @property
    def _entries(self):
        return self._file_list.entries

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [getattr(entry, self._field_name) for entry in self._entries[idx]]
        return getattr(self._entries[idx], self._field_name)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            values = list(value)
            if len(indices) != len(values):
                raise FileListFieldLengthError
            for entry_idx, item in zip(indices, values, strict=True):
                setattr(self._entries[entry_idx], self._field_name, item)
            return
        setattr(self._entries[idx], self._field_name, value)

    def __delitem__(self, _idx):
        raise ImmutableFieldViewError

    def insert(self, _idx, _value):
        raise ImmutableFieldViewError

    def __eq__(self, other):
        return list(self) == list(other)

    def __repr__(self):
        return repr(list(self))


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
        self._filespecs_view = _EntryFieldView(self, "filespec")
        self._orientations_view = _EntryFieldView(self, "orientation")
        self._linearize_view = _EntryFieldView(self, "linearize")
        self._image_slots_view = _EntryFieldView(self, "image_slot")
        self._loaded_images_view = _EntryFieldView(self, "loaded_image")
        self._images_view = _EntryFieldView(self, "image")
        self._metadata_view = _EntryFieldView(self, "metadata")
        self._update()
        self.reindexed = False

    @property
    def filespecs(self):
        return self._filespecs_view

    @filespecs.setter
    def filespecs(self, values):
        self._replace_entry_field("filespec", values)

    @property
    def orientations(self):
        return self._orientations_view

    @orientations.setter
    def orientations(self, values):
        self._replace_entry_field("orientation", values)

    @property
    def linearize(self):
        return self._linearize_view

    @linearize.setter
    def linearize(self, values):
        self._replace_entry_field("linearize", values)

    @property
    def image_slots(self):
        return self._image_slots_view

    @image_slots.setter
    def image_slots(self, values):
        self._replace_entry_field("image_slot", values)

    @property
    def loaded_images(self):
        return self._loaded_images_view

    @loaded_images.setter
    def loaded_images(self, values):
        self._replace_entry_field("loaded_image", values)

    @property
    def images(self):
        """Payloads already consumed by the UI/render thread."""
        return self._images_view

    @images.setter
    def images(self, values):
        self._replace_entry_field("image", values)

    @property
    def metadata(self):
        return self._metadata_view

    @metadata.setter
    def metadata(self, values):
        self._replace_entry_field("metadata", values)

    def ready_to_upload(self, idx):
        """
        Return True if the given image is ready to be uploaded to OpenGL,
        or has already been uploaded.
        """
        return self.image_status(idx) not in [ImageStatus.PENDING, ImageStatus.INVALID]

    def image_status(self, idx) -> ImageStatus:
        return self.entries[idx].image_slot.status

    def image_revision(self, idx) -> int:
        return self.entries[idx].image_slot.revision

    def image_slot_id(self, idx) -> int:
        return self.entries[idx].image_slot.slot_id

    def image_token(self, idx):
        slot = self.entries[idx].image_slot
        return (slot.slot_id, slot.revision)

    def mark_pending(self, idx):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.PENDING
        entry.image_slot.revision += 1
        entry.loaded_image = None
        entry.image = None

    def mark_loaded(self, idx, img):
        entry = self.entries[idx]
        entry.image_slot.status = ImageStatus.LOADED
        entry.loaded_image = img

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
            entries = tuple(self.entries)
            return FileListSnapshot(
                entries=entries,
                filespecs=tuple(entry.filespec for entry in entries),
                numfiles=self.numfiles,
                orientations=tuple(entry.orientation for entry in entries),
                linearize=tuple(entry.linearize for entry in entries),
                image_slots=tuple(entry.image_slot for entry in entries),
                metadata=tuple(entry.metadata for entry in entries),
            )

    def drop(self, indices):
        """ Drop the given images from this FileList, do not delete the files. """
        with self.mutex:
            try:
                self.entries = self._drop(self.entries, indices)
                self._update()
                print(f"[{threading.current_thread().name}] Dropped images {indices}")
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
                print(f"[{threading.current_thread().name}] Deleted {filespec}")
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

    def _replace_entry_field(self, field_name, values):
        if len(values) != len(self.entries):
            raise FileListFieldLengthError
        for entry, value in zip(self.entries, values, strict=True):
            setattr(entry, field_name, value)

    def _new_slot(self):
        slot = ImageSlot(slot_id=self._next_slot_id)
        self._next_slot_id += 1
        return slot


def main():  # noqa: PLR0915
    """ Parse command-line arguments and run the application. """
    config = types.SimpleNamespace()
    config.fullscreen = argv.exists("--fullscreen")
    config.numtiles = argv.intval("--split", default=1, accepted=[1, 2, 3, 4])
    config.downsample = argv.intval("--downsample", default=1, condition="v >= 1")
    config.smooth = argv.exists("--filter")
    config.normalize = argv.stringval("--normalize", default="off", accepted=list(NORMS.keys()))
    config.idt = argv.stringval("--idt", default="sRGB", accepted=list(COLORSPACES.keys()))
    config.odt = argv.stringval("--odt", default="sRGB", accepted=list(COLORSPACES.keys()))
    config.width = argv.intval("--width", default=None, condition="v >= 1")
    config.height = argv.intval("--height", default=None, condition="v >= 1")
    config.bpp = argv.intval("--bpp", default=None, accepted=[10, 12])
    config.stride = argv.intval("--stride", default=None, condition="v >= 1")
    config.packing = argv.stringval("--packing", default=None, accepted=["unpacked", "plain", "mipi"])
    config.debug = argv.stringval("--debug", default="1", accepted=list("1234rgbl"))
    config.verbose = argv.exists("--verbose")
    config.verbose += argv.exists("--verbose")
    show_version = argv.exists("--version")
    show_help = argv.exists("--help")
    argv.exitIfAnyUnparsedOptions()
    if show_version:
        print(f"glview version {version.__version__}")
        sys.exit()
    if show_help:
        print("Usage: glview [options] [image.(pgm|ppm|pnm|png|jpg|..)] ...")
        print()
        print("  options:")
        print("    --fullscreen            start in full-screen mode; default = windowed")
        print("    --split 1|2|3|4         display images in N separate tiles")
        print("    --downsample N          downsample images N-fold to save memory")
        print("    --normalize off|max|... exposure normalization mode; default = off")
        print("    --filter                use linear filtering; default = nearest")
        print("    --idt sRGB|P3|...       input image color space; default = sRGB")
        print("    --odt sRGB|P3|...       output device color space; default = sRGB")
        print("    --width N               [.raw only] width of the image in pixels")
        print("    --height N              [.raw only] height of the image in pixels")
        print("    --bpp 10|12             [.raw only] bit depth of the image: 10/12")
        print("    --stride N              [.raw only] row stride of the image in bytes")
        print("    --packing plain|...     [.raw only] bit packing: unpacked/plain/mipi")
        print("    --debug 1|2|...|r|g|b   select debug rendering mode; default = 1")
        print("    --verbose               print extra traces to the console")
        print("    --verbose               print even more traces to the console")
        print("    --version               show glview version number & exit")
        print("    --help                  show this help message")
        print()
        print("  keyboard commands:")
        print("    mouse wheel             zoom in/out; synchronized if multiple tiles")
        print("    + / -                   zoom in/out; synchronized if multiple tiles")
        print("    mouse left + move       pan image; synchronized if multiple tiles")
        print("    left / right            pan image; synchronized if multiple tiles")
        print("    PageUp / PageDown       cycle through images on active tile")
        print("    ctrl + left / right     cycle through images on all tiles")
        print("    s                       split window into 1/2/3/4 tiles")
        print("    1 / 2 / 3 / 4           select active tile for per-tile operations")
        print("    p                       in 2-tile layouts, flip the image pair")
        print("    h                       reset zoom/pan/exposure to initial state")
        print("    f                       toggle fullscreen <-> windowed")
        print("    g                       cycle through gamma curves: sRGB/HLG/HDR10")
        print("    n                       cycle through exposure normalization modes")
        print("    k                       cycle through gamut compression modes")
        print("    t                       toggle nearest <-> linear filtering")
        print("    e                       adjust exposure within [-2EV, +2EV]")
        print("    b                       toggle between HDR/LDR exposure control")
        print("    i                       toggle input color space: sRGB/P3/Rec2020")
        print("    o                       toggle output color space: sRGB/P3/Rec2020")
        print("    r                       [per-image] rotate 90 degrees clockwise")
        print("    l                       [per-image] toggle linearization on/off")
        print("    m                       [per-tile] toggle mirroring x/y/both/none")
        print("    a                       [per-tile] toggle autoexposure on/off")
        print("    z                       [per-tile] toggle sharpening on/off")
        print("    c                       [per-tile] toggle tonemapping on/off")
        print("    x                       [per-tile] print image information (EXIF)")
        print("    w                       write a screenshot as both JPG & PFM")
        print("    u                       reload currently shown images from disk")
        print("    d                       drop the currently shown image(s)")
        print("    del                     delete the currently shown image")
        print("    space                   toggle debug rendering on/off")
        print("    q / esc / ctrl+c        terminate")
        print()
        print("  supported file types:")
        print("   ", '\n    '.join(IMAGE_TYPES))
        print()
        print("  available debug rendering modes (--debug M):")
        print("    1 - red => overexposed; blue => out of gamut; magenta => both")
        print("    2 - shades of green => out-of-gamut distance")
        print("    3 - normalized color: rgb' = rgb / max(rgb)")
        print("    4 - red => above diffuse white; magenta => above peak white")
        print("    r - show red channel only, set others to zero")
        print("    g - show green channel only, set others to zero")
        print("    b - show blue channel only, set others to zero")
        print("    l - show image as grayscale (perceived brightness)")
        print()
        print(f"  glview version {version.__version__}.")
        print()
        sys.exit()
    else:
        print("See 'glview --help' for command-line options and keyboard commands.")

    filepatterns = sys.argv[1:] or ["*"]
    filenames = argv.filenames(filepatterns, IMAGE_TYPES, allowAllCaps=True)
    filenames = natsort.natsorted(natsort.natsorted(filenames), key=lambda p: pathlib.Path(p).parent)
    loader = imageprovider.ImageProvider(FileList(filenames), config)
    enforce(loader.files.numfiles > 0, "No valid images to show. Terminating.")

    ui = pygletui.PygletUI(loader.files, ord(config.debug) - ord("0"), bool(config.verbose))
    ui.version = version.__version__
    ui.config.texture_filter = "LINEAR" if config.smooth else "NEAREST"
    ui.config.normalize = NORMS[config.normalize]
    ui.config.cs_in = COLORSPACES[config.idt]
    ui.config.cs_out = COLORSPACES[config.odt]
    ui.fullscreen = config.fullscreen
    ui.state.numtiles = config.numtiles

    renderer = glrenderer.GLRenderer(ui, loader.files, loader, config.verbose)
    ui.start(renderer)
    loader.start()
    main_loop([ui, loader])
    loader.stop()
    ui.stop()


def main_loop(modules):
    """ Keep the application running until exit request or out of memory. """
    try:
        ram_minimum = 512  # exit if available RAM drops below 512 MB
        while all(m.running for m in modules):
            ram_available = psutil.virtual_memory().available / 1024**2
            if ram_available < ram_minimum:
                print(f"ERROR: Only {ram_available:.0f} MB of RAM remaining. Terminating.")
                break
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C pressed, terminating...")
        sys.exit(-1)
    finally:
        sys.exit()


def enforce(expression, message_if_false):
    """ Display the given error message and exit if 'expression' is False. """
    if not expression:
        print(message_if_false)
        sys.exit(-1)


if __name__ == "__main__":
    main()
