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
    if sys.platform == "darwin":
        loader.start()
        try:
            ui.run(renderer)
        finally:
            if loader.running:
                loader.stop()
    else:
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
