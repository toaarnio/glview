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


try:
    # package mode
    from glview import version        # local import
    from glview import argv           # local import
    from glview import pygletui       # local import
    from glview import glrenderer     # local import
    from glview import imageprovider  # local import
    from glview.imagestate import ImageSlot, ImageStatus
except ImportError:
    # stand-alone mode
    import version                 # local import
    import argv                    # local import
    import pygletui                # local import
    import glrenderer              # local import
    import imageprovider           # local import
    from imagestate import ImageSlot, ImageStatus


IMAGE_TYPES = imgio.RO_FORMATS

COLORSPACES = {"sRGB": 0, "P3": 1, "Rec2020": 2, "XYZ": 3}

NORMS = {"off": 0, "max": 1, "stretch": 2, "99": 3, "98": 4, "95": 5, "90": 6, "mean": 7}


@dataclass(frozen=True)
class FileListSnapshot:
    filespecs: tuple
    numfiles: int
    orientations: tuple
    linearize: tuple
    image_slots: tuple
    metadata: tuple
    is_url: tuple


class FileList:
    """ An indexed container for images and their source filenames. """

    def __init__(self, filespecs):
        """
        Create a new FileList with the given list of filenames. Files can only
        be removed from the list, not added or reordered.
        """
        self.mutex = threading.Lock()
        self.filespecs = filespecs
        self.numfiles = len(filespecs)
        self.orientations = [0] * self.numfiles
        self.linearize = [False] * self.numfiles
        self._next_slot_id = 0
        self.image_slots = [self._new_slot() for _ in range(self.numfiles)]
        self.loaded_images = [None] * self.numfiles
        self.images = [None] * self.numfiles  # payloads already consumed by the UI/render thread
        self.metadata = [None] * self.numfiles
        self.is_url = [None] * self.numfiles
        self._update()
        self.reindexed = False

    def ready_to_upload(self, idx):
        """
        Return True if the given image is ready to be uploaded to OpenGL,
        or has already been uploaded.
        """
        return self.image_status(idx) not in [ImageStatus.PENDING, ImageStatus.INVALID]

    def image_status(self, idx) -> ImageStatus:
        return self.image_slots[idx].status

    def image_revision(self, idx) -> int:
        return self.image_slots[idx].revision

    def image_slot_id(self, idx) -> int:
        return self.image_slots[idx].slot_id

    def image_token(self, idx):
        slot = self.image_slots[idx]
        return (slot.slot_id, slot.revision)

    def mark_pending(self, idx):
        self.image_slots[idx].status = ImageStatus.PENDING
        self.image_slots[idx].revision += 1
        self.loaded_images[idx] = None
        self.images[idx] = None

    def mark_loaded(self, idx, img):
        self.image_slots[idx].status = ImageStatus.LOADED
        self.loaded_images[idx] = img

    def mark_released(self, idx):
        self.image_slots[idx].status = ImageStatus.RELEASED
        self.loaded_images[idx] = None

    def mark_invalid(self, idx):
        self.image_slots[idx].status = ImageStatus.INVALID
        self.loaded_images[idx] = None
        self.images[idx] = None

    def consume_image(self, idx, img):
        self.images[idx] = img

    def clear_consumed_image(self, idx):
        self.images[idx] = None

    def snapshot(self) -> FileListSnapshot:
        """Return a consistent read-only view of the catalog state."""
        with self.mutex:
            return FileListSnapshot(
                filespecs=tuple(self.filespecs),
                numfiles=self.numfiles,
                orientations=tuple(self.orientations),
                linearize=tuple(self.linearize),
                image_slots=tuple(self.image_slots),
                metadata=tuple(self.metadata),
                is_url=tuple(self.is_url),
            )

    def drop(self, indices):
        """ Drop the given images from this FileList, do not delete the files. """
        with self.mutex:
            try:
                self.filespecs = self._drop(self.filespecs, indices)
                self.orientations = self._drop(self.orientations, indices)
                self.linearize = self._drop(self.linearize, indices)
                self.image_slots = self._drop(self.image_slots, indices)
                self.loaded_images = self._drop(self.loaded_images, indices)
                self.metadata = self._drop(self.metadata, indices)
                self.images = self._drop(self.images, indices)
                self._update()
                print(f"[{threading.current_thread().name}] Dropped images {indices}")
            except IndexError:
                pass

    def delete(self, idx):
        """ Remove the given image from this FileList and delete the file from disk. """
        with self.mutex:
            try:
                filespec = self.filespecs[idx]
                self.filespecs = self._drop(self.filespecs, [idx])
                self.orientations = self._drop(self.orientations, [idx])
                self.linearize = self._drop(self.linearize, [idx])
                self.image_slots = self._drop(self.image_slots, [idx])
                self.loaded_images = self._drop(self.loaded_images, [idx])
                self.metadata = self._drop(self.metadata, [idx])
                self.images = self._drop(self.images, [idx])
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
        self.numfiles = len(self.filespecs)
        self.is_url = ["://" in f for f in self.filespecs]
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
    config.url = argv.stringval("--url", default=None)
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
        print("    --url <address>         load image from the given web address")
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

    filepatterns = sys.argv[1:] or config.url or ["*"]
    filenames = argv.filenames(filepatterns, IMAGE_TYPES, allowAllCaps=True)
    filenames = natsort.natsorted(natsort.natsorted(filenames), key=lambda p: pathlib.Path(p).parent)
    filenames += [config.url] if config.url is not None else []
    loader = imageprovider.ImageProvider(FileList(filenames), config)
    enforce(loader.files.numfiles > 0, "No valid images to show. Terminating.")

    ui = pygletui.PygletUI(loader.files, ord(config.debug) - ord("0"), bool(config.verbose))
    ui.version = version.__version__
    ui.texture_filter = "LINEAR" if config.smooth else "NEAREST"
    ui.normalize = NORMS[config.normalize]
    ui.cs_in = COLORSPACES[config.idt]
    ui.cs_out = COLORSPACES[config.odt]
    ui.fullscreen = config.fullscreen
    ui.numtiles = config.numtiles

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


def warn(expression, message_if_true):
    """ Display the given warning message if 'expression' is True. """
    if expression:
        print(message_if_true)
    return expression


if __name__ == "__main__":
    main()
