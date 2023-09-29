#!/usr/bin/python3 -B

"""
The 'glview' command-line application.
"""

import sys                     # built-in library
import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
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
except ImportError:
    # stand-alone mode
    import version                 # local import
    import argv                    # local import
    import pygletui                # local import
    import glrenderer              # local import
    import imageprovider           # local import


IMAGE_TYPES = imgio.RO_FORMATS


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
        self.images = ["PENDING"] * self.numfiles  # PENDING | RELEASED | INVALID
        self.textures = [None] * self.numfiles     # None | <Texture>
        self.metadata = [None] * self.numfiles
        self.is_url = [None] * self.numfiles
        self._update()

    def drop(self, indices):
        """ Drop the given images from this FileList, do not delete the files. """
        with self.mutex:
            try:
                self.filespecs = self._drop(self.filespecs, indices)
                self.orientations = self._drop(self.orientations, indices)
                self.linearize = self._drop(self.linearize, indices)
                self.textures = self._drop(self.textures, indices)
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
                self.textures = self._drop(self.textures, [idx])
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


def main():
    """ Parse command-line arguments and run the application. """
    fullscreen = argv.exists("--fullscreen")
    numtiles = argv.intval("--split", default=1, accepted=[1, 2, 3, 4])
    url = argv.stringval("--url", default=None)
    smooth = argv.exists("--filter")
    debug = argv.intval("--debug", default=1, accepted=[1, 2, 3])
    verbose = argv.exists("--verbose")
    verbose += argv.exists("--verbose")
    show_version = argv.exists("--version")
    show_help = argv.exists("--help")
    argv.exitIfAnyUnparsedOptions()
    if show_version:
        print(f"glview version {version.__version__}")
        sys.exit(-1)
    if show_help:
        print("Usage: glview [options] [image.(pgm|ppm|pnm|png|jpg|..)] ...")
        print()
        print("  options:")
        print("    --fullscreen            start in full-screen mode; default = windowed")
        print("    --split 1|2|3|4         display images in N separate tiles")
        print("    --url <address>         load image from the given web address")
        print("    --filter                use linear filtering; default = nearest")
        print("    --debug N               select debug rendering mode; default = 1")
        print("    --verbose               print extra traces to the console")
        print("    --verbose               print even more traces to the console")
        print("    --version               show glview version number & exit")
        print("    --help                  show this help message")
        print()
        print("  runtime:")
        print("    mouse wheel             zoom in/out; synchronized if multiple tiles")
        print("    + / -                   zoom in/out; synchronized if multiple tiles")
        print("    mouse left + move       pan image; synchronized if multiple tiles")
        print("    left / right            pan image; synchronized if multiple tiles")
        print("    PageUp / PageDown       cycle through images on active tile")
        print("    ctrl + left / right     cycle through images on all tiles")
        print("    r                       rotate current image 90 degrees clockwise")
        print("    l                       toggle current image linearization on/off")
        print("    s                       split window into 1/2/3/4 tiles")
        print("    1 / 2 / 3 / 4           select active tile for PageUp/PageDown/r/l")
        print("    h                       reset zoom/pan/exposure to initial state")
        print("    f                       toggle fullscreen <-> windowed")
        print("    t                       toggle nearest <-> linear filtering")
        print("    g                       toggle output gamma: sRGB/HDR10/HLG")
        print("    n                       toggle exposure normalization on/off")
        print("    e                       slide exposure from -2EV to +2EV & back")
        print("    b                       toggle between HDR/LDR exposure control")
        print("    i                       toggle input color space: sRGB/P3/Rec2020")
        print("    o                       toggle output color space: sRGB/P3/Rec2020")
        print("    k                       toggle gamut compression strength")
        print("    x                       print image information (EXIF)")
        print("    d                       drop the currently shown image")
        print("    del                     delete the currently shown image")
        print("    space                   toggle debug rendering on/off")
        print("    q / esc / ctrl+c        terminate")
        print()
        print("  supported file types:")
        print("   ", '\n    '.join(IMAGE_TYPES))
        print()
        print("  available debug rendering modes (--debug N):")
        print("    1 - red => overexposed; blue => out of gamut; magenta => both")
        print("    2 - shades of green => out-of-gamut distance")
        print("    3 - normalized color: rgb' = rgb / max(rgb)")
        print()
        print(f"  glview version {version.__version__}.")
        print()
        sys.exit(-1)
    else:
        print("See 'glview --help' for command-line options and keyboard commands.")

    filepatterns = sys.argv[1:] or url or ["*"]
    filenames = argv.filenames(filepatterns, IMAGE_TYPES, allowAllCaps=True)
    filenames = natsort.natsorted(filenames)
    filenames += [url] if url is not None else []
    loader = imageprovider.ImageProvider(FileList(filenames), bool(verbose))
    enforce(loader.files.numfiles > 0, "No valid images to show. Terminating.")

    ui = pygletui.PygletUI(loader.files, debug, bool(verbose))
    ui.version = version.__version__
    ui.texture_filter = "LINEAR" if smooth else "NEAREST"
    ui.fullscreen = fullscreen
    ui.numtiles = numtiles

    renderer = glrenderer.GLRenderer(ui, loader.files, loader, verbose, manual_mipmaps=smooth)
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
