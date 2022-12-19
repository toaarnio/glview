#!/usr/bin/python3 -B

"""
The 'glview' command-line application.
"""

import sys                     # built-in library
import os                      # built-in library
import time                    # built-in library
import threading               # built-in library
import natsort                 # pip install natsort
import psutil                  # pip install psutil

try:
    # package mode
    from glview import argv           # local import
    from glview import pygletui       # local import
    from glview import glrenderer     # local import
    from glview import imageprovider  # local import
except ImportError:
    # stand-alone mode
    import argv                    # local import
    import pygletui                # local import
    import glrenderer              # local import
    import imageprovider           # local import


IMAGE_TYPES = [".pgm", ".ppm", ".pnm", ".pfm", ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".insp", ".exr", ".npy"]


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
        self.images = ["PENDING"] * self.numfiles  # PENDING | RELEASED | INVALID
        self.textures = [None] * self.numfiles     # None | <Texture>
        self.is_url = [None] * self.numfiles
        self._update()

    def remove(self, idx):
        """ Remove the given image from this FileList, do not delete the file. """
        with self.mutex:
            try:
                self.filespecs.pop(idx)
                self.orientations.pop(idx)
                self.textures.pop(idx)
                self.images.pop(idx)
                self._update()
            except IndexError:
                pass

    def delete(self, idx):
        """ Remove the given image from this FileList and delete the file from disk. """
        with self.mutex:
            try:
                filespec = self.filespecs.pop(idx)
                self.orientations.pop(idx)
                self.textures.pop(idx)
                self.images.pop(idx)
                self._update()
                print(f"[{threading.current_thread().name}] Deleting {filespec}...")
                os.remove(filespec)
            except IndexError:
                pass

    def _update(self):
        self.numfiles = len(self.filespecs)
        self.is_url = ["://" in f for f in self.filespecs]


def main():
    """ Parse command-line arguments and run the application. """
    fullscreen = argv.exists("--fullscreen")
    numtiles = argv.intval("--split", default=1, accepted=[1, 2, 3, 4])
    url = argv.stringval("--url", default=None)
    smooth = argv.exists("--filter")
    verbose = argv.exists("--verbose")
    show_help = argv.exists("--help")
    argv.exitIfAnyUnparsedOptions()
    if show_help:
        print("Usage: glview [options] [image.(pgm|ppm|pnm|png|jpg|..)] ...")
        print()
        print("  options:")
        print("    --fullscreen            start in full-screen mode; default = windowed")
        print("    --split 1|2|3|4         display images in N separate tiles")
        print("    --url <address>         load image from the given web address")
        print("    --filter                use linear filtering; default = nearest")
        print("    --verbose               print extra traces to the console")
        print("    --help                  show this help message")
        print()
        print("  runtime:")
        print("    mouse wheel             zoom in/out; synchronized if multiple tiles")
        print("    + / -                   zoom in/out; synchronized if multiple tiles")
        print("    mouse left + move       pan image; synchronized if multiple tiles")
        print("    left / right            pan image; synchronized if multiple tiles")
        print("    PageUp / PageDown       cycle through images on active tile")
        print("    ctrl + left / right     cycle through images on all tiles")
        print("    r                       rotate active tile 90 degrees clockwise")
        print("    s                       split window into 1/2/3/4 tiles")
        print("    1 / 2 / 3 / 4           select active tile for PageUp/PageDown/r")
        print("    w                       write current tile(s) to a PNG")
        print("    h                       reset zoom & pan to initial state")
        print("    f                       toggle fullscreen <-> windowed")
        print("    t                       toggle nearest <-> linear filtering")
        print("    g                       toggle sRGB gamma correction on/off")
        print("    b                       increase brightness in 0.5 EV steps")
        print("    i                       print image information (EXIF)")
        print("    d                       drop the currently shown image")
        print("    del                     delete the currently shown image")
        print("    q / esc / ctrl+c        terminate")
        print()
        print("  supported file types:")
        print("   ", '\n    '.join(IMAGE_TYPES))
        print()
        sys.exit(-1)
    else:
        print("See 'glview --help' for command-line options and keyboard commands.")

    filepatterns = sys.argv[1:] or url or ["*"]
    filenames = argv.filenames(filepatterns, IMAGE_TYPES, allowAllCaps=True)
    filenames = natsort.natsorted(filenames)
    filenames += [url] if url is not None else []
    numfiles = len(filenames)
    enforce(numfiles > 0, "No valid images to show. Terminating.")

    files = FileList(filenames)
    ui = pygletui.PygletUI(files, verbose)
    loader = imageprovider.ImageProvider(files, verbose)
    renderer = glrenderer.GLRenderer(ui, files, loader, verbose)
    ui.texture_filter = "LINEAR" if smooth else "NEAREST"
    ui.fullscreen = fullscreen
    ui.numtiles = numtiles
    ui.start(renderer)
    loader.start()
    main_loop([ui, loader])
    loader.stop()
    ui.stop()


def main_loop(modules):
    """ Keep the application running until exit request or out of memory. """
    try:
        ram_minimum = 512  # exit if available RAM drops below 512 MB
        while all([m.running for m in modules]):
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
