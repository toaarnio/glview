"""The 'glview' command-line application: argument parsing and bootstrap."""

import pathlib
import sys
import time
import types

import imgio
import natsort
import psutil

from glview import argv
from glview import desktop
from glview import glrenderer
from glview import imageprovider
from glview import macosapp
from glview import pygletui
from glview import version
import glview._io  # noqa: F401 (ensures builtins.print is patched on Windows)
from glview.filelist import FileEntry, FileEntrySnapshot, FileList, FileListSnapshot  # noqa: F401 (back-compat re-exports)


IMAGE_TYPES = imgio.RO_FORMATS

COLORSPACES = {"sRGB": 0, "P3": 1, "Rec2020": 2, "XYZ": 3}

NORMS = {"off": 0, "max": 1, "stretch": 2, "99": 3, "98": 4, "95": 5, "90": 6, "mean": 7}


HELP_TEXT = """\
Usage: glview [options] [image.(pgm|ppm|pnm|png|jpg|..)] ...

  options:
    --fullscreen            start in full-screen mode; default = windowed
    --split 1|2|3|4         display images in N separate tiles
    --downsample N          downsample images N-fold to save memory
    --normalize off|max|... exposure normalization mode; default = off
    --filter                use linear filtering; default = nearest
    --idt sRGB|P3|...       input image color space; default = sRGB
    --odt sRGB|P3|...       output device color space; default = sRGB
    --width N               [.raw only] width of the image in pixels
    --height N              [.raw only] height of the image in pixels
    --bpp 10|12             [.raw only] bit depth of the image: 10/12
    --stride N              [.raw only] row stride of the image in bytes
    --packing plain|...     [.raw only] bit packing: unpacked/plain/mipi
    --debug 1|2|...|r|g|b   select debug rendering mode; default = 1
    --verbose               print extra traces to the console
    --verbose               print even more traces to the console
    --install-default-handler  register glview for desktop file associations
    --create-macos-app      build a minimal macOS .app bundle into dist/
    --version               show glview version number & exit
    --help                  show this help message

  keyboard commands:
    mouse wheel             zoom in/out; synchronized if multiple tiles
    + / -                   zoom in/out; synchronized if multiple tiles
    mouse left + move       pan image; synchronized if multiple tiles
    left / right            pan image; synchronized if multiple tiles
    PageUp / PageDown       cycle through images on active tile
    ctrl + left / right     cycle through images on all tiles
    tab                     toggle thumbnail filmstrip on/off; click to jump
    v                       toggle on-screen HUD on/off
    s                       split window into 1/2/3/4 tiles
    1 / 2 / 3 / 4           select active tile for per-tile operations
    p                       in 2-tile layouts, flip the image pair
    h                       reset zoom/pan/exposure to initial state
    f                       toggle fullscreen <-> windowed
    g                       cycle through gamma curves: sRGB/HLG/HDR10
    n                       cycle through exposure normalization modes
    k                       cycle through gamut compression modes
    t                       toggle nearest <-> linear filtering
    e                       adjust exposure within [-2EV, +2EV]
    b                       toggle between HDR/LDR exposure control
    i                       toggle input color space: sRGB/P3/Rec2020
    o                       toggle output color space: sRGB/P3/Rec2020
    r                       [per-image] rotate 90 degrees clockwise
    l                       [per-image] toggle linearization on/off
    m                       [per-tile] toggle mirroring x/y/both/none
    a                       [per-tile] toggle autoexposure on/off
    z                       [per-tile] toggle sharpening on/off
    c                       [per-tile] toggle tonemapping on/off
    x                       [per-tile] print image information (EXIF)
    w                       write a screenshot as both JPG & PFM
    u                       reload currently shown images from disk
    d                       drop the currently shown image(s)
    del                     delete the currently shown image
    space                   toggle debug rendering on/off
    q / esc / ctrl+c        terminate

  available debug rendering modes (--debug M):
    1 - red => overexposed; blue => out of gamut; magenta => both
    2 - shades of green => out-of-gamut distance
    3 - normalized color: rgb' = rgb / max(rgb)
    4 - red => above diffuse white; magenta => above peak white
    r - show red channel only, set others to zero
    g - show green channel only, set others to zero
    b - show blue channel only, set others to zero
    l - show image as grayscale (perceived brightness)

  glview version {version}.
"""


def _parse_config():
    """Parse `sys.argv` into a SimpleNamespace of rendering/loading options."""
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
    # `argv.exists` consumes one matching token per call, so two `--verbose`
    # flags raise verbosity to level 2 (verbose loader output + GL traces).
    config.verbose = argv.exists("--verbose")
    config.verbose += argv.exists("--verbose")
    return config


def _print_help():
    print(HELP_TEXT.format(version=version.__version__))
    print("  supported file types:")
    print("   ", "\n    ".join(IMAGE_TYPES))
    print()


def _load_filenames():
    """Expand CLI file patterns into a natsorted list of image filenames."""
    filepatterns = sys.argv[1:] or ["*"]
    filenames = argv.filenames(filepatterns, IMAGE_TYPES, allowAllCaps=True)
    filenames = natsort.natsorted(natsort.natsorted(filenames), key=lambda p: pathlib.Path(p).parent)
    return filenames


def _build_ui(loader, config):
    ui = pygletui.PygletUI(loader.files, ord(config.debug) - ord("0"), bool(config.verbose))
    ui.version = version.__version__
    ui.config.texture_filter = "LINEAR" if config.smooth else "NEAREST"
    ui.config.normalize = NORMS[config.normalize]
    ui.config.cs_in = COLORSPACES[config.idt]
    ui.config.cs_out = COLORSPACES[config.odt]
    ui.fullscreen = config.fullscreen
    ui.state.numtiles = config.numtiles
    return ui


def _run_app(config):
    filenames = _load_filenames()
    loader = imageprovider.ImageProvider(FileList(filenames), config)
    enforce(loader.files.numfiles > 0, "No valid images to show. Terminating.")
    ui = _build_ui(loader, config)
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


def main():
    """Parse command-line arguments and run the application."""
    config = _parse_config()
    install_default_handler = argv.exists("--install-default-handler")
    create_macos_app = argv.exists("--create-macos-app")
    show_version = argv.exists("--version")
    show_help = argv.exists("--help")
    argv.exitIfAnyUnparsedOptions()
    if install_default_handler:
        _install_handler_and_exit()
    if create_macos_app:
        _create_macos_app_and_exit()
    if show_version:
        print(f"glview version {version.__version__}")
        sys.exit()
    if show_help:
        _print_help()
        sys.exit()
    print(f"glview version {version.__version__} [{pathlib.Path(__file__)}].")
    print("See 'glview --help' for command-line options and keyboard commands.")
    _run_app(config)


def main_loop(modules):
    """Keep the application running until exit request."""
    try:
        ram_minimum = 512  # exit if available RAM drops below 512 MB
        low_memory_active = False
        while all(m.running for m in modules):
            ram_available = psutil.virtual_memory().available / 1024**2
            if ram_available < ram_minimum:
                if not low_memory_active:
                    print(f"WARNING: Only {ram_available:.0f} MB of RAM remaining. Pausing background loading until memory recovers.")
                    low_memory_active = True
            elif low_memory_active:
                print(f"INFO: System RAM recovered to {ram_available:.0f} MB. Background loading can continue.")
                low_memory_active = False
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


def _install_handler_and_exit():
    try:
        result = desktop.install_default_handler()
    except NotImplementedError as exc:
        print(exc)
        sys.exit(-1)
    if result.desktop_path is not None:
        print(f"Installed desktop entry: {result.desktop_path}")
    if result.mime_xml_path is not None:
        print(f"Installed MIME definitions: {result.mime_xml_path}")
    if result.mimeapps_paths:
        print("Updated MIME defaults:")
        for mimeapps_path in result.mimeapps_paths:
            print(f"  {mimeapps_path}")
    elif result.mimeapps_path is not None:
        print(f"Updated MIME defaults: {result.mimeapps_path}")
    if result.bundle_path is not None:
        print(f"Registered macOS app bundle: {result.bundle_path}")
    if result.bundle_id is not None:
        print(f"Using macOS bundle identifier: {result.bundle_id}")
    if result.settings_uri is not None:
        print(f"Opened Windows Default Apps settings: {result.settings_uri}")
    if result.registry_paths:
        print(f"Updated {len(result.registry_paths)} registry paths for glview.")
    print(f"Registered {len(result.mime_types)} MIME types and {len(result.extensions)} file extensions for glview.")
    if result.note:
        print(result.note)
    sys.exit()


def _create_macos_app_and_exit():
    bundle_path = macosapp.build_app_bundle()
    print(f"Created macOS app bundle: {bundle_path}")
    print("This bundle uses the current Python environment and source tree; it is not a standalone app.")
    sys.exit()


if __name__ == "__main__":
    main()
