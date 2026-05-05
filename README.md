# GLView

Lightning-fast image viewer with synchronized split-screen zooming &amp;
panning. Designed mainly for algorithm developers who demand pixel-precise
and color-accurate output.

## Quick Start
```
pipx install glview
glview --help
glview <your-image-file>
glview --install-default-handler   # register desktop file associations
glview --create-macos-app          # [MACOS] create dist/glview.app from the current environment

# [LINUX] If there's a failure due to missing libGL.so
sudo apt update
sudo apt install libegl1 mesa-utils libegl1-mesa-dev

# [MACOS] If there's a failure due to missing EXIF libraries
brew install exiv2
```

`--install-default-handler` is implemented for:

- Linux desktop environments that honor XDG desktop entries and `mimeapps.list`.
  It installs a per-user desktop entry under `~/.local/share/applications/` and
  updates both `~/.config/mimeapps.list` and
  `~/.local/share/applications/mimeapps.list` for broader desktop-environment
  compatibility. It also installs user-scoped shared MIME definitions under
  `~/.local/share/mime/packages/` and runs `update-mime-database` so `.pfm`,
  `.pnm`, `.ppm`, and `.pgm` resolve to concrete image MIME types.
- Windows. It registers `glview` under `HKCU\Software\Classes` for supported file
  types, adds a `RegisteredApplications`/Capabilities entry for the Default Apps
  UI, and opens the `glview` page in Windows Default Apps directly. File
  associations point at the `glview-launcher` GUI entrypoint so shell activations
  can be batched without popping a console window for every file. On modern
  Windows, the actual default app choice is still protected by the OS, so the
  user still needs to confirm the default through the Windows Settings UI.
- macOS, but only when `glview` is running from a proper `.app` bundle with a
  `CFBundleIdentifier`. In that case it registers the bundle with Launch Services
  and assigns the resolvable image UTIs to that bundle. A plain `python` or
  `pipx` executable is not an app bundle, so macOS will reject that path.

## macOS app bundle
`glview --create-macos-app` builds a minimal `dist/glview.app` bundle that
launches `glview` through the current Python interpreter and source tree. That
gives macOS a real app bundle identity so `--install-default-handler` can work.

This is intentionally a local development bundle, not a fully standalone app
distribution. Moving it to a machine without the same Python environment or
source checkout will break it.

## Windows association checks from WSL
Use `make windows-assoc-check` to print the Windows-side commands needed to
verify the registration from within WSL. The helper prints `reg.exe` queries for
the `RegisteredApplications`, Capabilities, `Applications\\glview.exe`, and
`OpenWithProgids` entries, a deep link into the `glview` page in Default Apps,
a `UserChoice` query for the current default app, and a `cmd.exe /c start`
example to test the actual shell resolution path.

You can also run the helper directly with custom paths:
```bash
bash scripts/windows_assoc_check.sh 'C:\path\to\glview.exe' 'C:\path\to\test.png'
```

## Usage
```
Usage: glview [options] [image.(pgm|ppm|pnm|png|jpg|..)] ...

  options:
    --fullscreen            start in full-screen mode; default = windowed
    --split 1|2|3|4         display images in N separate tiles
    --url <address>         load image from the given web address
    --downsample N          downsample images N-fold to save memory
    --normalize off|max|... exposure normalization mode; default = off
    --filter                use linear filtering; default = nearest
    --idt sRGB|P3|...       input image color space; default = sRGB
    --odt sRGB|P3|...       output device color space; default = sRGB
    --debug 1|2|...|r|g|b   select debug rendering mode; default = 1
    --verbose               print extra traces to the console
    --verbose               print even more traces to the console
    --install-default-handler
                            register glview for desktop file associations
    --install-handler       alias for --install-default-handler
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

  supported file types:
    .pnm
    .pgm
    .ppm
    .pfm
    .png
    .jpg
    .jpeg
    .bmp
    .tif
    .tiff
    .insp
    .npy
    .raw
    .exr
    .hdr

  available debug rendering modes (--debug M):
    1 - red => overexposed; blue => out of gamut; magenta => both
    2 - shades of green => out-of-gamut distance
    3 - normalized color: rgb` = rgb / max(rgb)
    4 - red => above diffuse white; magenta => above peak white
    r - show red channel only, set others to zero
    g - show green channel only, set others to zero
    b - show blue channel only, set others to zero
    l - show image as grayscale (perceived brightness)

  glview version 1.19.1.
```
