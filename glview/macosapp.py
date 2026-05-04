"""
Helpers for creating a minimal macOS app bundle for glview.
"""

from __future__ import annotations

import os
from pathlib import Path
import plistlib
import re
import shlex
import stat

from glview import version


DEFAULT_BUNDLE_ID = "io.github.toaarnio.glview"
DEFAULT_BUNDLE_NAME = "glview"
SUPPORTED_IMAGE_UTIS = (
    "com.microsoft.bmp",
    "public.exr",
    "public.jpeg",
    "public.png",
    "public.radiance",
    "public.tiff",
)


def build_app_bundle(
    output_dir: Path | str = "dist",
    python_executable: Path | str | None = None,
    source_root: Path | str | None = None,
    bundle_id: str = DEFAULT_BUNDLE_ID,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
    version_string: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    python_executable = Path(os.sys.executable if python_executable is None else python_executable).resolve()
    source_root = Path(_default_source_root() if source_root is None else source_root).resolve()
    version_string = version.__version__ if version_string is None else version_string

    bundle_path = output_dir / f"{bundle_name}.app"
    contents_dir = bundle_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"

    if bundle_path.exists() and not bundle_path.is_dir():
        message = f"Cannot create app bundle because target exists and is not a directory: {bundle_path}"
        raise FileExistsError(message)

    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    _write_info_plist(contents_dir / "Info.plist", bundle_id, bundle_name, version_string)
    _write_pkginfo(contents_dir / "PkgInfo")
    _write_launcher(macos_dir / bundle_name, python_executable, source_root)

    return bundle_path


def _default_source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _write_info_plist(path: Path, bundle_id: str, bundle_name: str, version_string: str) -> None:
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": bundle_name,
        "CFBundleExecutable": bundle_name,
        "CFBundleIdentifier": bundle_id,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": bundle_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version_string,
        "CFBundleVersion": _bundle_version(version_string),
        "LSApplicationCategoryType": "public.app-category.graphics-design",
        "NSHighResolutionCapable": True,
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": "Supported image files",
            "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "Alternate",
            "LSItemContentTypes": list(SUPPORTED_IMAGE_UTIS),
        }],
    }
    with path.open("wb") as handle:
        plistlib.dump(info, handle, sort_keys=False)


def _write_pkginfo(path: Path) -> None:
    path.write_text("APPL????", encoding="ascii")


def _write_launcher(path: Path, python_executable: Path, source_root: Path) -> None:
    launcher = "\n".join([
        "#!/bin/sh",
        f'PYTHON_BIN={shlex.quote(str(python_executable))}',
        f'SOURCE_ROOT={shlex.quote(str(source_root))}',
        'if [ -n "$PYTHONPATH" ]; then',
        '  export PYTHONPATH="$SOURCE_ROOT:$PYTHONPATH"',
        "else",
        '  export PYTHONPATH="$SOURCE_ROOT"',
        "fi",
        'exec "$PYTHON_BIN" -m glview.glview "$@"',
        "",
    ])
    path.write_text(launcher, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _bundle_version(version_string: str) -> str:
    parts = re.findall(r"\d+", version_string)
    if not parts:
        return "0.0.0"
    return ".".join(parts[:3])
