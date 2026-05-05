"""
Helpers for desktop integration.
"""

from __future__ import annotations

from configparser import RawConfigParser
from dataclasses import dataclass
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
from urllib.parse import quote


DESKTOP_FILENAME = "glview.desktop"
WINDOWS_APPLICATION_ID = "glview"
WINDOWS_REGISTERED_APP_NAME = "glview"
WINDOWS_APPLICATION_KEY = r"Software\Classes\Applications\glview.exe"
WINDOWS_REGISTERED_APPLICATIONS_KEY = r"Software\RegisteredApplications"
WINDOWS_CAPABILITIES_KEY = r"Software\glview\Capabilities"
WINDOWS_MULTISELECT_MODEL = "Player"

# Keep this list focused on MIME types that are reasonably well-defined and map
# to formats that glview can open directly.
SUPPORTED_MIME_TYPES = (
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/vnd.radiance",
    "image/x-exr",
    "image/x-portable-anymap",
    "image/x-portable-bitmap",
    "image/x-portable-floatmap",
    "image/x-portable-graymap",
    "image/x-portable-pixmap",
)

SUPPORTED_EXTENSIONS = (
    ".bmp",
    ".exr",
    ".hdr",
    ".insp",
    ".jpeg",
    ".jpg",
    ".mipi",
    ".npy",
    ".pfm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".raw",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True)
class InstallResult:
    platform: str
    mime_types: tuple[str, ...]
    extensions: tuple[str, ...]
    desktop_path: Path | None = None
    mimeapps_path: Path | None = None
    mimeapps_paths: tuple[Path, ...] = ()
    mime_xml_path: Path | None = None
    bundle_id: str | None = None
    bundle_path: Path | None = None
    settings_uri: str | None = None
    registry_paths: tuple[str, ...] = ()
    note: str | None = None


def desktop_entry_text(exec_command: str = "glview", mime_types: tuple[str, ...] = SUPPORTED_MIME_TYPES) -> str:
    mime_list = ";".join(mime_types)
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Terminal=false",
        "Name=glview",
        "GenericName=Image Viewer",
        "Comment=Fast image viewer with zooming and tiling",
        "TryExec=glview",
        f"Exec={exec_command} %F",
        f"MimeType={mime_list};",
        "Categories=Graphics;Viewer;",
        "",
    ])


def install_default_handler(
    exec_command: str = "glview",
    data_home: Path | None = None,
    config_home: Path | None = None,
    platform_name: str | None = None,
    mime_database_backend=None,
    registry_backend=None,
    launcher_backend=None,
    executable_path: Path | None = None,
    bundle_backend=None,
) -> InstallResult:
    platform_name = os.sys.platform if platform_name is None else platform_name
    if platform_name.startswith("linux"):
        return _install_linux(
            exec_command=exec_command,
            data_home=data_home,
            config_home=config_home,
            mime_database_backend=mime_database_backend,
        )
    if platform_name.startswith("win"):
        return _install_windows(
            exec_command=exec_command,
            registry_backend=registry_backend,
            launcher_backend=launcher_backend,
        )
    if platform_name == "darwin":
        return _install_macos(
            executable_path=executable_path,
            bundle_backend=bundle_backend,
        )
    message = f"Default file associations are currently unsupported on platform: {platform_name}"
    raise NotImplementedError(message)


def _install_linux(
    exec_command: str,
    data_home: Path | None,
    config_home: Path | None,
    mime_database_backend,
) -> InstallResult:
    data_home = _data_home() if data_home is None else Path(data_home)
    config_home = _config_home() if config_home is None else Path(config_home)
    applications_dir = data_home / "applications"
    mime_dir = data_home / "mime"
    mime_packages_dir = mime_dir / "packages"
    desktop_path = applications_dir / DESKTOP_FILENAME
    mimeapps_paths = _linux_mimeapps_paths(config_home, data_home)
    mime_xml_path = mime_packages_dir / "glview.xml"

    applications_dir.mkdir(parents=True, exist_ok=True)
    config_home.mkdir(parents=True, exist_ok=True)
    mime_packages_dir.mkdir(parents=True, exist_ok=True)

    desktop_path.write_text(desktop_entry_text(exec_command=exec_command), encoding="utf-8")
    mime_xml_path.write_text(shared_mime_info_xml(), encoding="utf-8")
    backend = mime_database_backend if mime_database_backend is not None else _UpdateMimeDatabaseBackend()
    backend.update_database(mime_dir)
    for mimeapps_path in mimeapps_paths:
        mimeapps_path.parent.mkdir(parents=True, exist_ok=True)
        _update_mimeapps_list(mimeapps_path, DESKTOP_FILENAME, SUPPORTED_MIME_TYPES)

    return InstallResult(
        platform="linux",
        desktop_path=desktop_path,
        mimeapps_path=mimeapps_paths[0],
        mimeapps_paths=mimeapps_paths,
        mime_xml_path=mime_xml_path,
        mime_types=SUPPORTED_MIME_TYPES,
        extensions=SUPPORTED_EXTENSIONS,
    )


def _install_windows(exec_command: str, registry_backend=None, launcher_backend=None) -> InstallResult:
    backend = registry_backend if registry_backend is not None else _import_winreg()
    launcher = launcher_backend if launcher_backend is not None else _WindowsSettingsLauncher()
    launcher_command = _windows_launcher_command(exec_command)
    command = f'"{launcher_command}" "%1"'
    registry_paths = set()
    settings_uri = _windows_default_apps_uri(WINDOWS_REGISTERED_APP_NAME)

    _registry_set_string(backend, WINDOWS_APPLICATION_KEY, "FriendlyAppName", "glview")
    registry_paths.add(WINDOWS_APPLICATION_KEY)
    _registry_set_string(backend, rf"{WINDOWS_APPLICATION_KEY}\shell\open", "MultiSelectModel", WINDOWS_MULTISELECT_MODEL)
    registry_paths.add(rf"{WINDOWS_APPLICATION_KEY}\shell\open")
    _registry_set_string(backend, rf"{WINDOWS_APPLICATION_KEY}\shell\open\command", "", command)
    registry_paths.add(rf"{WINDOWS_APPLICATION_KEY}\shell\open\command")

    supported_types_key = rf"{WINDOWS_APPLICATION_KEY}\SupportedTypes"
    for extension in SUPPORTED_EXTENSIONS:
        _registry_set_string(backend, supported_types_key, extension, "")
    registry_paths.add(supported_types_key)

    _registry_set_string(backend, WINDOWS_CAPABILITIES_KEY, "ApplicationName", "glview")
    _registry_set_string(backend, WINDOWS_CAPABILITIES_KEY, "ApplicationDescription", "Fast image viewer with zooming and tiling")
    registry_paths.add(WINDOWS_CAPABILITIES_KEY)
    capabilities_assoc_key = rf"{WINDOWS_CAPABILITIES_KEY}\FileAssociations"
    for extension in SUPPORTED_EXTENSIONS:
        _registry_set_string(backend, capabilities_assoc_key, extension, _windows_progid(extension))
    registry_paths.add(capabilities_assoc_key)
    _registry_set_string(
        backend,
        WINDOWS_REGISTERED_APPLICATIONS_KEY,
        WINDOWS_REGISTERED_APP_NAME,
        WINDOWS_CAPABILITIES_KEY,
    )
    registry_paths.add(WINDOWS_REGISTERED_APPLICATIONS_KEY)

    for extension in SUPPORTED_EXTENSIONS:
        progid = _windows_progid(extension)
        _registry_set_string(backend, rf"Software\Classes\{extension}\OpenWithProgids", progid, "")
        _registry_set_string(backend, rf"Software\Classes\{progid}", "", f"glview {extension} file")
        _registry_set_string(backend, rf"Software\Classes\{progid}\shell\open", "MultiSelectModel", WINDOWS_MULTISELECT_MODEL)
        _registry_set_string(backend, rf"Software\Classes\{progid}\shell\open\command", "", command)
        registry_paths.add(rf"Software\Classes\{extension}\OpenWithProgids")
        registry_paths.add(rf"Software\Classes\{progid}")
        registry_paths.add(rf"Software\Classes\{progid}\shell\open")
        registry_paths.add(rf"Software\Classes\{progid}\shell\open\command")

    launcher.open_uri(settings_uri)
    note = (
        "Registered glview for Windows file associations under HKCU. "
        "Opened the Windows Default Apps page for glview. "
        "Windows 10/11 still require the user to confirm the actual default app in system UI."
    )
    return InstallResult(
        platform="windows",
        mime_types=SUPPORTED_MIME_TYPES,
        extensions=SUPPORTED_EXTENSIONS,
        settings_uri=settings_uri,
        registry_paths=tuple(sorted(registry_paths)),
        note=note,
    )


def _install_macos(executable_path: Path | None = None, bundle_backend=None) -> InstallResult:
    exec_path = Path(os.sys.executable) if executable_path is None else Path(executable_path)
    bundle_path = _macos_bundle_path(exec_path)
    if bundle_path is None:
        message = (
            "macOS default-handler installation requires glview to run from a bundled .app. "
            "A plain Python or pipx executable has no Launch Services bundle identity."
        )
        raise NotImplementedError(message)

    bundle_id = _macos_bundle_id(bundle_path)
    if bundle_id is None:
        message = f"macOS app bundle is missing CFBundleIdentifier: {bundle_path}"
        raise NotImplementedError(message)

    backend = bundle_backend if bundle_backend is not None else _MacOSLaunchServicesBackend()
    backend.register_app(bundle_path)

    registered_utis = []
    for extension in SUPPORTED_EXTENSIONS:
        uti = backend.uti_for_extension(extension)
        if uti is None:
            continue
        backend.set_default_role_handler(uti, bundle_id)
        registered_utis.append(uti)

    note = (
        "Registered glview with macOS Launch Services for the UTIs resolvable from its supported extensions. "
        "This requires running from a proper .app bundle with a CFBundleIdentifier."
    )
    return InstallResult(
        platform="macos",
        mime_types=SUPPORTED_MIME_TYPES,
        extensions=SUPPORTED_EXTENSIONS,
        bundle_id=bundle_id,
        bundle_path=bundle_path,
        note=note,
    )


def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def shared_mime_info_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="image/x-portable-floatmap">
    <comment>Portable FloatMap image</comment>
    <glob pattern="*.pfm"/>
    <magic priority="80">
      <match type="string" offset="0" value="PF"/>
      <match type="string" offset="0" value="Pf"/>
    </magic>
  </mime-type>
  <mime-type type="image/x-portable-anymap">
    <glob pattern="*.pnm"/>
  </mime-type>
  <mime-type type="image/x-portable-pixmap">
    <glob pattern="*.ppm"/>
  </mime-type>
  <mime-type type="image/x-portable-graymap">
    <glob pattern="*.pgm"/>
  </mime-type>
</mime-info>
"""


def _linux_mimeapps_paths(config_home: Path, data_home: Path) -> tuple[Path, ...]:
    candidates = (
        config_home / "mimeapps.list",
        data_home / "applications" / "mimeapps.list",
    )
    unique_paths = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_paths.append(candidate)
    return tuple(unique_paths)


def _update_mimeapps_list(path: Path, desktop_filename: str, mime_types: tuple[str, ...]) -> None:
    parser = RawConfigParser(interpolation=None, strict=False)
    parser.optionxform = str

    if path.exists():
        parser.read(path, encoding="utf-8")

    for section in ("Default Applications", "Added Associations"):
        if not parser.has_section(section):
            parser.add_section(section)

    for mime_type in mime_types:
        parser.set("Default Applications", mime_type, f"{desktop_filename};")
        existing = parser.get("Added Associations", mime_type, fallback="")
        parser.set("Added Associations", mime_type, _association_list(existing, desktop_filename))

    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle, space_around_delimiters=False)


def _association_list(existing: str, desktop_filename: str) -> str:
    entries = [entry for entry in existing.split(";") if entry]
    entries = [entry for entry in entries if entry != desktop_filename]
    entries.insert(0, desktop_filename)
    return ";".join(entries) + ";"


class _UpdateMimeDatabaseBackend:

    def __init__(self, command_path: str | None = None) -> None:
        resolved = command_path if command_path is not None else shutil.which("update-mime-database")
        if resolved is None:
            message = "update-mime-database was not found on PATH"
            raise FileNotFoundError(message)
        self._command_path = resolved

    def update_database(self, mime_dir: Path) -> None:
        subprocess.run(  # noqa: S603 - fixed command path and installer-owned target directory.
            [self._command_path, str(mime_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _import_winreg():
    import winreg  # noqa: PLC0415

    return winreg


class _WindowsSettingsLauncher:

    def open_uri(self, uri: str) -> None:
        os.startfile(uri)  # noqa: S606


def _registry_set_string(registry_backend, key_path: str, value_name: str, value: str) -> None:
    key = registry_backend.CreateKeyEx(registry_backend.HKEY_CURRENT_USER, key_path, 0, _registry_access(registry_backend))
    try:
        registry_backend.SetValueEx(key, value_name, 0, registry_backend.REG_SZ, value)
    finally:
        registry_backend.CloseKey(key)


def _registry_access(registry_backend) -> int:
    return getattr(registry_backend, "KEY_WRITE", 0) | getattr(registry_backend, "KEY_WOW64_64KEY", 0)


def _windows_progid(extension: str) -> str:
    return f"{WINDOWS_APPLICATION_ID}{extension}"


def _windows_default_apps_uri(registered_app_name: str) -> str:
    return f"ms-settings:defaultapps?registeredAppUser={quote(registered_app_name, safe='')}"


def _windows_launcher_command(exec_command: str) -> str:
    command_path = Path(exec_command)
    if command_path.suffix.lower() == ".exe":
        return str(command_path.with_name(f"{command_path.stem}-launcher{command_path.suffix}"))
    return f"{exec_command}-launcher"


def _macos_bundle_path(executable_path: Path) -> Path | None:
    resolved = executable_path.resolve()
    for parent in [resolved, *resolved.parents]:
        parts = parent.parts
        if ".app" not in parent.name and not any(part.endswith(".app") for part in parts):
            continue
        for idx, part in enumerate(parts):
            if part.endswith(".app"):
                bundle_path = Path(*parts[:idx + 1])
                info_plist = bundle_path / "Contents" / "Info.plist"
                if info_plist.exists():
                    return bundle_path
    return None


def _macos_bundle_id(bundle_path: Path) -> str | None:
    info_plist = bundle_path / "Contents" / "Info.plist"
    if not info_plist.exists():
        return None
    with info_plist.open("rb") as handle:
        plist = plistlib.load(handle)
    return plist.get("CFBundleIdentifier")


class _MacOSLaunchServicesBackend:
    _K_LS_ROLES_ALL = 0xFFFFFFFF

    def __init__(self):
        import ctypes  # noqa: PLC0415

        self.ctypes = ctypes
        self.core_foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.core_services = ctypes.CDLL("/System/Library/Frameworks/CoreServices.framework/CoreServices")
        self._configure_symbols()

    def register_app(self, bundle_path: Path) -> None:
        bundle_url = self._cfurl_from_path(bundle_path)
        try:
            status = self.core_services.LSRegisterURL(bundle_url, True)
            if status != 0:
                message = f"LSRegisterURL failed for {bundle_path} with status {status}"
                raise OSError(message)
        finally:
            self.core_foundation.CFRelease(bundle_url)

    def uti_for_extension(self, extension: str) -> str | None:
        ext_ref = self._cfstring(extension.lstrip("."))
        tag_class = self._cfstring("public.filename-extension")
        conforming = self._cfstring("public.image")
        try:
            uti = self.core_services.UTTypeCreatePreferredIdentifierForTag(tag_class, ext_ref, conforming)
            if not uti:
                return None
            try:
                return self._cfstring_to_str(uti)
            finally:
                self.core_foundation.CFRelease(uti)
        finally:
            self.core_foundation.CFRelease(ext_ref)
            self.core_foundation.CFRelease(tag_class)
            self.core_foundation.CFRelease(conforming)

    def set_default_role_handler(self, uti: str, bundle_id: str) -> None:
        uti_ref = self._cfstring(uti)
        bundle_ref = self._cfstring(bundle_id)
        try:
            status = self.core_services.LSSetDefaultRoleHandlerForContentType(uti_ref, self._K_LS_ROLES_ALL, bundle_ref)
            if status != 0:
                message = f"LSSetDefaultRoleHandlerForContentType failed for {uti} with status {status}"
                raise OSError(message)
        finally:
            self.core_foundation.CFRelease(uti_ref)
            self.core_foundation.CFRelease(bundle_ref)

    def _configure_symbols(self) -> None:
        ctypes = self.ctypes
        self.core_foundation.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.core_foundation.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        self.core_foundation.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        self.core_foundation.CFURLCreateFromFileSystemRepresentation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_bool,
        ]
        self.core_foundation.CFStringGetCStringPtr.restype = ctypes.c_char_p
        self.core_foundation.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.core_foundation.CFStringGetCString.restype = ctypes.c_bool
        self.core_foundation.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        self.core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
        self.core_services.LSRegisterURL.restype = ctypes.c_int32
        self.core_services.LSRegisterURL.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.core_services.UTTypeCreatePreferredIdentifierForTag.restype = ctypes.c_void_p
        self.core_services.UTTypeCreatePreferredIdentifierForTag.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.core_services.LSSetDefaultRoleHandlerForContentType.restype = ctypes.c_int32
        self.core_services.LSSetDefaultRoleHandlerForContentType.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]

    def _cfstring(self, value: str):
        encoding_utf8 = 0x08000100
        return self.core_foundation.CFStringCreateWithCString(None, value.encode("utf-8"), encoding_utf8)

    def _cfurl_from_path(self, path: Path):
        path_bytes = os.fsencode(path)
        return self.core_foundation.CFURLCreateFromFileSystemRepresentation(None, path_bytes, len(path_bytes), path.is_dir())

    def _cfstring_to_str(self, value_ref) -> str:
        encoding_utf8 = 0x08000100
        direct = self.core_foundation.CFStringGetCStringPtr(value_ref, encoding_utf8)
        if direct:
            return direct.decode("utf-8")
        buffer = self.ctypes.create_string_buffer(1024)
        ok = self.core_foundation.CFStringGetCString(value_ref, buffer, len(buffer), encoding_utf8)
        if not ok:
            message = "Failed to decode CoreFoundation string."
            raise ValueError(message)
        return buffer.value.decode("utf-8")
