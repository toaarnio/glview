import plistlib
import tempfile
import unittest
from pathlib import Path

from glview import desktop


class _FakeRegistryKey:
    def __init__(self, path):
        self.path = path


class _FakeRegistryBackend:
    HKEY_CURRENT_USER = "HKCU"
    KEY_WRITE = 0x00020006
    KEY_WOW64_64KEY = 0x0100
    REG_SZ = "REG_SZ"

    def __init__(self):
        self.values = {}

    def CreateKeyEx(self, _hive, path, _reserved, _access):
        return _FakeRegistryKey(path)

    def SetValueEx(self, key, value_name, _reserved, value_type, value):
        self.values[(key.path, value_name)] = (value_type, value)

    def CloseKey(self, _key):
        return None


class _FakeWindowsLauncher:
    def __init__(self):
        self.uris = []

    def open_uri(self, uri):
        self.uris.append(uri)


class _FakeMacOSBackend:
    def __init__(self):
        self.registered_bundles = []
        self.default_handlers = []

    def register_app(self, bundle_path):
        self.registered_bundles.append(Path(bundle_path))

    def uti_for_extension(self, extension):
        mapping = {
            ".jpg": "public.jpeg",
            ".jpeg": "public.jpeg",
            ".png": "public.png",
            ".tif": "public.tiff",
            ".tiff": "public.tiff",
            ".bmp": "com.microsoft.bmp",
            ".exr": "public.exr",
            ".hdr": "public.radiance",
        }
        return mapping.get(extension)

    def set_default_role_handler(self, uti, bundle_id):
        self.default_handlers.append((uti, bundle_id))


class _FakeMimeDatabaseBackend:
    def __init__(self):
        self.updated_dirs = []

    def update_database(self, mime_dir):
        self.updated_dirs.append(Path(mime_dir))


class DesktopIntegrationTests(unittest.TestCase):

    def test_shared_mime_info_xml_contains_netpbm_types(self):
        xml = desktop.shared_mime_info_xml()

        self.assertIn('mime-type type="image/x-portable-floatmap"', xml)
        self.assertIn('glob pattern="*.pfm"', xml)
        self.assertIn('mime-type type="image/x-portable-anymap"', xml)
        self.assertIn('glob pattern="*.pnm"', xml)
        self.assertIn('mime-type type="image/x-portable-pixmap"', xml)
        self.assertIn('glob pattern="*.ppm"', xml)
        self.assertIn('mime-type type="image/x-portable-graymap"', xml)
        self.assertIn('glob pattern="*.pgm"', xml)

    def test_desktop_entry_contains_exec_and_mime_types(self):
        entry = desktop.desktop_entry_text(exec_command="glview")

        self.assertIn("Exec=glview %F", entry)
        self.assertIn("image/jpeg;", entry)
        self.assertIn("image/x-exr;", entry)

    def test_install_default_handler_writes_desktop_entry_and_mimeapps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_home = Path(tmpdir) / "data"
            config_home = Path(tmpdir) / "config"
            mime_database = _FakeMimeDatabaseBackend()

            result = desktop.install_default_handler(
                data_home=data_home,
                config_home=config_home,
                platform_name="linux",
                mime_database_backend=mime_database,
            )

            self.assertEqual(result.desktop_path, data_home / "applications" / "glview.desktop")
            self.assertEqual(result.mimeapps_path, config_home / "mimeapps.list")
            self.assertEqual(result.mime_xml_path, data_home / "mime" / "packages" / "glview.xml")
            self.assertEqual(
                result.mimeapps_paths,
                (
                    config_home / "mimeapps.list",
                    data_home / "applications" / "mimeapps.list",
                ),
            )
            self.assertTrue(result.desktop_path.exists())
            self.assertTrue(result.mimeapps_path.exists())
            self.assertTrue((data_home / "applications" / "mimeapps.list").exists())
            self.assertTrue(result.mime_xml_path.exists())
            self.assertEqual(mime_database.updated_dirs, [data_home / "mime"])

            desktop_text = result.desktop_path.read_text(encoding="utf-8")
            mimeapps_text = result.mimeapps_path.read_text(encoding="utf-8")
            applications_mimeapps_text = (data_home / "applications" / "mimeapps.list").read_text(encoding="utf-8")
            mime_xml_text = result.mime_xml_path.read_text(encoding="utf-8")

            self.assertIn("Exec=glview %F", desktop_text)
            self.assertIn("image/vnd.radiance;", desktop_text)
            self.assertIn("[Default Applications]", mimeapps_text)
            self.assertIn("image/jpeg=glview.desktop;", mimeapps_text)
            self.assertIn("[Added Associations]", mimeapps_text)
            self.assertIn("image/jpeg=glview.desktop;", applications_mimeapps_text)
            self.assertIn('glob pattern="*.pfm"', mime_xml_text)

    def test_install_default_handler_keeps_glview_first_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_home = Path(tmpdir) / "config"
            config_home.mkdir(parents=True, exist_ok=True)
            mimeapps_path = config_home / "mimeapps.list"
            mime_database = _FakeMimeDatabaseBackend()
            mimeapps_path.write_text(
                (
                    "[Added Associations]\n"
                    "image/jpeg=other.desktop;glview.desktop;\n"
                    "[Default Applications]\n"
                    "image/jpeg=other.desktop;\n"
                ),
                encoding="utf-8",
            )

            desktop.install_default_handler(
                data_home=Path(tmpdir) / "data",
                config_home=config_home,
                platform_name="linux",
                mime_database_backend=mime_database,
            )

            mimeapps_text = mimeapps_path.read_text(encoding="utf-8")

            self.assertIn("image/jpeg=glview.desktop;", mimeapps_text)
            self.assertIn("image/jpeg=glview.desktop;other.desktop;", mimeapps_text)

    def test_install_default_handler_rejects_non_linux_platforms(self):
        with self.assertRaises(NotImplementedError):
            desktop.install_default_handler(platform_name="darwin")

    def test_install_default_handler_writes_windows_registry_entries(self):
        registry = _FakeRegistryBackend()
        launcher = _FakeWindowsLauncher()

        result = desktop.install_default_handler(
            exec_command=r"C:\Users\me\AppData\Local\Programs\glview\glview.exe",
            platform_name="win32",
            registry_backend=registry,
            launcher_backend=launcher,
        )

        command_key = r"Software\Classes\Applications\glview.exe\shell\open\command"
        application_open_key = r"Software\Classes\Applications\glview.exe\shell\open"
        supported_types_key = r"Software\Classes\Applications\glview.exe\SupportedTypes"
        capabilities_key = r"Software\glview\Capabilities"
        capabilities_assoc_key = r"Software\glview\Capabilities\FileAssociations"
        png_open_key = r"Software\Classes\glview.png\shell\open"

        self.assertEqual(result.platform, "windows")
        self.assertIn("Opened the Windows Default Apps page", result.note)
        self.assertIn(command_key, result.registry_paths)
        self.assertIn(application_open_key, result.registry_paths)
        self.assertIn(supported_types_key, result.registry_paths)
        self.assertIn(capabilities_key, result.registry_paths)
        self.assertIn(capabilities_assoc_key, result.registry_paths)
        self.assertIn(png_open_key, result.registry_paths)
        self.assertEqual(result.settings_uri, "ms-settings:defaultapps?registeredAppUser=glview")
        self.assertEqual(launcher.uris, [result.settings_uri])
        self.assertEqual(
            registry.values[(command_key, "")],
            ("REG_SZ", r'"C:\Users\me\AppData\Local\Programs\glview\glview-launcher.exe" "%1"'),
        )
        self.assertEqual(
            registry.values[(application_open_key, "MultiSelectModel")],
            ("REG_SZ", "Player"),
        )
        self.assertEqual(registry.values[(supported_types_key, ".png")], ("REG_SZ", ""))
        self.assertEqual(registry.values[(capabilities_key, "ApplicationName")], ("REG_SZ", "glview"))
        self.assertEqual(
            registry.values[(r"Software\RegisteredApplications", "glview")],
            ("REG_SZ", r"Software\glview\Capabilities"),
        )
        self.assertEqual(
            registry.values[(capabilities_assoc_key, ".png")],
            ("REG_SZ", "glview.png"),
        )
        self.assertEqual(
            registry.values[(png_open_key, "MultiSelectModel")],
            ("REG_SZ", "Player"),
        )
        self.assertEqual(
            registry.values[(r"Software\Classes\.png\OpenWithProgids", "glview.png")],
            ("REG_SZ", ""),
        )

    def test_windows_progid_uses_extension_suffix(self):
        self.assertEqual(desktop._windows_progid(".jpg"), "glview.jpg")

    def test_windows_default_apps_uri_escapes_registered_app_name(self):
        uri = desktop._windows_default_apps_uri("glview beta")

        self.assertEqual(uri, "ms-settings:defaultapps?registeredAppUser=glview%20beta")

    def test_windows_launcher_command_uses_gui_launcher_name(self):
        command = desktop._windows_launcher_command(r"C:\Programs\glview\glview.exe")

        self.assertEqual(command, r"C:\Programs\glview\glview-launcher.exe")

    def test_install_default_handler_writes_macos_launch_services_entries(self):
        backend = _FakeMacOSBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "glview.app"
            contents = bundle_path / "Contents"
            macos_dir = contents / "MacOS"
            macos_dir.mkdir(parents=True, exist_ok=True)
            (contents / "Info.plist").write_bytes(
                plistlib.dumps({"CFBundleIdentifier": "com.example.glview"}),
            )
            executable = macos_dir / "glview"
            executable.write_text("", encoding="utf-8")

            result = desktop.install_default_handler(
                platform_name="darwin",
                executable_path=executable,
                bundle_backend=backend,
            )

        self.assertEqual(result.platform, "macos")
        self.assertEqual(result.bundle_id, "com.example.glview")
        self.assertEqual(result.bundle_path, bundle_path)
        self.assertEqual(backend.registered_bundles, [bundle_path])
        self.assertIn(("public.png", "com.example.glview"), backend.default_handlers)
        self.assertIn("proper .app bundle", result.note)

    def test_install_default_handler_rejects_unbundled_macos_executable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executable = Path(tmpdir) / "glview"
            executable.write_text("", encoding="utf-8")

            with self.assertRaises(NotImplementedError):
                desktop.install_default_handler(
                    platform_name="darwin",
                    executable_path=executable,
                    bundle_backend=_FakeMacOSBackend(),
                )


if __name__ == "__main__":
    unittest.main()
