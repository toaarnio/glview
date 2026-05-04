import plistlib
import tempfile
import unittest
from pathlib import Path

from glview import macosapp


class MacOSAppBundleTests(unittest.TestCase):

    def test_build_app_bundle_writes_expected_layout_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "dist"
            source_root = Path(tmpdir) / "src"
            source_root.mkdir()
            python_bin = Path(tmpdir) / "venv" / "bin" / "python3"
            python_bin.parent.mkdir(parents=True)
            python_bin.write_text("", encoding="utf-8")

            bundle_path = macosapp.build_app_bundle(
                output_dir=output_dir,
                python_executable=python_bin,
                source_root=source_root,
                bundle_id="com.example.glview",
                bundle_name="glview",
                version_string="1.22.2",
            )

            info_plist = bundle_path / "Contents" / "Info.plist"
            launcher = bundle_path / "Contents" / "MacOS" / "glview"
            pkginfo = bundle_path / "Contents" / "PkgInfo"

            self.assertEqual(bundle_path, output_dir / "glview.app")
            self.assertTrue(info_plist.exists())
            self.assertTrue(launcher.exists())
            self.assertTrue(pkginfo.exists())

            plist = plistlib.loads(info_plist.read_bytes())

            self.assertEqual(plist["CFBundleIdentifier"], "com.example.glview")
            self.assertEqual(plist["CFBundleExecutable"], "glview")
            self.assertEqual(plist["CFBundlePackageType"], "APPL")
            self.assertEqual(plist["CFBundleShortVersionString"], "1.22.2")
            self.assertEqual(plist["CFBundleVersion"], "1.22.2")
            self.assertEqual(plist["CFBundleDocumentTypes"][0]["CFBundleTypeRole"], "Viewer")
            self.assertIn("public.png", plist["CFBundleDocumentTypes"][0]["LSItemContentTypes"])

            launcher_text = launcher.read_text(encoding="utf-8")
            self.assertIn(str(python_bin), launcher_text)
            self.assertIn(str(source_root), launcher_text)
            self.assertIn('exec "$PYTHON_BIN" -m glview.glview "$@"', launcher_text)
            self.assertEqual(pkginfo.read_text(encoding="ascii"), "APPL????")

    def test_bundle_version_extracts_numeric_components(self):
        self.assertEqual(macosapp._bundle_version("2.3.4"), "2.3.4")
        self.assertEqual(macosapp._bundle_version("v2.3.4-beta1"), "2.3.4")
        self.assertEqual(macosapp._bundle_version("release-7"), "7")
        self.assertEqual(macosapp._bundle_version("dev"), "0.0.0")


if __name__ == "__main__":
    unittest.main()
