"""Windows installer safety checks using private payloads and product identities.

No test starts the application, a model, or a camera. NSIS integration requires
SOP_RUN_NSIS_INTEGRATION=1 and the pinned compiler from build_installer.ps1.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "installer" / "deploy.ps1"
NSIS = ROOT / "installer" / ".tools" / "nsis-3.13" / "makensis.exe"
POWERSHELL = shutil.which("powershell.exe")
MARKER = ".sop-install.json"
REG_BASE = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"


@unittest.skipUnless(os.name == "nt" and POWERSHELL and HELPER.exists(),
                     "Requires Windows PowerShell and the installer helper")
class InstallerHelperSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sop-installer-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.install = self.root / "\u7e41\u4e2d \u7a7a\u767d" / "SOP"
        self.product = "SOP.WorkInstruction.Smoke." + uuid.uuid4().hex
        self.log = self.root / "deploy.log"

    def payload(self, version="1.0.0", files=None, entries=None):
        files = files or {"SOP.exe": b"test-only payload; never execute", "lib/old.txt": b"old"}
        directory = self.root / uuid.uuid4().hex
        directory.mkdir()
        manifest = directory / "manifest.json"
        document = {
            "productId": self.product,
            "version": version,
            "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                       "size": len(data)} for name, data in files.items()],
        }
        manifest.write_text(json.dumps(document), encoding="utf-8")
        archive = directory / "payload.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            for name, data in (entries if entries is not None else files.items()):
                output.writestr(name, data)
        return {"version": version, "manifest": manifest, "archive": archive,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "document": document}

    def invoke(self, action, payload=None, install=None, product=None, version=None, expected=0):
        args = [POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-File", str(HELPER),
                "-Action", action, "-InstallDir", str(install or self.install),
                "-ProductId", product or self.product, "-Version",
                version or (payload["version"] if payload else "1.0.0"),
                "-LogPath", str(self.log)]
        if payload:
            args.extend(["-ManifestPath", str(payload["manifest"]),
                         "-PayloadPath", str(payload["archive"]),
                         "-PayloadSha256", payload["sha256"]])
        result = subprocess.run(args, capture_output=True, timeout=45)
        diagnostic = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if expected == 0:
            self.assertEqual(result.returncode, 0, diagnostic)
        else:
            self.assertNotEqual(result.returncode, 0, diagnostic)
        return result

    def test_install_reinstall_upgrade_and_uninstall_preserve_unknown_files(self):
        initial = self.payload()
        self.invoke("Validate", initial)
        self.assertFalse(self.install.exists())
        self.invoke("Install", initial)
        self.invoke("Install", initial)
        unknown = self.install / "user documents" / "notes.txt"
        unknown.parent.mkdir()
        unknown.write_bytes(b"must survive upgrades and removal")
        upgrade = self.payload("1.1.0", {"SOP.exe": b"new test payload", "lib/new.txt": b"new"})
        self.invoke("Install", upgrade)
        self.assertFalse((self.install / "lib/old.txt").exists())
        self.assertEqual((self.install / "SOP.exe").read_bytes(), b"new test payload")
        marker = json.loads((self.install / MARKER).read_text(encoding="utf-8-sig"))
        self.assertEqual((marker["productId"], marker["version"], marker["scope"]),
                         (self.product, "1.1.0", "current-user"))
        self.invoke("Uninstall", version="1.1.0")
        self.assertFalse((self.install / "SOP.exe").exists())
        self.assertFalse((self.install / "lib").exists())
        self.assertEqual(unknown.read_bytes(), b"must survive upgrades and removal")
        # NSIS removes the marker only after its own metadata cleanup succeeds.
        self.assertTrue((self.install / MARKER).is_file())

    def test_hash_and_traversal_fail_before_changing_installed_payload(self):
        initial = self.payload()
        self.invoke("Install", initial)
        marker = (self.install / MARKER).read_bytes()
        invalid_hash = self.payload("1.1.0", {"SOP.exe": b"different"})
        invalid_hash["sha256"] = "0" * 64
        self.invoke("Install", invalid_hash, expected=1)
        for path in ("../escape.txt", "lib/../../escape.txt", "C:/escape.txt", "SOP.exe:stream"):
            with self.subTest(path=path):
                malicious = self.payload("1.1.0", {"SOP.exe": b"different"},
                                         entries=[("SOP.exe", b"different"), (path, b"no")])
                self.invoke("Install", malicious, expected=1)
        self.assertEqual((self.install / MARKER).read_bytes(), marker)
        self.assertEqual((self.install / "SOP.exe").read_bytes(), b"test-only payload; never execute")
        self.assertFalse((self.install.parent / "escape.txt").exists())
        self.assertFalse(list(self.install.parent.glob(".sop-stage-*")))

    def test_non_product_directories_and_unknown_file_collisions_are_rejected(self):
        initial = self.payload()
        self.install.mkdir(parents=True)
        sentinel = self.install / "existing.txt"
        sentinel.write_bytes(b"unrelated")
        self.invoke("Install", initial, expected=1)
        self.assertEqual(sentinel.read_bytes(), b"unrelated")
        sentinel.unlink()
        self.invoke("Install", initial)
        unknown = self.install / "new.txt"
        unknown.write_bytes(b"not owned")
        upgrade = self.payload("1.1.0", {"SOP.exe": b"new", "new.txt": b"overwrite attempt"})
        self.invoke("Install", upgrade, expected=1)
        self.assertEqual(unknown.read_bytes(), b"not owned")
        self.assertEqual((self.install / "SOP.exe").read_bytes(), b"test-only payload; never execute")

    def test_identity_downgrade_and_wrong_uninstaller_version_are_rejected(self):
        initial = self.payload("2.0.0")
        self.invoke("Install", initial)
        marker = (self.install / MARKER).read_bytes()
        self.invoke("Install", self.payload("1.9.0"), expected=1)
        self.invoke("Validate", product=self.product + ".Other", version="2.0.0", expected=1)
        self.invoke("Uninstall", version="2.1.0", expected=1)
        self.assertEqual((self.install / MARKER).read_bytes(), marker)
        self.assertTrue((self.install / "SOP.exe").is_file())

    def test_locked_file_leaves_installation_and_marker_available_for_retry(self):
        initial = self.payload()
        self.invoke("Install", initial)
        marker = (self.install / MARKER).read_bytes()
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.CreateFileW(str(self.install / "SOP.exe"), 0x80000000, 0, None, 3, 0, None)
        self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
        try:
            self.invoke("Install", self.payload("1.1.0"), expected=1)
            self.invoke("Uninstall", expected=1)
            self.assertEqual((self.install / MARKER).read_bytes(), marker)
            self.assertTrue((self.install / "lib/old.txt").is_file())
        finally:
            kernel.CloseHandle(handle)
        self.invoke("Uninstall")
        self.assertFalse((self.install / "SOP.exe").exists())

    def test_junction_ancestor_is_rejected_without_touching_its_target(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_bytes(b"preserve")
        link = self.root / "junction"
        created = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
                                 capture_output=True, timeout=10)
        if created.returncode:
            self.skipTest("Junction creation is unavailable on this test host")
        try:
            self.invoke("Install", self.payload(), install=link / "SOP", expected=1)
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertFalse((outside / "SOP").exists())
        finally:
            # Remove only the junction itself, never its target tree.
            os.rmdir(link)

    def test_registered_install_location_cannot_be_silently_moved(self):
        import winreg

        name = REG_BASE + "\\" + self.product
        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, name, 0,
                                 winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY)
        try:
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(self.root / "registered"))
        finally:
            key.Close()
        try:
            self.invoke("Validate", self.payload(), expected=1)
            self.invoke("Install", self.payload(), expected=1)
            self.assertFalse(self.install.exists())
        finally:
            winreg.DeleteKeyEx(winreg.HKEY_CURRENT_USER, name, winreg.KEY_WOW64_64KEY)

    def test_failed_upgrade_restores_previous_files_and_marker(self):
        initial = self.payload()
        self.invoke("Install", initial)
        marker = (self.install / MARKER).read_bytes()
        blocked = self.install / "write-blocked"
        blocked.mkdir()
        sid_query = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
                                    "[Security.Principal.WindowsIdentity]::GetCurrent().User.Value"],
                                   capture_output=True, timeout=10, check=True)
        sid = sid_query.stdout.decode("ascii").strip()
        # Deny creating files only in this freshly created test directory.
        result = subprocess.run(["icacls.exe", str(blocked), "/deny", "*" + sid + ":(WD)"],
                                capture_output=True, timeout=10)
        if result.returncode:
            self.skipTest("A temporary directory ACL cannot be set on this host")
        try:
            upgrade = self.payload("1.1.0", {"SOP.exe": b"changed before the denied write",
                                            "write-blocked/new.txt": b"must fail"})
            self.invoke("Install", upgrade, expected=1)
            self.assertEqual((self.install / "SOP.exe").read_bytes(), b"test-only payload; never execute")
            self.assertEqual((self.install / "lib/old.txt").read_bytes(), b"old")
            self.assertEqual((self.install / MARKER).read_bytes(), marker)
            self.assertFalse((blocked / "new.txt").exists())
            self.assertIn("restoring the previous files", self.log.read_text(encoding="utf-8"))
        finally:
            restored = subprocess.run(["icacls.exe", str(blocked), "/remove:d", "*" + sid],
                                      capture_output=True, timeout=10)
            self.assertEqual(restored.returncode, 0, restored.stdout.decode(errors="replace"))


@unittest.skipUnless(os.environ.get("SOP_RUN_NSIS_INTEGRATION") == "1" and
                     os.name == "nt" and NSIS.is_file() and POWERSHELL,
                     "Opt in with SOP_RUN_NSIS_INTEGRATION=1 on an isolated Windows test host")
class NsisInstallerLifecycleTests(unittest.TestCase):
    def test_isolated_silent_install_upgrade_shortcuts_and_normal_uninstall(self):
        import winreg

        with tempfile.TemporaryDirectory(prefix="sop-nsis-") as temporary:
            root = Path(temporary)
            product = "SOP.WorkInstruction.Smoke." + uuid.uuid4().hex
            name = "SOP Smoke " + uuid.uuid4().hex
            key_name = REG_BASE + "\\" + product
            install = root / "\u5b89\u88dd \u8def\u5f91" / "SOP"
            try:
                winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_name, 0,
                               winreg.KEY_READ | winreg.KEY_WOW64_64KEY).Close()
            except FileNotFoundError:
                pass
            else:
                self.fail("The generated test product identity unexpectedly exists")

            query = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
                                    "@([Environment]::GetFolderPath('Desktop'),"
                                    "[Environment]::GetFolderPath('Programs')) | ConvertTo-Json -Compress"],
                                   capture_output=True, timeout=15, check=True)
            folders = json.loads(query.stdout.decode("utf-8-sig", errors="strict"))
            links = [Path(folder) / (name + ".lnk") for folder in folders]
            self.assertTrue(all(not link.exists() for link in links))

            def build(version, files):
                directory = root / ("build-" + version)
                directory.mkdir()
                archive = directory / "fixture.zip"
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                    for relative, data in files.items():
                        output.writestr(relative, data)
                manifest = directory / "manifest.json"
                manifest.write_text(json.dumps({"productId": product, "version": version,
                    "files": [{"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                               "size": len(data)} for relative, data in files.items()]}), encoding="utf-8")
                setup = directory / ("SOPWizard-" + version + ".exe")
                defines = {
                    "PRODUCT_ID": product, "PRODUCT_NAME": name, "PRODUCT_VERSION": version,
                    "FILE_VERSION": version + ".0", "PUBLISHER": "Installer Test Only",
                    "ENTRY_EXE": "SOP.exe", "PAYLOAD_NAME": archive.name,
                    "PAYLOAD_HASH": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "PAYLOAD_MANIFEST": str(manifest), "OUTPUT_EXE": str(setup),
                    "ICON_FILE": str(NSIS.parent / "Contrib/Graphics/Icons/modern-install.ico"),
                    "ESTIMATED_KB": "1", "DEPLOY_SCRIPT": str(HELPER),
                    "SHORTCUT_SCRIPT": str(ROOT / "installer/shortcuts.ps1"),
                }
                (directory / "Product.nsh").write_text("\n".join(
                    '!define ' + key + ' "' + value + '"' for key, value in defines.items()), encoding="utf-8")
                result = subprocess.run([str(NSIS), "/V2", "/WX", "/INPUTCHARSET", "UTF8",
                                         '/X!addincludedir "' + str(directory) + '"',
                                         str(ROOT / "installer/Setup.nsi")], capture_output=True, timeout=45)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))
                return setup

            def run_setup(setup, directory=install):
                # NSIS /D= is the final argument and its value is not quoted.
                command = '"' + str(setup) + '" /S /DESKTOP=1 /STARTMENU=1 /D=' + str(directory)
                return subprocess.run(command, capture_output=True, timeout=60)

            def registry_value(field):
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_name, 0,
                                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                    return winreg.QueryValueEx(key, field)[0]

            try:
                first = build("1.0.0", {"SOP.exe": b"fixture; do not run", "lib/old.txt": b"old"})
                second = build("1.1.0", {"SOP.exe": b"new fixture; do not run", "lib/new.txt": b"new"})
                self.assertEqual(run_setup(first).returncode, 0)
                self.assertEqual(registry_value("InstallLocation"), str(install))
                self.assertEqual(registry_value("DisplayVersion"), "1.0.0")
                self.assertTrue(all(link.is_file() for link in links))
                unknown = install / "user-note.txt"
                unknown.write_bytes(b"preserve")
                self.assertEqual(run_setup(second).returncode, 0)
                self.assertEqual(registry_value("DisplayVersion"), "1.1.0")
                self.assertFalse((install / "lib/old.txt").exists())
                self.assertNotEqual(run_setup(first).returncode, 0)
                self.assertEqual((install / "SOP.exe").read_bytes(), b"new fixture; do not run")
                self.assertNotEqual(run_setup(second, root / "moved").returncode, 0)
                self.assertFalse((root / "moved").exists())
                uninstall = install / "Uninstall.exe"
                result = subprocess.run([str(uninstall), "/S"], capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and (uninstall.exists() or (install / MARKER).exists()):
                    time.sleep(0.2)
                self.assertFalse(uninstall.exists(), "Normal /S must remove the original uninstaller")
                self.assertFalse((install / MARKER).exists())
                self.assertFalse((install / "SOP.exe").exists())
                self.assertEqual(unknown.read_bytes(), b"preserve")
                self.assertTrue(all(not link.exists() for link in links))
                with self.assertRaises(FileNotFoundError):
                    registry_value("InstallLocation")
            finally:
                # These exact names contain fresh UUIDs and were absent initially.
                for link in links:
                    link.unlink(missing_ok=True)
                try:
                    winreg.DeleteKeyEx(winreg.HKEY_CURRENT_USER, key_name, winreg.KEY_WOW64_64KEY)
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    unittest.main()
