"""Signing build guards with inert bytes and mocked Authenticode results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from tools import prepare_installer, windows_signing


class WindowsReleaseSigningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sop-signing-tests-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.signer = Mock()
        self.signer.thumbprint = "A" * 40
        self.signer.sign.side_effect = self._fake_sign
        self.signer.uninstaller_environment.side_effect = lambda report: {
            windows_signing.UNINSTALLER_REPORT: str(report)
        }

    def _record(self, data):
        return {"status": "Valid", "thumbprint": self.signer.thumbprint, "timestamped": True,
                "sha256": hashlib.sha256(data).hexdigest()}

    def _fake_sign(self, path):
        data = path.read_bytes() + b" [test signature placeholder]"
        path.write_bytes(data)
        return self._record(data)

    def test_required_signing_without_credentials_stops_before_building(self):
        output = self.root / "never-created"
        args = SimpleNamespace(payload=self.root / "payload", output=output, rebuild=False)
        with patch.object(prepare_installer, "compiler") as compiler:
            with self.assertRaisesRegex(ValueError, "Signing requires"):
                prepare_installer.prepare(args, {"signingPolicy": "required"}, "2026.10.19.1")
        compiler.assert_not_called()
        self.assertFalse(output.exists())

    def test_cannot_rebuild_in_directory_with_existing_verification(self):
        output = self.root / "existing"
        output.mkdir()
        report = output / "verification.json"
        report.write_bytes(b"original evidence")
        args = SimpleNamespace(payload=self.root / "payload", output=output, rebuild=True)
        with patch.object(prepare_installer, "compiler") as compiler:
            with self.assertRaisesRegex(FileExistsError, "Existing delivery verification"):
                prepare_installer.prepare(args, {"signingPolicy": "optional"}, "2026.10.19.1")
        compiler.assert_not_called()
        self.assertEqual(report.read_bytes(), b"original evidence")

    def test_preflight_rejects_unsigned_vendor_and_invalid_signatures_without_signing(self):
        vendor = self.root / "vendor.dll"
        vendor.write_bytes(b"fixture")
        name = "payload/_internal/vendor.dll"
        for status, approved in (
            ("NotSigned", {}),
            ("NotSigned", {name: "0" * 64}),
            ("HashMismatch", {name: hashlib.sha256(b"fixture").hexdigest()}),
        ):
            with self.subTest(status=status, approved=approved), patch.object(
                windows_signing, "_signatures", return_value=[{"status": status}]
            ):
                with self.assertRaisesRegex(ValueError, "vendor.dll"):
                    windows_signing.prepare_components(self.signer, {name: vendor}, set(), approved,
                                                       self.root / "staged")
        self.signer.sign.assert_not_called()
        self.assertEqual(vendor.read_bytes(), b"fixture")

    def test_stages_both_trusted_and_authorized_components_before_hashing(self):
        signed, authorized = self.root / "trusted.dll", self.root / "unsigned.dll"
        signed.write_bytes(b"valid vendor")
        authorized.write_bytes(b"authorized vendor")
        trusted_name, unsigned_name = "payload/_internal/trusted.dll", "payload/_internal/unsigned.dll"
        components = {trusted_name: signed, unsigned_name: authorized}
        approvals = {unsigned_name: hashlib.sha256(b"authorized vendor").hexdigest()}
        scratch = self.root / "staged"

        def inventory(paths):
            self.assertTrue(all(path.is_relative_to(scratch) for path in paths))
            # Original input can change after staging without changing the release bytes.
            signed.write_bytes(b"source changed later")
            return [{"status": "Valid"}, {"status": "NotSigned"}]

        with patch.object(windows_signing, "_signatures", side_effect=inventory):
            staged, reports = windows_signing.prepare_components(self.signer, components, set(), approvals, scratch)
        self.signer.verify_vendor.assert_called_once_with([staged[trusted_name]])
        self.signer.sign.assert_called_once_with(staged[unsigned_name])
        self.assertEqual(staged[trusted_name].read_bytes(), b"valid vendor")
        self.assertEqual(authorized.read_bytes(), b"authorized vendor")
        self.assertEqual(reports[unsigned_name]["sha256"], windows_signing.file_sha256(staged[unsigned_name]))

    def test_compiler_temp_uninstaller_is_signed_using_exe_extension(self):
        path = self.root / "nsis-file.tmp"
        path.write_bytes(b"MZ inert temp PE stub")
        receipt = self.root / "uninstaller-receipt.json"
        settings = {"tool": str(self.root / "signtool.exe"), "thumbprint": self.signer.thumbprint,
                    "timestamp_url": "https://timestamp.example.invalid"}
        env = {"TEMP": str(self.root), windows_signing.UNINSTALLER_SETTINGS: json.dumps(settings),
               windows_signing.UNINSTALLER_REPORT: str(receipt)}
        with patch.dict(windows_signing.os.environ, env), patch.object(
            windows_signing.WindowsSigner, "configured", return_value=self.signer
        ):
            windows_signing._sign_uninstaller(path)
        self.assertEqual(self.signer.sign.call_args.args[0].suffix, ".exe")
        self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["sha256"], windows_signing.file_sha256(path))
        self.assertIn(b"signature placeholder", path.read_bytes())

    def test_signature_requires_selected_publisher_and_timestamp(self):
        signtool = windows_signing.WindowsSigner(Path("inert-signtool.exe"), "A" * 40, "https://example.invalid")
        for candidate in (
            {"status": "Valid", "thumbprint": "A" * 40, "timestamped": False},
            {"status": "Valid", "thumbprint": "B" * 40, "timestamped": True},
            {"status": "NotTrusted", "thumbprint": "A" * 40, "timestamped": True},
            {"status": "Valid", "thumbprint": "A" * 40, "timestamped": True, "selfIssued": True},
        ):
            with self.subTest(candidate=candidate), patch.object(windows_signing.subprocess, "run") as verify:
                with self.assertRaisesRegex(ValueError, "valid timestamped"):
                    signtool.verify_owned(self.root / "not-executed.exe", candidate)
                verify.assert_not_called()

    def _package_fixture(self):
        payload = self.root / "payload"
        payload.mkdir()
        (payload / "SOP.exe").write_bytes(b"unsigned fixture")
        installer = self.root / "installer"
        (installer / ".build/generated").mkdir(parents=True)
        (installer / ".build/generated/SOP.ico").write_bytes(b"test icon")
        (installer / "README.md").write_text("test")
        (installer / "deploy.ps1").write_text("# test deploy")
        (installer / "shortcuts.ps1").write_text("# test shortcuts")
        (self.root / "tools").mkdir()
        (self.root / "tools/check_windows_trust.ps1").write_text("# test trust")
        config = {"productId": "SOP.UnitTest", "productName": "Unit test", "entryExe": "SOP.exe",
                  "publisher": "Unit test", "nsisVersion": "mock", "architecture": "x64",
                  "installScope": "current-user", "channel": "test", "signingPolicy": "required",
                  "updateMode": "manual", "distribution": "local"}
        args = SimpleNamespace(payload=payload, output=self.root / "out", rebuild=True)
        return args, config, installer

    def _fixture_compiler(self, _config, *, unpack_to):
        plugins = unpack_to / "Plugins/x86-unicode"
        plugins.mkdir(parents=True)
        for name in prepare_installer.NSIS_RUNTIME_PLUGINS:
            (plugins / name).write_bytes(b"test plugin")
        return unpack_to / "makensis.exe"

    def test_required_build_hashes_only_after_signing(self):
        args, config, installer = self._package_fixture()
        setup_name = "SOP-2026.10.19.1-windows-x64-Setup.exe"

        def mock_compile(command, *, env, **_options):
            wrapper = Path(command[-1])
            self.assertIn("!uninstfinalize", wrapper.read_text(encoding="utf-8"))
            out = wrapper.parent.parent / "delivery"
            (out / setup_name).write_bytes(b"test setup")
            Path(env[windows_signing.UNINSTALLER_REPORT]).write_text(json.dumps(self._record(b"test uninstaller")))

        def inventory(paths):
            return [{"status": "Valid" if path.name in prepare_installer.NSIS_RUNTIME_PLUGINS else "NotSigned"}
                    for path in paths]

        with patch.object(prepare_installer.WindowsSigner, "configured", return_value=self.signer), \
                patch.object(prepare_installer, "ROOT", self.root), \
                patch.object(prepare_installer, "INSTALLER", installer), \
                patch.object(prepare_installer, "compiler", side_effect=self._fixture_compiler), \
                patch.object(windows_signing, "_signatures", side_effect=inventory), \
                patch.object(prepare_installer.subprocess, "run", side_effect=mock_compile), \
                patch.object(prepare_installer.subprocess, "check_output", return_value=""):
            prepare_installer.prepare(args, config, "2026.10.19.1")
        manifest = json.loads((args.output / "release-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["signingStatus"], "Valid")
        for deliverable in manifest["artifacts"]:
            self.assertEqual(deliverable["sha256"], windows_signing.file_sha256(args.output / deliverable["path"]))
        with zipfile.ZipFile(args.output / "SOP-2026.10.19.1-windows-x64.payload.zip") as archive:
            digest = hashlib.sha256(archive.read("SOP.exe")).hexdigest()
        self.assertEqual(manifest["authenticode"]["components"]["payload/SOP.exe"]["sha256"], digest)
        self.assertEqual((args.payload / "SOP.exe").read_bytes(), b"unsigned fixture")

    def test_uninstaller_verification_failure_preserves_prior_delivery(self):
        args, config, installer = self._package_fixture()
        args.output.mkdir()
        old_setup = args.output / "SOP-2026.10.19.1-windows-x64-Setup.exe"
        old_setup.write_bytes(b"prior installer")

        def inventory(paths):
            return [{"status": "Valid" if path.name in prepare_installer.NSIS_RUNTIME_PLUGINS else "NotSigned"}
                    for path in paths]

        with patch.object(prepare_installer.WindowsSigner, "configured", return_value=self.signer), \
                patch.object(prepare_installer, "ROOT", self.root), \
                patch.object(prepare_installer, "INSTALLER", installer), \
                patch.object(prepare_installer, "compiler", side_effect=self._fixture_compiler), \
                patch.object(windows_signing, "_signatures", side_effect=inventory), \
                patch.object(prepare_installer.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "uninstaller signature report"):
                prepare_installer.prepare(args, config, "2026.10.19.1")
        self.assertEqual(old_setup.read_bytes(), b"prior installer")
        self.assertEqual(list(args.output.iterdir()), [old_setup])

    def test_missing_or_mismatched_onnx_marker_does_not_prepare_an_installer(self):
        args, config, _installer = self._package_fixture()
        args.runtime = "onnx"
        config["signingPolicy"] = "optional"
        with patch.object(prepare_installer, "compiler") as compiler:
            with self.assertRaisesRegex(ValueError, "Missing runtime identity"):
                prepare_installer.prepare(args, config, "2026.10.19.1")
            marker = args.payload / "_internal/sop-runtime.txt"
            marker.parent.mkdir()
            marker.write_text("torch", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime does not match"):
                prepare_installer.prepare(args, config, "2026.10.19.1")
        compiler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
