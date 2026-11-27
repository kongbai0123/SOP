"""Windows signature inventory inspection without executing fixture files."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_windows_trust.ps1"
POWERSHELL = shutil.which("pwsh.exe") or shutil.which("powershell.exe")


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Requires PowerShell on Windows")
class WindowsTrustInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sop-trust-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.payload = self.root / "payload"
        self.payload.mkdir()
        artifact = b"Fixture data"
        (self.release / "data.zip").write_bytes(artifact)
        (self.release / "release-manifest.json").write_text(json.dumps({
            "artifacts": [{"path": "data.zip", "size": len(artifact),
                           "sha256": hashlib.sha256(artifact).hexdigest()}],
        }), encoding="utf-8")

    def inspect(self, payload=True):
        command = [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(TOOL),
                   "-ReleaseDirectory", str(self.release), "-MaximumEvents", "1"]
        if payload:
            command += ["-PayloadDirectory", str(self.payload)]
        process = subprocess.run(command, capture_output=True, encoding="utf-8",
                                 errors="replace", timeout=60)
        if process.returncode and not process.stdout and all(
                error in process.stderr for error in ("SecurityError", "UnauthorizedAccess")):
            self.skipTest("The host policy rejected the diagnostic script")
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_reports_recursive_files_by_signature_result_without_executing_them(self):
        fixtures = ["SOP.exe", "_internal/fixture.dll", "_internal/sub/fixture.pyd",
                    "scripts/fixture.ps1", "scripts/ignored.cmd"]
        for name in fixtures:
            path = self.payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("throw 'inspect without execution'", encoding="ascii")

        report = self.inspect()
        inventory = report["payloadSignatureInventory"]
        signatures = {entry["relativePath"].replace("\\", "/"): entry for entry in report["signatures"]}
        self.assertEqual(report["integrity"]["status"], "Passed")
        self.assertFalse(report["artifactExecutionPerformed"])
        self.assertEqual(inventory["status"], "Collected")
        self.assertEqual(inventory["fileCount"], 4)
        self.assertEqual(set(signatures), set(fixtures[:4]))
        self.assertEqual(signatures["SOP.exe"]["role"], "explicit-payload-entry")
        self.assertEqual(signatures["_internal/sub/fixture.pyd"]["role"], "payload-component")
        self.assertEqual(inventory["statusCounts"], dict(Counter(sig["status"] for sig in signatures.values())))
        self.assertIn("scripts/fixture.ps1", {name.replace("\\", "/") for name in inventory["unsignedPaths"]})
        self.assertEqual({name.replace("\\", "/") for name in inventory["nonValidPaths"]}, set(signatures))
        self.assertEqual(inventory["uninspectedEntries"], [])

    def test_omitted_payload_keeps_original_report_and_missing_entry_is_reported(self):
        skipped = self.inspect(payload=False)
        self.assertEqual(skipped["payloadSignatureInventory"]["status"], "NotRequested")
        self.assertEqual(skipped["signatures"], [])

        missing = self.inspect()
        inventory = missing["payloadSignatureInventory"]
        self.assertEqual(inventory["status"], "Partial")
        self.assertEqual(inventory["fileCount"], 0)
        self.assertEqual(inventory["unsignedPaths"], [])
        self.assertEqual(inventory["nonValidPaths"], ["SOP.exe"])
        self.assertEqual(missing["signatures"][0]["status"], "Missing")

    def test_junction_contents_are_never_inventoried_through_the_link(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "unrelated.dll").write_bytes(b"outside inventory")
        (self.payload / "SOP.exe").write_bytes(b"test-only payload")
        link = self.payload / "linked"
        command = ("New-Item -ItemType Junction -Path $env:SOP_TRUST_TEST_LINK "
                   "-Target $env:SOP_TRUST_TEST_TARGET -ErrorAction Stop | Out-Null")
        process = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
                                 env={**os.environ, "SOP_TRUST_TEST_LINK": str(link),
                                      "SOP_TRUST_TEST_TARGET": str(outside)},
                                 capture_output=True, timeout=30)
        if process.returncode:
            self.skipTest("Cannot create a junction on the current volume")
        report = self.inspect()
        inventory = report["payloadSignatureInventory"]
        self.assertEqual(inventory["status"], "Partial")
        self.assertEqual(inventory["fileCount"], 1)
        self.assertEqual([entry["relativePath"] for entry in inventory["uninspectedEntries"]], ["linked"])
        self.assertEqual([entry["relativePath"] for entry in report["signatures"]], ["SOP.exe"])


if __name__ == "__main__":
    unittest.main()
