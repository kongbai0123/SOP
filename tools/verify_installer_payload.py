"""Exercise the real offline payload in a private install directory, never HKCU.

This tests deployment/installed startup, not the NSIS wizard's temp-copy uninstall.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "installer/.build"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--skip-runtime", action="store_true", help="Deploy/remove only when final-binary execution was already refused by OS policy")
    args = parser.parse_args()
    release = args.release.resolve()
    configuration = json.loads((ROOT / "installer/release-config.json").read_text(encoding="utf-8"))
    metadata = json.loads((release / "release-manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((BUILD / ("nsis-" + configuration["productId"]) / "payload-manifest.json").read_text(encoding="utf-8"))
    inventory["productId"] = "SOP.WorkInstruction.Smoke.FullPayload." + uuid.uuid4().hex
    scratch = Path(tempfile.mkdtemp(prefix="real-payload-", dir=BUILD)).resolve()
    if scratch.parent != BUILD.resolve():
        raise ValueError("Private verification directory escaped its expected boundary")
    install = scratch / "安裝 路徑" / "SOP"
    manifest = scratch / "manifest.json"
    write_json(manifest, inventory)
    log = scratch / "deployment.log"
    report = {"identity": inventory["productId"], "version": metadata["version"],
              "registry_or_shortcuts_changed": False, "installation": {}, "runtime": {}, "removal": {},
              "scope": "Real full-size offline payload; direct deployment helper and private data. NSIS GUI/uninstaller not exercised."}
    artifact = next(item for item in metadata["artifacts"] if item["path"].endswith(".zip"))
    powershell = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PATH"] = os.pathsep.join([str(Path(os.environ["SystemRoot"]) / "System32"), os.environ["SystemRoot"]])
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["SOP_LAUNCH_DIAGNOSTICS"] = str(scratch / "launch-diagnostics.jsonl")

    def helper(action):
        command = [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(ROOT / "installer/deploy.ps1"), "-Action", action,
                   "-InstallDir", str(install), "-ProductId", inventory["productId"],
                   "-Version", metadata["version"], "-LogPath", str(log)]
        if action == "Install":
            command += ["-ManifestPath", str(manifest), "-PayloadPath", str(release / artifact["path"]),
                        "-PayloadSha256", artifact["sha256"]]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, timeout=900,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        return {"exitCode": result.returncode, "seconds": round(time.monotonic() - started, 3),
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace")}

    try:
        print("Installing the full verified payload into a private path...", flush=True)
        report["installation"] = helper("Install")
        if report["installation"]["exitCode"] != 0:
            raise RuntimeError("Real-payload deployment failed; see deployment diagnostics")
        installed_marker = json.loads((install / ".sop-install.json").read_text(encoding="utf-8"))
        report["installation"]["ownedFiles"] = len(installed_marker["files"])
        runtime_path = scratch / "runtime.json"
        print("Checking deployment; respecting any known OS execution block...", flush=True)
        try:
            if args.skip_runtime:
                report["runtime"] = {"status": "not-executed", "blockedByOS": True,
                                     "reason": "Final binary was already refused by Windows Application Control (WinError 4551); not retried."}
            else:
                result = subprocess.run([str(install / "SOP.exe"), "--installer-runtime-test", str(runtime_path)],
                                        cwd=scratch, env=environment, timeout=660,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                report["runtime"] = {"exitCode": result.returncode}
                if runtime_path.exists():
                    report["runtime"]["report"] = json.loads(runtime_path.read_text(encoding="utf-8"))
        except OSError as error:
            report["runtime"] = {"error": str(error), "winerror": error.winerror,
                                 "blockedByOS": error.winerror in (4551, 577, 1260)}
        except subprocess.TimeoutExpired as error:
            report["runtime"] = {"error": str(error), "timeout": True}
        unknown = install / "user-note.txt"
        unknown.write_bytes(b"Unowned test file must survive removal")
        print("Verifying owned-file removal and preservation of unknown files...", flush=True)
        report["removal"] = helper("Uninstall")
        report["removal"]["unknownFilePreserved"] = unknown.read_bytes() == b"Unowned test file must survive removal"
        report["removal"]["entryRemoved"] = not (install / "SOP.exe").exists()
        report["deploymentOk"] = (report["removal"]["exitCode"] == 0 and
                                  report["removal"]["unknownFilePreserved"] and report["removal"]["entryRemoved"])
        report["ok"] = (report["deploymentOk"] and report["runtime"].get("exitCode") == 0 and
                        report["runtime"].get("report", {}).get("ok") is True)
        report["status"] = ("deployment_passed_runtime_blocked" if args.skip_runtime and report["deploymentOk"]
                            else "passed" if report["ok"] else "failed")
    except Exception as error:
        report["error"] = str(error)
        report["ok"] = False
    report["diagnosticDirectory"] = str(scratch)
    write_json(release / "real-payload-verification.json", report)
    # Preserve failure evidence. Success cleanup is restricted to this newly
    # created UUID directory directly below the project's build directory.
    scope_passed = report.get("deploymentOk") and (args.skip_runtime or report.get("ok"))
    if scope_passed:
        if scratch.parent != BUILD.resolve() or not scratch.name.startswith("real-payload-"):
            raise ValueError("Refusing cleanup outside the private verification boundary")
        shutil.rmtree(scratch)
        report["diagnosticsCleaned"] = True
        write_json(release / "real-payload-verification.json", report)
    print(json.dumps({"ok": report.get("ok"), "report": str(release / "real-payload-verification.json")}), flush=True)
    return 0 if scope_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
