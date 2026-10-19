"""Prepare the offline NSIS payload and pinned local compiler; no publishing."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "installer"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def app_version():
    tree = ast.parse((ROOT / "sop_app/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Application version is missing")


def compiler(config):
    tool_root = INSTALLER / ".tools"
    tool_root.mkdir(parents=True, exist_ok=True)
    version = config["nsisVersion"]
    archive = tool_root / f"nsis-{version}.zip"
    if not archive.exists() or digest(archive) != config["nsisSha256"]:
        print(f"Downloading official NSIS {version}...", flush=True)
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(config["nsisUrl"], temporary)
        if digest(temporary) != config["nsisSha256"]:
            temporary.unlink()
            raise ValueError("NSIS download SHA-256 differs from the pinned official checksum")
        temporary.replace(archive)
    # Always refresh compiler bytes from the verified archive, not an unverified PATH tool.
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
                raise ValueError("Unsafe compiler archive")
        bundle.extractall(tool_root)
    return tool_root / f"nsis-{version}" / "makensis.exe"


def nsis_string(value):
    return str(value).replace("$", "$$").replace('"', '$\\"')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(args, config, version):
    payload = args.payload.resolve()
    if not (payload / config["entryExe"]).is_file():
        raise ValueError(f"Frozen entry point missing: {payload / config['entryExe']}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = INSTALLER / ".build" / ("nsis-" + config["productId"])
    generated.mkdir(parents=True, exist_ok=True)
    stem = f"SOP-{version}-windows-x64"
    setup = output / f"{stem}-Setup.exe"
    if setup.exists() and not args.rebuild:
        raise FileExistsError(f"Existing delivery is preserved: {setup}. Choose another output or use --rebuild for a local test.")
    archive = output / f"{stem}.payload.zip"
    inventory_path = generated / "payload-manifest.json"
    print("Hashing clean frozen payload...", flush=True)
    files = []
    for path in sorted(payload.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Links cannot be packaged: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(payload).as_posix()
        if relative in ("Uninstall.exe", ".sop-install.json"):
            raise ValueError(f"Reserved installation file: {relative}")
        files.append({"path": relative, "size": path.stat().st_size, "sha256": digest(path)})
    manifest = {"productId": config["productId"], "version": version, "files": files}
    old_manifest = None
    if inventory_path.exists():
        old_manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    cache_path = generated / "archive-cache.json"
    reusable = archive.exists() and old_manifest == manifest and cache_path.exists()
    if reusable:
        cache = json.loads(cache_path.read_text())
        reusable = cache.get("archive") == str(archive) and cache.get("sha256") == digest(archive)
    if not reusable:
        print(f"Compressing {sum(f['size'] for f in files) / 2**30:.2f} GiB into the offline payload...", flush=True)
        partial = archive.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as bundle:
            for record in files:
                bundle.write(payload / record["path"], record["path"])
        partial.replace(archive)
    archive_hash = digest(archive)
    write_json(inventory_path, manifest)
    write_json(cache_path, {"archive": str(archive), "sha256": archive_hash})
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() and int(part) <= 65535 for part in parts):
        raise ValueError("A release version with three numeric parts is required")
    icon = INSTALLER / ".build/generated/SOP.ico"
    if not icon.exists():
        raise ValueError("Product icon missing; first run tools/build_payload.py")
    defines = {
        "PRODUCT_ID": config["productId"], "PRODUCT_NAME": config["productName"],
        "PRODUCT_VERSION": version, "FILE_VERSION": version + ".0", "PUBLISHER": config["publisher"],
        "ENTRY_EXE": config["entryExe"], "OUTPUT_EXE": setup, "PAYLOAD_NAME": archive.name,
        "PAYLOAD_HASH": archive_hash, "PAYLOAD_MANIFEST": inventory_path,
        "ICON_FILE": icon, "DEPLOY_SCRIPT": INSTALLER / "deploy.ps1",
        "SHORTCUT_SCRIPT": INSTALLER / "shortcuts.ps1",
    }
    header = generated / "Product.nsh"
    header.write_text("\n".join(f'!define {key} "{nsis_string(value)}"' for key, value in defines.items())
                      + f"\n!define ESTIMATED_KB {(sum(f['size'] for f in files) + 1023) // 1024}\n", encoding="utf-8")
    wrapper = generated / "Compile.nsi"
    wrapper.write_text(f'!include "{nsis_string(header)}"\n!include "{nsis_string(INSTALLER / "Setup.nsi")}"\n', encoding="utf-8")
    tool = compiler(config)
    print("Compiling NSIS Modern UI 2...", flush=True)
    subprocess.run([str(tool), "/NOCONFIG", "/INPUTCHARSET", "UTF8", "/WX", str(wrapper)], check=True, cwd=INSTALLER)
    tools = {"NSIS": config["nsisVersion"], "Python": sys.version.split()[0]}
    for package in ("PyInstaller", "pyinstaller-hooks-contrib", "PySide6", "torch", "torchvision", "ultralytics", "opencv-python-headless"):
        try:
            tools[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            tools[package] = "not available"
    deliverables = [{"path": path.name, "size": path.stat().st_size, "sha256": digest(path)} for path in (setup, archive)]
    release = {key: config[key] for key in ("productId", "productName", "architecture", "installScope", "channel", "signingPolicy", "updateMode", "distribution")}
    release.update({"version": version, "fileVersion": version + ".0", "artifacts": deliverables, "tools": tools,
                    "signingStatus": "NotSigned", "verification": {"status": "build-only", "report": "verification.json"}})
    release["builtAtUtc"] = datetime.now(timezone.utc).isoformat()
    try:
        release["source"] = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "workingTreeModified": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        release["source"] = {"commit": None, "workingTreeModified": None}
    write_json(output / "release-manifest.json", release)
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{f['sha256']}  {f['path']}" for f in deliverables) + "\n", encoding="ascii")
    (output / "INSTALL.txt").write_text(
        f"SOP {version} - Windows x64 / current user / local test delivery\n\n"
        f"Keep {setup.name} and {archive.name} in the same folder.\n"
        "Run Setup to choose the install folder and shortcuts. No Python installation is needed.\n"
        "Models are not included. Select your model in the application.\n"
        "Settings and records are stored under %LOCALAPPDATA%\\SOP.WorkInstruction and preserved on uninstall.\n"
        "This local test package is not Authenticode signed and has not been publicly deployed.\n"
        "Windows application-control policy can block unsigned software. Signing is not a universal allow rule.\n"
        "Read INSTALLATION.md for silent options, licensing and validation limits.\n"
        "Read verification.json when present for observed results, not just build success.\n", encoding="utf-8")
    (output / "INSTALLATION.md").write_text((INSTALLER / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    (output / "check_windows_trust.ps1").write_bytes((ROOT / "tools/check_windows_trust.ps1").read_bytes())
    print(f"SETUP={setup}\nPAYLOAD={archive}\nMANIFEST={output / 'release-manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=INSTALLER / ".build/payload/SOP")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="Explicitly rebuild an unpublished local delivery")
    parser.add_argument("--test-id", help="Dedicated installer smoke identity; never changes production identity")
    parser.add_argument("--test-name")
    args = parser.parse_args()
    config = json.loads((INSTALLER / "release-config.json").read_text(encoding="utf-8"))
    if config["signingPolicy"] == "required":
        raise ValueError("Required signing needs an authorized signing pipeline before building a delivery")
    if args.test_id:
        if not re.fullmatch(r"SOP\.WorkInstruction\.Smoke\.[A-Za-z0-9_-]+", args.test_id):
            raise ValueError("A test identity must use the isolated SOP.WorkInstruction.Smoke.* namespace")
        config["productId"] = args.test_id
        config["productName"] = args.test_name or args.test_id
    elif args.test_name:
        raise ValueError("A test name requires an isolated test identity")
    version = app_version()
    args.output = args.output or INSTALLER / "releases" / config["channel"] / version / "windows-x64"
    prepare(args, config, version)


if __name__ == "__main__":
    main()
