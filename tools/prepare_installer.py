"""Prepare the offline NSIS payload and pinned local compiler; no publishing."""
from __future__ import annotations

import argparse
import ast
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sop_app.release_version import parse_release_version, windows_file_version
from tools.windows_signing import EXECUTABLE_SUFFIXES, WindowsSigner, prepare_components, read_authorizations

INSTALLER = ROOT / "installer"
NSIS_RUNTIME_PLUGINS = ("System.dll", "nsDialogs.dll")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def app_version():
    tree = ast.parse((ROOT / "sop_app/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            version = ast.literal_eval(node.value)
            parse_release_version(version)
            return version
    raise ValueError("Application version is missing")


def compiler(config, *, unpack_to=None):
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
    unpack_to = unpack_to if unpack_to is not None else tool_root
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
                raise ValueError("Unsafe compiler archive")
        bundle.extractall(unpack_to)
    return unpack_to / f"nsis-{version}" / "makensis.exe"


def nsis_string(value):
    return str(value).replace("$", "$$").replace('"', '$\\"')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _preserve_previous_evidence(output):
    reports = ("verification.json", "real-payload-verification.json", "windows-trust-report.json")
    if any((output / name).exists() for name in reports):
        raise FileExistsError("Choose a new output directory. Existing delivery verification must stay with its original files.")


def prepare(args, config, version):
    windows_file_version(version)
    _preserve_previous_evidence(args.output.resolve())
    if config.get("signingPolicy") != "required" and not getattr(args, "sign", False):
        return _prepare(args, config, version)

    signer = WindowsSigner.configured(getattr(args, "sign_tool", None),
                                      getattr(args, "signer_thumbprint", None),
                                      getattr(args, "timestamp_url", None))
    approved = read_authorizations(getattr(args, "approved_unsigned", None))
    output = args.output.resolve()
    setup = output / f"SOP-{version}-windows-x64-Setup.exe"
    if setup.exists() and not args.rebuild:
        raise FileExistsError(f"Existing delivery is preserved: {setup}. Choose another output or use --rebuild.")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the existing delivery and the original PyInstaller payload intact if signing or compiling fails.
    with TemporaryDirectory(prefix=".signing-", dir=output) as temporary:
        scratch = Path(temporary)
        staged = copy(args)
        staged.output = scratch / "delivery"
        _prepare(staged, config, version, signer=signer, approved=approved, scratch=scratch)
        for path in sorted(staged.output.iterdir(), key=lambda item: item.name == "release-manifest.json"):
            path.replace(output / path.name)
    print(f"SETUP={setup}\nPAYLOAD={output / f'SOP-{version}-windows-x64.payload.zip'}"
          f"\nMANIFEST={output / 'release-manifest.json'}", flush=True)


def _prepare(args, config, version, *, signer=None, approved=None, scratch=None):
    file_version = windows_file_version(version)
    payload = args.payload.resolve()
    if not (payload / config["entryExe"]).is_file():
        raise ValueError(f"Frozen entry point missing: {payload / config['entryExe']}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = getattr(args, "runtime", "torch")
    suffix = "-onnx" if runtime == "onnx" else ""
    runtime_marker = payload / "_internal" / "sop-runtime.txt"
    if runtime_marker.is_file():
        if runtime_marker.read_text(encoding="utf-8").strip() != runtime:
            raise ValueError(f"Payload runtime does not match --runtime {runtime}")
    elif runtime == "onnx":
        raise ValueError("Missing runtime identity for the ONNX payload. Run tools/build_payload.py --runtime onnx.")
    generated = scratch / "nsis-work" if signer else INSTALLER / ".build" / ("nsis-" + config["productId"] + suffix)
    generated.mkdir(parents=True, exist_ok=True)
    stem = f"SOP-{version}-windows-x64"
    setup = output / f"{stem}-Setup.exe"
    if setup.exists() and not args.rebuild:
        raise FileExistsError(f"Existing delivery is preserved: {setup}. Choose another output or use --rebuild for a local test.")
    archive = output / f"{stem}.payload.zip"
    inventory_path = generated / "payload-manifest.json"
    payload_files = {}
    for path in sorted(payload.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Links cannot be packaged: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(payload).as_posix()
        if relative in ("Uninstall.exe", ".sop-install.json"):
            raise ValueError(f"Reserved installation file: {relative}")
        payload_files[relative] = path
    scripts = {"deploy.ps1": INSTALLER / "deploy.ps1", "shortcuts.ps1": INSTALLER / "shortcuts.ps1",
               "check_windows_trust.ps1": ROOT / "tools/check_windows_trust.ps1"}
    signatures = {}
    compiler_tool = None
    compile_env = None
    compiler_temp = None
    uninstaller_report = None
    if signer:
        # Start from a private extraction of the checksum-pinned compiler so
        # stale cache files cannot enter this signing build.
        compiler_tool = compiler(config, unpack_to=scratch / "nsis-compiler")
        plugin_dir = compiler_tool.parent / "Plugins/x86-unicode"
        entry_component = f"payload/{config['entryExe']}"
        components = {f"payload/{name}": path for name, path in payload_files.items()
                      if path.suffix.lower() in EXECUTABLE_SUFFIXES}
        components.update({f"installer/{name}": path for name, path in scripts.items()})
        components.update({f"nsis/{name}": plugin_dir / name for name in NSIS_RUNTIME_PLUGINS})
        owned = {entry_component, *(f"installer/{name}" for name in scripts)}
        prepared, signatures = prepare_components(signer, components, owned, approved, scratch / "components")
        payload_files.update({name.removeprefix("payload/"): path for name, path in prepared.items()
                              if name.startswith("payload/")})
        scripts = {name: prepared[f"installer/{name}"] for name in scripts}
        for name in NSIS_RUNTIME_PLUGINS:
            signed_plugin = prepared[f"nsis/{name}"]
            if signed_plugin != plugin_dir / name:
                shutil.copyfile(signed_plugin, plugin_dir / name)
        uninstaller_report = scratch / "signed-uninstaller.json"
        compiler_temp = scratch / "nsis-temp"
        compiler_temp.mkdir()
        compile_env = {**os.environ, "TEMP": str(compiler_temp), "TMP": str(compiler_temp),
                       **signer.uninstaller_environment(uninstaller_report)}

    print("Hashing clean frozen payload...", flush=True)
    files = []
    for name, path in payload_files.items():
        file_hash = digest(path)
        if signer and f"payload/{name}" in signatures and file_hash != signatures[f"payload/{name}"]["sha256"]:
            raise ValueError(f"Payload bytes changed after signature verification: {name}")
        files.append({"path": name, "size": path.stat().st_size, "sha256": file_hash})
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
                bundle.write(payload_files[record["path"]], record["path"])
        partial.replace(archive)
    archive_hash = digest(archive)
    write_json(inventory_path, manifest)
    write_json(cache_path, {"archive": str(archive), "sha256": archive_hash})
    icon = INSTALLER / ".build" / ("generated" + suffix) / "SOP.ico"
    if not icon.exists():
        raise ValueError("Product icon missing; first run tools/build_payload.py")
    defines = {
        "PRODUCT_ID": config["productId"], "PRODUCT_NAME": config["productName"],
        "PRODUCT_VERSION": version, "FILE_VERSION": file_version, "PUBLISHER": config["publisher"],
        "ENTRY_EXE": config["entryExe"], "OUTPUT_EXE": setup, "PAYLOAD_NAME": archive.name,
        "PAYLOAD_HASH": archive_hash, "PAYLOAD_MANIFEST": inventory_path,
        "ICON_FILE": icon, "DEPLOY_SCRIPT": scripts["deploy.ps1"],
        "SHORTCUT_SCRIPT": scripts["shortcuts.ps1"],
    }
    header = generated / "Product.nsh"
    header.write_text("\n".join(f'!define {key} "{nsis_string(value)}"' for key, value in defines.items())
                      + f"\n!define ESTIMATED_KB {(sum(f['size'] for f in files) + 1023) // 1024}\n", encoding="utf-8")
    wrapper = generated / "Compile.nsi"
    preamble = _uninstaller_finalize_command(compiler_temp) if signer else ""
    wrapper.write_text(preamble + f'!include "{nsis_string(header)}"\n!include "{nsis_string(INSTALLER / "Setup.nsi")}"\n',
                       encoding="utf-8")
    tool = compiler_tool or compiler(config)
    print("Compiling NSIS Modern UI 2...", flush=True)
    subprocess.run([str(tool), "/NOCONFIG", "/INPUTCHARSET", "UTF8", "/WX", str(wrapper)], check=True, cwd=INSTALLER,
                   env=compile_env)
    if signer:
        if not uninstaller_report.is_file():
            raise ValueError("NSIS did not produce a verified uninstaller signature report")
        embedded = json.loads(uninstaller_report.read_text(encoding="utf-8"))
        if (not isinstance(embedded, dict) or embedded.get("status") != "Valid"
                or embedded.get("thumbprint", "").upper() != signer.thumbprint
                or embedded.get("timestamped") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", embedded.get("sha256", ""))):
            raise ValueError("Invalid uninstaller signature verification report")
        signatures["installer/Uninstall.exe"] = embedded
        signatures[f"installer/{setup.name}"] = signer.sign(setup)
    tools = {"NSIS": config["nsisVersion"], "Python": sys.version.split()[0]}
    for package in ("PyInstaller", "pyinstaller-hooks-contrib", "PySide6", "torch", "torchvision", "ultralytics",
                    "opencv-python-headless", "onnxruntime"):
        try:
            tools[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            tools[package] = "not available"
    trust_script = output / "check_windows_trust.ps1"
    trust_script.write_bytes(scripts["check_windows_trust.ps1"].read_bytes())
    counted = (setup, archive, trust_script) if signer else (setup, archive)
    deliverables = [{"path": path.name, "size": path.stat().st_size, "sha256": digest(path)} for path in counted]
    release = {key: config[key] for key in ("productId", "productName", "architecture", "installScope", "channel", "signingPolicy", "updateMode", "distribution")}
    release.update({"version": version, "fileVersion": file_version, "artifacts": deliverables, "tools": tools,
                    "runtime": runtime, "signingStatus": "Valid" if signer else "NotSigned",
                    "verification": {"status": "build-only", "report": "verification.json"}})
    if signer:
        release["authenticode"] = {"components": signatures, "scope": (
            "Verified embedded Authenticode on the build host for the listed files and the generated NSIS uninstaller. "
            "Target-machine execution and application-control policy still require validation.")}
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
    signing_note = ("Listed Authenticode signatures were verified on the build host." if signer else
                    "This local test package is not Authenticode signed and has not been publicly deployed.")
    (output / "INSTALL.txt").write_text(
        f"SOP {version} - Windows x64 / current user / local test delivery\n\n"
        f"Keep {setup.name} and {archive.name} in the same folder.\n"
        "Run Setup to choose the install folder and shortcuts. No Python installation is needed.\n"
        "Models are not included. Select your model in the application.\n"
        "Settings and records are stored under %LOCALAPPDATA%\\SOP.WorkInstruction and preserved on uninstall.\n"
        f"{signing_note}\n"
        "Windows application-control policy can block unsigned software. Signing is not a universal allow rule.\n"
        "Read INSTALLATION.md for silent options, licensing and validation limits.\n"
        "Read verification.json when present for observed results, not just build success.\n", encoding="utf-8")
    (output / "INSTALLATION.md").write_text((INSTALLER / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    if not signer:
        print(f"SETUP={setup}\nPAYLOAD={archive}\nMANIFEST={output / 'release-manifest.json'}", flush=True)


def _uninstaller_finalize_command(compiler_temp):
    interpreter = str(Path(sys.executable).resolve())
    helper = str(Path(__file__).with_name("windows_signing.py").resolve())
    # NSIS executes this build-only hook via the Windows command interpreter.
    if any(character in interpreter + helper + str(compiler_temp) for character in "'\"$%!\r\n"):
        raise ValueError("The signing build requires a Python and source path without shell expansion characters")
    return f'!uninstfinalize \'"{interpreter}" "{helper}" "%1"\' = 0\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="Explicitly rebuild an unpublished local delivery")
    parser.add_argument("--test-id", help="Dedicated installer smoke identity; never changes production identity")
    parser.add_argument("--test-name")
    parser.add_argument("--sign", action="store_true", help="Require verified signing before creating the delivery")
    parser.add_argument("--sign-tool", type=Path, help="Path to an installed Windows SDK signtool.exe")
    parser.add_argument("--signer-thumbprint", help="The authorized certificate thumbprint in CurrentUser\\My")
    parser.add_argument("--timestamp-url", help="The authorized RFC 3161 time-stamping service")
    parser.add_argument("--approved-unsigned", type=Path, help="JSON of exact SHA-256 authorizations for vendor files")
    args = parser.parse_args()
    config = json.loads((INSTALLER / "release-config.json").read_text(encoding="utf-8"))
    signed = args.sign or config["signingPolicy"] == "required"
    if not signed and any((args.sign_tool, args.signer_thumbprint, args.timestamp_url, args.approved_unsigned)):
        parser.error("Signing parameters require --sign or signingPolicy required")
    if args.test_id:
        if not re.fullmatch(r"SOP\.WorkInstruction\.Smoke\.[A-Za-z0-9_-]+", args.test_id):
            raise ValueError("A test identity must use the isolated SOP.WorkInstruction.Smoke.* namespace")
        config["productId"] = args.test_id
        config["productName"] = args.test_name or args.test_id
    elif args.test_name:
        raise ValueError("A test name requires an isolated test identity")
    version = app_version()
    suffix = "-onnx" if args.runtime == "onnx" else ""
    args.payload = args.payload or INSTALLER / ".build" / ("payload" + suffix) / "SOP"
    delivery_name = "windows-x64" + suffix + ("-signed" if signed else "")
    args.output = args.output or INSTALLER / "releases" / config["channel"] / version / delivery_name
    prepare(args, config, version)


if __name__ == "__main__":
    main()
