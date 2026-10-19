r"""Build the self-contained Windows x64 payload using the current project venv.

Usage: .venv\Scripts\python.exe tools\build_payload.py [--install-build-tools]
Only the optional build-tool bootstrap accesses the network, using verified
official PyPI wheel digests. Runtime packages are never upgraded by this script.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import urllib.request

from collect_notices import collect_notices

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sop_app.release_version import parse_release_version, windows_file_version, windows_version_tuple

BUILD = ROOT / "installer" / ".build"
GENERATED = BUILD / "generated"
BUILD_TOOLS = {"pyinstaller": "6.22.3", "pyinstaller-hooks-contrib": "2026.8", "altgraph": "0.17.5",
               "pefile": "2024.8.26", "pywin32-ctypes": "0.2.3", "packaging": "26.3", "setuptools": "84.0.0"}


def read_version() -> str:
    module = ast.parse((ROOT / "sop_app" / "__init__.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            value = ast.literal_eval(node.value)
            try:
                parse_release_version(value)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            return value
    raise SystemExit("Expected a numeric three- or four-part sop_app.__version__ for PE resources")


def bootstrap_tools() -> None:
    target = ROOT / "installer" / ".tools" / "python-build"
    target.mkdir(parents=True, exist_ok=True)
    downloads = []
    wheels = []
    for name, version in BUILD_TOOLS.items():
        try:
            if metadata.version(name) == version:
                continue
        except metadata.PackageNotFoundError:
            pass
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url, timeout=60) as response:
            release = json.load(response)
        candidates = [f for f in release["urls"] if f["filename"].endswith(".whl")
                      and ("win_amd64" in f["filename"] if name == "pyinstaller" else "any.whl" in f["filename"])]
        if len(candidates) != 1:
            raise SystemExit(f"Expected one compatible official wheel for {name}=={version}")
        record = candidates[0]
        with urllib.request.urlopen(record["url"], timeout=60) as response:
            blob = response.read()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != record["digests"]["sha256"]:
            raise SystemExit(f"Official PyPI SHA256 mismatch: {name}")
        output = target / record["filename"]
        output.write_bytes(blob)
        wheels.append(str(output))
        downloads.append({"name": name, "version": version, "url": record["url"], "sha256": digest})
    if downloads:
        audit = target / "verified-downloads.json"
        prior = json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else []
        audit.write_text(json.dumps(prior + downloads, indent=2), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheels], check=True)


def generate_resources(version: str) -> None:
    version_parts = windows_version_tuple(version)
    file_version = windows_file_version(version)
    from PIL import Image, ImageDraw

    GENERATED.mkdir(parents=True, exist_ok=True)
    sample = json.loads((ROOT / "sops" / "example_sop.json").read_text(encoding="utf-8"))
    # The development example references a model exported outside this project.
    # Preserve the source example, but never ship that machine-specific path.
    sample["model_path"] = ""
    sample["model_version_id"] = ""
    (GENERATED / "example_sop.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Original geometry, no third-party branding: blue work panel and check mark.
    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, 246, 246), radius=24, fill="#154C79")
    for top in (58, 108, 158):
        draw.rectangle((44, top, 66, top + 22), fill="#FFFFFF")
        draw.rectangle((86, top + 6, 202, top + 16), fill="#C6D9E8")
    draw.line((126, 196, 150, 218, 210, 157), fill="#62E0B5", width=17, joint="curve")
    image.save(GENERATED / "SOP.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    info = f'''VSVersionInfo(
  ffi=FixedFileInfo(filevers={version_parts!r}, prodvers={version_parts!r},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040404b0', [
    StringStruct('FileDescription', 'SOP 工序監控'),
    StringStruct('FileVersion', '{file_version}'),
    StringStruct('InternalName', 'SOP'),
    StringStruct('OriginalFilename', 'SOP.exe'),
    StringStruct('ProductName', 'SOP 工序監控'),
    StringStruct('ProductVersion', '{version}')
  ])]), VarFileInfo([VarStruct('Translation', [1028, 1200])])])
'''
    (GENERATED / "version-info.txt").write_text(info, encoding="utf-8")
    collect_notices(GENERATED / "notices")


def verify_frozen_contents(payload: Path) -> dict:
    import pefile
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(payload / "SOP.exe"))
    module = archive.open_embedded_archive("PYZ.pyz").extract("sop_app.launcher")
    entry = next(value for value in module.co_consts if getattr(value, "co_name", "") == "start")
    def literal_strings(values):
        for value in values:
            if isinstance(value, str):
                yield value
            elif isinstance(value, (tuple, list)):
                yield from literal_strings(value)
    entry_strings = set(literal_strings(entry.co_consts))
    for marker in ("--installer-smoke-test", "--installer-runtime-test"):
        if marker not in entry_strings:
            raise RuntimeError(f"Frozen launcher contains stale bytecode; missing {marker}")
    diagnose = next(value for value in module.co_consts if getattr(value, "co_name", "") == "_diagnose")
    if "SOP_LAUNCH_DIAGNOSTICS" not in diagnose.co_consts:
        raise RuntimeError("Frozen launch diagnostics code is stale")
    internal = payload / "_internal"
    qt = pefile.PE(str(internal / "PySide6" / "Qt6Core.dll"), max_symbol_exports=100000)
    qt_exports = {item.name for item in qt.DIRECTORY_ENTRY_EXPORT.symbols if item.name}
    extension = pefile.PE(str(internal / "PySide6" / "QtCore.pyd"), max_symbol_exports=100000)
    imported = next(item for item in extension.DIRECTORY_ENTRY_IMPORT if item.dll.lower() == b"qt6core.dll")
    missing = [item.name for item in imported.imports if item.name and item.name not in qt_exports]
    if missing:
        raise RuntimeError(f"QtCore.pyd / Qt6Core.dll export mismatch: {missing[:5]}")
    icu_imports = next((item for item in qt.DIRECTORY_ENTRY_IMPORT if item.dll.lower() == b"icuuc.dll"), None)
    icu_source = "not required by this Qt build"
    if icu_imports:
        bundled = list(internal.rglob("icuuc.dll"))
        if len(bundled) > 1:
            raise RuntimeError("Multiple incompatible ICU candidates are bundled")
        icu_path = bundled[0] if bundled else Path(os.environ["SystemRoot"]) / "System32" / "icuuc.dll"
        icu = pefile.PE(str(icu_path), max_symbol_exports=100000)
        exports = {item.name for item in icu.DIRECTORY_ENTRY_EXPORT.symbols if item.name}
        missing = [item.name for item in icu_imports.imports if item.name and item.name not in exports]
        if missing:
            raise RuntimeError(f"Qt resolved an incompatible ICU ({icu_path}): {missing[:5]}")
        icu_source = str(icu_path) if bundled else "Windows System32 ICU (operating-system component)"
    return {"launcherBytecodeCurrent": True, "qtCoreExportsCompatible": True, "icuSource": icu_source}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-build-tools", action="store_true")
    parser.add_argument("--resources-only", action="store_true")
    options = parser.parse_args()
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise SystemExit("Build requires Windows with a 64-bit Python interpreter")
    if options.install_build_tools:
        bootstrap_tools()
    for name, version in BUILD_TOOLS.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"Missing {name}; rerun with --install-build-tools")
        if installed != version:
            raise SystemExit(f"Expected {name}=={version}; found {installed}. Rerun with --install-build-tools.")
    version = read_version()
    generate_resources(version)
    environment = {"productId": "SOP.WorkInstruction", "version": version, "architecture": "x64",
                   "python": sys.version, "platform": platform.platform(),
                   "tools": BUILD_TOOLS,
                   "runtimePackages": {d.metadata["Name"]: d.version for d in metadata.distributions()}}
    (GENERATED / "build-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    if options.resources_only:
        return 0
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
               "--distpath", str(BUILD / "payload"), "--workpath", str(BUILD / "pyinstaller"),
               str(ROOT / "installer" / "SOP.spec")]
    print("Building SOP", version, flush=True)
    environment_variables = os.environ.copy()
    # Ultralytics imports may check optional training/tracker packages. Analysis
    # must never run its AutoUpdate/pip path or touch the developer's settings.
    environment_variables["YOLO_AUTOINSTALL"] = "false"
    environment_variables["YOLO_CONFIG_DIR"] = str(BUILD / "ultralytics-config")
    Path(environment_variables["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    # Native analysis must not resolve Qt's Windows ICU import to an unrelated
    # Poppler/Conda ICU DLL inherited from the agent host's PATH.
    windows_root = Path(os.environ["SystemRoot"])
    environment_variables["PATH"] = os.pathsep.join(str(item) for item in (
        Path(sys.executable).parent, Path(sys.base_prefix), Path(sys.base_prefix) / "DLLs",
        windows_root / "System32", windows_root,
    ))
    source_hash = hashlib.sha256()
    for source in sorted([*ROOT.glob("sop_app/**/*.py"), ROOT / "main.py", ROOT / "installer" / "frozen_entry.py"]):
        source_hash.update(source.relative_to(ROOT).as_posix().encode("utf-8"))
        source_hash.update(source.read_bytes())
    source_digest = source_hash.hexdigest()
    environment["sourceDigest"] = source_digest
    # Separate bytecode namespaces avoid reading stale same-timestamp .pyc files
    # while sources are edited during release preparation.
    environment_variables["PYTHONPYCACHEPREFIX"] = str(BUILD / "python-cache" / source_digest[:16])
    environment_variables["PYINSTALLER_CONFIG_DIR"] = str(BUILD / "pyinstaller-cache")
    result = subprocess.run(command, cwd=ROOT, env=environment_variables)
    if result.returncode:
        return result.returncode
    payload = BUILD / "payload" / "SOP"
    environment["staticVerification"] = verify_frozen_contents(payload)
    # Include bootloader notices and inventory the final build environment.
    collect_notices(payload / "_internal" / "notices")
    files = [p for p in payload.rglob("*") if p.is_file()]
    environment["payloadFiles"] = len(files)
    environment["payloadBytes"] = sum(p.stat().st_size for p in files)
    environment["payloadPath"] = "installer/.build/payload/SOP"
    (GENERATED / "build-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: environment[k] for k in ("payloadPath", "payloadFiles", "payloadBytes")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
