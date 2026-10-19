"""Collect installed runtime distributions' original notices, without inventing licenses.

This is a conservative runtime dependency inventory, not a legal clearance or an
assertion that every installed module is reachable in the frozen application.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

# PyInstaller itself stays in the inventory: its bootloader ships in SOP.exe.
BUILD_ONLY = {"pip", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes"}


def collect_notices(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        if name.lower().replace("_", "-") in BUILD_ONLY:
            continue
        license_files = []
        target = destination / "licenses" / name
        for item in distribution.files or ():
            parts = tuple(p.lower() for p in item.parts)
            basename = item.name.lower()
            # Include original third-party notices and bundled vendor licenses.
            if not (basename.startswith(("license", "copying", "copyright", "notice"))
                    or "licenses" in parts or "license" in parts):
                continue
            source = Path(distribution.locate_file(item))
            if not source.is_file() or source.suffix.lower() in {".pyc", ".pyd", ".dll"}:
                continue
            # RECORD can contain ../ entries; never allow them to escape staging.
            safe = Path(*[p for p in item.parts if p not in ("..", ".")])
            output = target / safe
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            license_files.append(output.relative_to(destination).as_posix())
        if name.lower() == "nvidia-ml-py" and not license_files:
            # This distribution embeds its BSD notice in the module's comment
            # header rather than installing a separate license document.
            module = Path(distribution.locate_file("pynvml.py"))
            if module.is_file():
                header = []
                for line in module.read_text(encoding="utf-8").splitlines(keepends=True):
                    if line.startswith("#"):
                        header.append(line)
                    else:
                        break
                if header:
                    output = target / "pynvml-source-license-header.txt"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text("".join(header), encoding="utf-8")
                    license_files.append(output.relative_to(destination).as_posix())
        license_text = distribution.metadata.get("License-Expression") or distribution.metadata.get("License") or "Not declared in installed package metadata"
        records.append({
            "name": name, "version": distribution.version,
            "declaredLicense": license_text,
            "licenseFiles": license_files,
            "reviewRequired": not bool(license_files) or name.lower().startswith("ultralytics"),
        })

    # CPython's license is outside distributions in a Windows virtual environment.
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    python_files = []
    if python_license.is_file():
        python_target = destination / "licenses" / "CPython" / "LICENSE.txt"
        python_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(python_license, python_target)
        python_files.append(python_target.relative_to(destination).as_posix())
    # These canonical GNU texts complement Qt wheels that ship only the Qt
    # commercial-reference file despite declaring LGPL/GPL alternatives.
    original_texts = Path(__file__).resolve().parents[1] / "installer" / "licenses"
    if original_texts.is_dir():
        shutil.copytree(original_texts, destination / "licenses" / "GNU", dirs_exist_ok=True)
    inventory = {
        "scope": "Conservative inventory of installed runtime distributions used for this payload build; not legal clearance.",
        "python": {"version": sys.version, "licenseFiles": python_files},
        "reviewRequired": ["Ultralytics packages declare AGPL licensing. Proprietary/commercial redistribution terms have not been established for this project.",
                           "Bundled Qt/PySide6, CUDA/vendor libraries and other dependencies retain their own license requirements; review original license texts before public distribution.",
                           "Project/application license terms have not been supplied."],
        "distributions": records,
    }
    (destination / "dependency-inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["THIRD-PARTY NOTICES — SOP 工序監控", "", inventory["scope"], "", "REVIEW BEFORE PUBLIC DISTRIBUTION:"]
    lines.extend("- " + item for item in inventory["reviewRequired"])
    lines.extend(["", "Original package license files are under licenses/.", "CPython: " + sys.version, ""])
    for item in records:
        declaration = item["declaredLicense"].strip().splitlines()[0]
        lines.append(f"{item['name']} {item['version']}: {declaration}")
        lines.extend("  " + path for path in item["licenseFiles"])
        if not item["licenseFiles"]:
            lines.append("  No separate license file found in installed package; manual review required.")
    (destination / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    collect_notices(parser.parse_args().destination)
