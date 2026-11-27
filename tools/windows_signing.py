"""Authenticode signing for release builders with an existing authorized certificate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit


EXECUTABLE_SUFFIXES = frozenset({".exe", ".dll", ".pyd", ".ps1", ".sys", ".ocx"})
UNINSTALLER_SETTINGS = "SOP_BUILD_UNINSTALLER_SIGNING"
UNINSTALLER_REPORT = "SOP_BUILD_UNINSTALLER_SIGNATURE_REPORT"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _signatures(paths: list[Path]) -> list[dict]:
    if not paths:
        return []
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Import-Module -Name (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1')
$paths = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd())
$inventory = @(foreach ($path in $paths) {
  $signature = Get-AuthenticodeSignature -LiteralPath $path
  [ordered]@{
    status = [string]$signature.Status
    signer = [string]$signature.SignerCertificate.Subject
    thumbprint = [string]$signature.SignerCertificate.Thumbprint
    selfIssued = ($null -ne $signature.SignerCertificate -and
                  $signature.SignerCertificate.Subject -eq $signature.SignerCertificate.Issuer)
    timestamped = ($null -ne $signature.TimeStamperCertificate)
  }
})
ConvertTo-Json -InputObject $inventory -Compress
"""
    completed = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
                               input=json.dumps([str(path) for path in paths]), capture_output=True,
                               encoding="utf-8", check=True, timeout=300)
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list) or len(rows) != len(paths) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Windows did not return a complete Authenticode inventory")
    return rows


@dataclass(frozen=True)
class WindowsSigner:
    tool: Path
    thumbprint: str
    timestamp_url: str

    @classmethod
    def configured(cls, tool: Path | None, thumbprint: str | None, timestamp_url: str | None):
        if tool is None or not thumbprint or not timestamp_url:
            raise ValueError("Signing requires --sign-tool, --signer-thumbprint and --timestamp-url")
        if os.name != "nt":
            raise ValueError("Authenticode release signing requires Windows")
        tool = tool.resolve(strict=True)
        if tool.name.lower() != "signtool.exe" or not tool.is_file():
            raise ValueError("--sign-tool must name an installed Windows SDK signtool.exe")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", thumbprint):
            raise ValueError("--signer-thumbprint must identify an existing certificate in the current user's My store")
        url = urlsplit(timestamp_url)
        if url.scheme not in ("https", "http") or not url.hostname or url.username is not None or url.password is not None:
            raise ValueError("--timestamp-url must be an RFC 3161 HTTP(S) service without credentials")
        signature = _signatures([tool])[0]
        subject = signature.get("signer", "")
        if signature.get("status") != "Valid" or not re.search(
            r"(?:^|,\s*)CN=Microsoft Corporation(?:,|$)", subject, re.IGNORECASE
        ):
            raise ValueError("The configured SignTool must have a valid Microsoft Authenticode signature")
        return cls(tool, thumbprint.upper(), timestamp_url)

    def sign(self, path: Path) -> dict:
        subprocess.run([str(self.tool), "sign", "/q", "/s", "My", "/sha1", self.thumbprint,
                        "/fd", "SHA256", "/tr", self.timestamp_url, "/td", "SHA256", str(path)],
                       check=True, timeout=180)
        return self.verify_owned(path)

    def verify_owned(self, path: Path, signature: dict | None = None) -> dict:
        signature = _signatures([path])[0] if signature is None else signature
        if (signature.get("status") != "Valid" or signature.get("thumbprint", "").upper() != self.thumbprint
                or signature.get("timestamped") is not True or signature.get("selfIssued") is True):
            raise ValueError(f"Missing a valid timestamped signature from the selected publisher: {path.name}")
        subprocess.run([str(self.tool), "verify", "/q", "/pa", "/all", "/tw", str(path)], check=True, timeout=120)
        return {**signature, "sha256": file_sha256(path)}

    def verify_vendor(self, paths: list[Path]) -> None:
        # Check embedded Authenticode, without relying on the local catalog store.
        for start in range(0, len(paths), 20):
            subprocess.run([str(self.tool), "verify", "/q", "/pa", "/all",
                            *map(str, paths[start:start + 20])], check=True, timeout=180)

    def uninstaller_environment(self, report_path: Path) -> dict[str, str]:
        return {UNINSTALLER_SETTINGS: json.dumps({"tool": str(self.tool), "thumbprint": self.thumbprint,
                                                 "timestamp_url": self.timestamp_url}),
                UNINSTALLER_REPORT: str(report_path)}


def read_authorizations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in data.values()
    ):
        raise ValueError("--approved-unsigned must be a JSON object of component names and source SHA-256 digests")
    return {name: value.lower() for name, value in data.items()}


def prepare_components(signer: WindowsSigner, components: dict[str, Path], owned: set[str],
                       approved: dict[str, str], scratch: Path) -> tuple[dict[str, Path], dict[str, dict]]:
    """Use only the private copies from signature checks through delivery."""
    staged = {}
    for name, source in components.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or linked executable component: {name}")
        target = scratch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        staged[name] = target
    signatures = dict(zip(staged, _signatures(list(staged.values())), strict=True))
    to_sign, vendor_paths, rejected = [], [], []
    for name, source in staged.items():
        status = signatures[name].get("status")
        if status == "Valid":
            if name not in owned:
                vendor_paths.append(source)
        elif status == "NotSigned" and (name in owned or approved.get(name) == file_sha256(source)):
            to_sign.append(name)
        else:
            rejected.append(f"{name} ({status or 'unknown'})")
    if rejected:
        raise ValueError("Unsigned vendor components require exact SHA-256 authorization; other invalid signatures "
                         "require replacement:\n  " + "\n  ".join(rejected))
    signer.verify_vendor(vendor_paths)

    result = {}
    for name, source in staged.items():
        if name not in to_sign:
            result[name] = (signer.verify_owned(source, signatures[name]) if name in owned
                            else {**signatures[name], "sha256": file_sha256(source)})
            continue
        if name not in owned and file_sha256(source) != approved[name]:
            raise ValueError(f"Authorized source bytes changed while staging: {name}")
        result[name] = signer.sign(source)
    return staged, result


def _sign_uninstaller(path: Path) -> None:
    settings = json.loads(os.environ[UNINSTALLER_SETTINGS])
    signer = WindowsSigner.configured(Path(settings["tool"]), settings["thumbprint"], settings["timestamp_url"])
    parent = Path(os.environ["TEMP"]).resolve()
    if (not path.is_file() or path.is_symlink() or path.resolve().parent != parent
            or path.suffix.lower() not in (".tmp", ".exe")):
        raise ValueError("NSIS did not provide a generated uninstaller EXE")
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("NSIS did not provide a valid uninstaller PE header")
    # NSIS 3.x passes a .tmp file. Use an EXE name for Authenticode and hand the
    # signed bytes back to NSIS before it embeds the uninstaller into Setup.
    with TemporaryDirectory(prefix="sop-uninstaller-", dir=path.parent) as temporary:
        candidate = Path(temporary) / "Uninstall.exe"
        shutil.copyfile(path, candidate)
        report = signer.sign(candidate)
        candidate.replace(path)
    Path(os.environ[UNINSTALLER_REPORT]).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("This entry point is only used by the installer build's uninstaller finalizer.")
    _sign_uninstaller(Path(sys.argv[1]))
