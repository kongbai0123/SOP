param(
    [switch]$SkipPayload,
    [switch]$InstallBuildTools,
    [switch]$Rebuild,
    [string]$OutputDirectory,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $taskRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Build Python was not found. Pass -Python or prepare the project .venv.' }
if (-not $SkipPayload) {
    $payloadArgs = @((Join-Path $PSScriptRoot 'build_payload.py'))
    if ($InstallBuildTools) { $payloadArgs += '--install-build-tools' }
    & $Python @payloadArgs
    if ($LASTEXITCODE -ne 0) { throw 'Frozen payload build failed.' }
}
$prepareArgs = @((Join-Path $PSScriptRoot 'prepare_installer.py'))
if ($Rebuild) { $prepareArgs += '--rebuild' }
if ($OutputDirectory) { $prepareArgs += @('--output', [IO.Path]::GetFullPath($OutputDirectory)) }
& $Python @prepareArgs
if ($LASTEXITCODE -ne 0) { throw 'NSIS preparation or compilation failed.' }
