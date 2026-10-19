param(
    [Parameter(Mandatory=$true)][ValidateSet('Install','Uninstall')][string]$Action,
    [Parameter(Mandatory=$true)][string]$InstallDir,
    [Parameter(Mandatory=$true)][string]$ProductName,
    [ValidateSet('0','1')][string]$Desktop = '1',
    [ValidateSet('0','1')][string]$StartMenu = '1',
    [Parameter(Mandatory=$true)][string]$LogPath
)
$ErrorActionPreference = 'Stop'
try {
    if ($ProductName.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) { throw 'Invalid shortcut name.' }
    $target = [IO.Path]::GetFullPath((Join-Path $InstallDir 'SOP.exe'))
    $shell = New-Object -ComObject WScript.Shell
    $links = @(
        @{Path=(Join-Path ([Environment]::GetFolderPath('Desktop')) ($ProductName + '.lnk')); Enabled=$Desktop},
        @{Path=(Join-Path ([Environment]::GetFolderPath('Programs')) ($ProductName + '.lnk')); Enabled=$StartMenu}
    )
    foreach ($item in $links) {
        $owned = $false
        if (Test-Path -LiteralPath $item.Path) {
            $existing = $shell.CreateShortcut($item.Path)
            $owned = [string]::Equals($existing.TargetPath, $target, [StringComparison]::OrdinalIgnoreCase)
            if (-not $owned) {
                if ($Action -eq 'Install' -and $item.Enabled -eq '1') { throw ('Shortcut belongs to another target: ' + $item.Path) }
                continue
            }
        }
        if ($Action -eq 'Install' -and $item.Enabled -eq '1') {
            $link = $shell.CreateShortcut($item.Path)
            $link.TargetPath = $target
            $link.WorkingDirectory = $InstallDir
            $link.Description = $ProductName
            $link.IconLocation = $target + ',0'
            $link.Save()
            if (-not (Test-Path -LiteralPath $item.Path)) { throw ('Shortcut was not written: ' + $item.Path) }
        } elseif ($owned) {
            Remove-Item -LiteralPath $item.Path -Force -ErrorAction Stop
        }
    }
    exit 0
} catch {
    ('{0:o} shortcuts {1}: {2}' -f [DateTime]::Now, $Action, $_.Exception.Message) | Out-File -LiteralPath $LogPath -Append -Encoding utf8
    exit 1
}
