[CmdletBinding()]
param(
    [string]$Action,
    [string]$InstallDir,
    [string]$ProductId,
    [string]$Version,
    [string]$PayloadPath,
    [string]$PayloadSha256,
    [string]$ManifestPath,
    [string]$LogPath,
    [string]$ExeName = 'SOP.exe'
)

# Keep this source ASCII so Windows PowerShell 5.1 reads it consistently.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:MarkerName = '.sop-install.json'
$script:StageRoot = $null
$script:PayloadStream = $null
$script:Archive = $null
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)
$script:CreatedTargetDirectories = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)

function Write-DeployLog([string]$Message) {
    $line = '[{0:u}] {1}' -f [DateTime]::UtcNow, $Message
    if ($LogPath) {
        try {
            [IO.File]::AppendAllText($LogPath, $line + [Environment]::NewLine, $script:Utf8)
        } catch {
            [Console]::Error.WriteLine('Could not write installer log: ' + $_.Exception.Message)
        }
    }
    [Console]::Error.WriteLine($line)
}

function Assert-NoReparse([string]$Path) {
    $cursor = $Path
    while ($cursor) {
        $item = $null
        try {
            $item = Get-Item -LiteralPath $cursor -Force
        } catch {
            if ($_.CategoryInfo.Category -ne [Management.Automation.ErrorCategory]::ObjectNotFound) { throw }
        }
        if ($item) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are not supported in installation paths: $cursor"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if (!$parent -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
}

function Get-SafeRelativePath([object]$Value) {
    if (!($Value -is [string]) -or !$Value -or $Value.Contains('\') -or
        $Value.StartsWith('/') -or $Value.Contains(':')) {
        throw 'Payload paths must be relative paths using forward slashes.'
    }
    foreach ($part in $Value.Split('/')) {
        if (!$part -or $part -eq '.' -or $part -eq '..' -or
            $part.EndsWith('.') -or $part.EndsWith(' ') -or
            $part.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
            $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw "Unsafe payload path: $Value"
        }
    }
    $firstComponent = $Value.Split('/')[0]
    if ($firstComponent -ieq $script:MarkerName -or $firstComponent -ieq 'Uninstall.exe') {
        throw "The payload cannot own installer metadata: $Value"
    }
    return $Value
}

function Get-ContainedPath([string]$Root, [string]$Relative) {
    $safe = Get-SafeRelativePath $Relative
    $path = [IO.Path]::GetFullPath([IO.Path]::Combine($Root, $safe.Replace('/', '\')))
    if (!$path.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes its installation directory: $Relative"
    }
    Assert-NoReparse $path
    return $path
}

function Get-SafeInstallDir([string]$Value) {
    if (!$Value -or $Value -notmatch '^[A-Za-z]:[\\/]' -or $Value.Substring(2).Contains(':')) {
        throw 'Choose a fully qualified local drive path.'
    }
    foreach ($part in $Value.Substring(3).Replace('/', '\').Split('\')) {
        if ($part -eq '.' -or $part -eq '..' -or $part.EndsWith('.') -or $part.EndsWith(' ') -or
            ($part -and ($part.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
             $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)'))) {
            throw 'The installation path contains an unsupported path component.'
        }
    }
    $path = [IO.Path]::GetFullPath($Value).TrimEnd('\')
    $root = [IO.Path]::GetPathRoot($path).TrimEnd('\')
    if ($path -ieq $root) { throw 'Installing directly into a drive root is not supported.' }
    foreach ($protected in @($env:WINDIR, $env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:ProgramW6432)) {
        if ($protected) {
            $protected = [IO.Path]::GetFullPath($protected).TrimEnd('\')
            if ($path -ieq $protected -or $path.StartsWith($protected + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "The installation directory is in a protected system location: $protected"
            }
        }
    }
    foreach ($protected in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:APPDATA)) {
        if ($protected -and $path -ieq [IO.Path]::GetFullPath($protected).TrimEnd('\')) {
            throw 'Choose a dedicated product directory, not a user-data root.'
        }
    }
    Assert-NoReparse $path
    if ((Test-Path -LiteralPath $path) -and !(Test-Path -LiteralPath $path -PathType Container)) {
        throw 'The installation path is an existing file.'
    }
    return $path
}

function Get-NumericVersion([object]$Value) {
    if (!($Value -is [string]) -or $Value -notmatch '^\d+\.\d+\.\d+(?:\.\d+)?$') {
        throw "Unsupported numeric product version: $Value"
    }
    try {
        $parsed = [Version]$Value
        if ($parsed.Revision -lt 0) { return New-Object Version($parsed.Major, $parsed.Minor, $parsed.Build, 0) }
        return $parsed
    } catch { throw "Invalid product version: $Value" }
}

function Read-Ownership([string]$Path, [bool]$IsInstalled) {
    Assert-NoReparse $Path
    $document = [IO.File]::ReadAllText($Path, $script:Utf8) | ConvertFrom-Json
    if ($document.productId -cne $ProductId) { throw 'The product identity does not match.' }
    $null = Get-NumericVersion $document.version
    if ($IsInstalled -and $document.scope -cne 'current-user') {
        throw 'The installation scope does not match current-user.'
    }
    $files = @($document.files)
    if (!$files.Count) { throw 'The ownership manifest is empty.' }
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $map = @{}
    foreach ($file in $files) {
        $relative = Get-SafeRelativePath $file.path
        if (!$seen.Add($relative)) { throw "Duplicate ownership path: $relative" }
        if (!($file.sha256 -is [string]) -or $file.sha256 -notmatch '^[0-9a-fA-F]{64}$') {
            throw "Invalid SHA-256 in manifest: $relative"
        }
        if ($null -eq $file.size -or $file.size -is [string] -or
            $file.size -is [bool] -or [decimal]$file.size -lt 0 -or
            [decimal]$file.size -ne [long]$file.size) {
            throw "Invalid file size in manifest: $relative"
        }
        $map[$relative] = $file
    }
    foreach ($relative in $map.Keys) {
        $cursor = $relative
        while ($cursor.Contains('/')) {
            $cursor = $cursor.Substring(0, $cursor.LastIndexOf('/'))
            if ($map.ContainsKey($cursor)) { throw "A manifest file is also a directory: $cursor" }
        }
    }
    if (!$map.ContainsKey($ExeName)) { throw "The manifest is missing the application entry point: $ExeName" }
    return @{ Document = $document; Files = $files; Map = $map }
}

function Get-ExistingInstallation {
    $marker = Join-Path $script:Target $script:MarkerName
    if (Test-Path -LiteralPath $marker -PathType Leaf) {
        $owned = Read-Ownership $marker $true
        if ((Get-NumericVersion $owned.Document.version) -gt $script:DesiredVersion) {
            throw 'Downgrading an existing installation is not supported.'
        }
        return $owned
    }
    if (Test-Path -LiteralPath $script:Target) {
        if (@(Get-ChildItem -LiteralPath $script:Target -Force).Count -gt 0) {
            throw 'The selected directory is not empty and has no matching product marker.'
        }
    }
    return $null
}

function Assert-RegisteredLocation {
    $registry = $null
    $key = $null
    try {
        $registry = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
            [Microsoft.Win32.RegistryHive]::CurrentUser,
            [Microsoft.Win32.RegistryView]::Registry64)
        $key = $registry.OpenSubKey('Software\Microsoft\Windows\CurrentVersion\Uninstall\' + $ProductId)
        if ($key) {
            $location = $key.GetValue('InstallLocation', $null)
            if (!($location -is [string]) -or !$location) {
                throw 'The existing installation entry has no valid InstallLocation.'
            }
            $registered = Get-SafeInstallDir $location
            if ($registered -ine $script:Target) {
                throw 'This product is already registered at another location. Uninstall it before moving it.'
            }
        }
    } finally {
        if ($key) { $key.Dispose() }
        if ($registry) { $registry.Dispose() }
    }
}

function Assert-ExclusiveFile([string]$Path) {
    Assert-NoReparse $Path
    if (Test-Path -LiteralPath $Path -PathType Container) { throw "An installed file is a directory: $Path" }
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $stream = $null
        try {
            $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
            if (([IO.File]::GetAttributes($Path) -band [IO.FileAttributes]::ReadOnly) -ne 0) {
                throw "An installed file is read-only: $Path"
            }
        } finally {
            if ($stream) { $stream.Dispose() }
        }
    }
}

function Get-Sha256File([string]$Path) {
    $stream = $null
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '')
    } finally {
        if ($stream) { $stream.Dispose() }
        $algorithm.Dispose()
    }
}

function Assert-InstallLayout($Existing, $Desired) {
    if ($Existing) {
        Assert-ExclusiveFile (Join-Path $script:Target $script:MarkerName)
        foreach ($file in $Existing.Files) {
            Assert-ExclusiveFile (Get-ContainedPath $script:Target $file.path)
        }
    }
    foreach ($file in $Desired.Files) {
        $destination = Get-ContainedPath $script:Target $file.path
        if (Test-Path -LiteralPath $destination) {
            if (!$Existing -or !$Existing.Map.ContainsKey($file.path)) {
                throw "An unknown file would be overwritten: $($file.path)"
            }
            if (Test-Path -LiteralPath $destination -PathType Container) {
                throw "A directory would be overwritten: $($file.path)"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($destination)
        while ($parent -and $parent -ine $script:Target) {
            if ((Test-Path -LiteralPath $parent) -and !(Test-Path -LiteralPath $parent -PathType Container)) {
                throw "A file blocks a payload directory: $parent"
            }
            $parent = [IO.Path]::GetDirectoryName($parent)
        }
    }
}

function New-SafeDirectory([string]$Path) {
    Assert-NoReparse $Path
    $cursor = $Path
    while ($cursor -and ($cursor -ieq $script:Target -or
           $cursor.StartsWith($script:Target + '\', [StringComparison]::OrdinalIgnoreCase))) {
        if (!(Test-Path -LiteralPath $cursor)) { $null = $script:CreatedTargetDirectories.Add($cursor) }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    $null = [IO.Directory]::CreateDirectory($Path)
    Assert-NoReparse $Path
}

function Remove-EmptyOwnedDirectories([object[]]$Files) {
    $directories = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $Files) {
        $parent = [IO.Path]::GetDirectoryName((Get-ContainedPath $script:Target $file.path))
        while ($parent -and $parent -ine $script:Target) {
            $null = $directories.Add($parent)
            $parent = [IO.Path]::GetDirectoryName($parent)
        }
    }
    foreach ($directory in @($directories | Sort-Object Length -Descending)) {
        Assert-NoReparse $directory
        if ((Test-Path -LiteralPath $directory -PathType Container) -and
            [IO.Directory]::GetFileSystemEntries($directory).Length -eq 0) {
            [IO.Directory]::Delete($directory, $false)
        }
    }
}

function Remove-PrivateStage([string]$Path) {
    # This recursion is limited to our newly created sibling staging directory.
    if (!$script:StageRoot -or $Path -ine $script:StageRoot) { throw 'Invalid staging cleanup boundary.' }
    Assert-NoReparse $Path
    if (!(Test-Path -LiteralPath $Path)) { return }
    $directories = New-Object 'System.Collections.Generic.List[string]'
    $queue = New-Object 'System.Collections.Generic.Queue[string]'
    $queue.Enqueue($Path)
    while ($queue.Count) {
        $directory = $queue.Dequeue()
        $directories.Add($directory)
        foreach ($item in @(Get-ChildItem -LiteralPath $directory -Force)) {
            Assert-NoReparse $item.FullName
            if ($item.PSIsContainer) { $queue.Enqueue($item.FullName) }
            else { [IO.File]::Delete($item.FullName) }
        }
    }
    foreach ($directory in @($directories | Sort-Object Length -Descending)) {
        [IO.Directory]::Delete($directory, $false)
    }
}

function Expand-VerifiedPayload($Desired, [string]$Destination) {
    if (!$PayloadPath -or !$PayloadSha256 -or $PayloadSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Install requires a payload ZIP and its SHA-256.'
    }
    Assert-NoReparse ([IO.Path]::GetFullPath($PayloadPath))
    $script:PayloadStream = [IO.File]::Open($PayloadPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $actual = [BitConverter]::ToString($algorithm.ComputeHash($script:PayloadStream)).Replace('-', '')
    } finally { $algorithm.Dispose() }
    if ($actual -ine $PayloadSha256) { throw 'The payload ZIP SHA-256 does not match.' }
    $script:PayloadStream.Position = 0
    Add-Type -AssemblyName System.IO.Compression
    $script:Archive = New-Object IO.Compression.ZipArchive($script:PayloadStream, [IO.Compression.ZipArchiveMode]::Read, $true)
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $buffer = New-Object byte[] 1048576
    foreach ($entry in $script:Archive.Entries) {
        $name = $entry.FullName
        $isDirectory = $name.EndsWith('/')
        if ($isDirectory) { $name = $name.TrimEnd('/') }
        $relative = Get-SafeRelativePath $name
        $unixType = ($entry.ExternalAttributes -shr 16) -band 0xF000
        if ($unixType -eq 0xA000 -or ($entry.ExternalAttributes -band 0x400) -ne 0) {
            throw "Links are not supported in the payload ZIP: $relative"
        }
        if (!$seen.Add($relative)) { throw "Duplicate ZIP entry: $relative" }
        if ($isDirectory) {
            if ($Desired.Map.ContainsKey($relative)) { throw "ZIP directory conflicts with a manifest file: $relative" }
            continue
        }
        if (!$Desired.Map.ContainsKey($relative)) { throw "ZIP entry is not owned by the manifest: $relative" }
        $expected = $Desired.Map[$relative]
        if ($entry.Length -ne [long]$expected.size) { throw "ZIP size does not match: $relative" }
        $destinationFile = Get-ContainedPath $Destination $relative
        New-SafeDirectory ([IO.Path]::GetDirectoryName($destinationFile))
        $inputStream = $null
        $outputStream = $null
        try {
            $inputStream = $entry.Open()
            $outputStream = [IO.File]::Open($destinationFile, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            [long]$written = 0
            while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $written += $read
                if ($written -gt [long]$expected.size) { throw "ZIP expands beyond its declared size: $relative" }
                $outputStream.Write($buffer, 0, $read)
            }
        } finally {
            if ($outputStream) { $outputStream.Dispose() }
            if ($inputStream) { $inputStream.Dispose() }
        }
        if ((Get-Item -LiteralPath $destinationFile).Length -ne [long]$expected.size -or
            (Get-Sha256File $destinationFile) -ine $expected.sha256) {
            throw "Extracted payload verification failed: $relative"
        }
    }
    foreach ($file in $Desired.Files) {
        if (!$seen.Contains($file.path) -or !(Test-Path -LiteralPath (Get-ContainedPath $Destination $file.path) -PathType Leaf)) {
            throw "The ZIP is missing a manifest file: $($file.path)"
        }
    }
}

function Invoke-Install($Existing, $Desired) {
    $parent = [IO.Path]::GetDirectoryName($script:Target)
    New-SafeDirectory $parent
    $script:StageRoot = Join-Path $parent ('.sop-stage-' + [Guid]::NewGuid().ToString('N'))
    New-SafeDirectory $script:StageRoot
    $newRoot = Join-Path $script:StageRoot 'new'
    $backupRoot = Join-Path $script:StageRoot 'backup'
    New-SafeDirectory $newRoot
    New-SafeDirectory $backupRoot
    Write-DeployLog 'Verifying and staging the payload before changing installed files.'
    Expand-VerifiedPayload $Desired $newRoot
    Assert-InstallLayout $Existing $Desired
    $backups = @{}
    if ($Existing) {
        foreach ($file in $Existing.Files) {
            $source = Get-ContainedPath $script:Target $file.path
            if (Test-Path -LiteralPath $source -PathType Leaf) {
                $backup = Get-ContainedPath $backupRoot $file.path
                New-SafeDirectory ([IO.Path]::GetDirectoryName($backup))
                [IO.File]::Copy($source, $backup, $false)
                $backups[$file.path] = $backup
            }
        }
    }
    $changed = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $marker = Join-Path $script:Target $script:MarkerName
    $oldMarker = if (Test-Path -LiteralPath $marker) { [IO.File]::ReadAllBytes($marker) } else { $null }
    $markerChanged = $false
    $rollbackFailed = $false
    try {
        New-SafeDirectory $script:Target
        foreach ($file in $Desired.Files) {
            $destination = Get-ContainedPath $script:Target $file.path
            New-SafeDirectory ([IO.Path]::GetDirectoryName($destination))
            if ((Test-Path -LiteralPath $destination) -and (!$Existing -or !$Existing.Map.ContainsKey($file.path))) {
                throw "An unknown file appeared during installation: $($file.path)"
            }
            $null = $changed.Add($file.path)
            [IO.File]::Copy((Get-ContainedPath $newRoot $file.path), $destination, $true)
        }
        if ($Existing) {
            foreach ($file in $Existing.Files) {
                if (!$Desired.Map.ContainsKey($file.path)) {
                    $destination = Get-ContainedPath $script:Target $file.path
                    $null = $changed.Add($file.path)
                    [IO.File]::Delete($destination)
                }
            }
        }
        $newMarker = Join-Path $script:StageRoot 'new-marker.json'
        $markerDocument = [ordered]@{
            productId = $ProductId; version = $Version; scope = 'current-user'; files = @($Desired.Files)
        }
        [IO.File]::WriteAllText($newMarker, ($markerDocument | ConvertTo-Json -Depth 8), $script:Utf8)
        $markerChanged = $true
        if (Test-Path -LiteralPath $marker) {
            [IO.File]::Replace($newMarker, $marker, (Join-Path $script:StageRoot 'old-marker.json'))
        }
        else { [IO.File]::Move($newMarker, $marker) }
    } catch {
        Write-DeployLog ('Installation failed; restoring the previous files: ' + $_.Exception.Message)
        foreach ($relative in $changed) {
            try {
                $destination = Get-ContainedPath $script:Target $relative
                if ($backups.ContainsKey($relative)) { [IO.File]::Copy($backups[$relative], $destination, $true) }
                elseif (Test-Path -LiteralPath $destination -PathType Leaf) { [IO.File]::Delete($destination) }
            } catch {
                $rollbackFailed = $true
                Write-DeployLog ("Rollback failed for ${relative}: " + $_.Exception.Message)
            }
        }
        if ($markerChanged) {
            try {
                if ($null -ne $oldMarker) { [IO.File]::WriteAllBytes($marker, $oldMarker) }
                elseif (Test-Path -LiteralPath $marker) { [IO.File]::Delete($marker) }
            } catch {
                $rollbackFailed = $true
                Write-DeployLog ('Marker rollback failed: ' + $_.Exception.Message)
            }
        }
        if ($rollbackFailed) {
            Write-DeployLog ('Recovery copies retained at: ' + $script:StageRoot)
            $script:StageRoot = $null
        } else {
            try {
                foreach ($directory in @($script:CreatedTargetDirectories | Sort-Object Length -Descending)) {
                    Assert-NoReparse $directory
                    if ((Test-Path -LiteralPath $directory -PathType Container) -and
                        [IO.Directory]::GetFileSystemEntries($directory).Length -eq 0) {
                        [IO.Directory]::Delete($directory, $false)
                    }
                }
            } catch { Write-DeployLog ('Rollback retained empty directories: ' + $_.Exception.Message) }
        }
        throw
    }
    if ($Existing) {
        try { Remove-EmptyOwnedDirectories $Existing.Files }
        catch { Write-DeployLog ('Unused directories were retained: ' + $_.Exception.Message) }
    }
    Write-DeployLog 'Payload installation completed; user data and unknown files were preserved.'
}

function Invoke-Uninstall($Existing) {
    if (!$Existing) { throw 'No matching installed product marker was found.' }
    if ($Existing.Document.version -cne $Version) {
        throw 'The uninstaller version does not match the installed product.'
    }
    $marker = Join-Path $script:Target $script:MarkerName
    Assert-ExclusiveFile $marker
    # Preflight every owned file before deleting anything. A locked file leaves
    # both marker and the NSIS uninstall entry available for a retry.
    foreach ($file in $Existing.Files) {
        Assert-ExclusiveFile (Get-ContainedPath $script:Target $file.path)
    }
    foreach ($file in $Existing.Files) {
        $path = Get-ContainedPath $script:Target $file.path
        [IO.File]::Delete($path)
        if (Test-Path -LiteralPath $path) { throw "Could not remove an owned file: $($file.path)" }
    }
    try { Remove-EmptyOwnedDirectories $Existing.Files }
    catch { Write-DeployLog ('Empty directories were retained: ' + $_.Exception.Message) }
    # NSIS removes this marker only after uninstall registry/shortcut cleanup.
    Write-DeployLog 'Owned payload removed; unknown files and AppData were preserved.'
}

try {
    if ($Action -notin @('Validate', 'Install', 'Uninstall')) { throw 'Action must be Validate, Install, or Uninstall.' }
    if (!$ProductId -or $ProductId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$') { throw 'Invalid product identity.' }
    if (!$ExeName -or [IO.Path]::GetFileName($ExeName) -cne $ExeName -or $ExeName -notmatch '(?i)\.exe$') {
        throw 'ExeName must be a simple executable file name.'
    }
    $script:DesiredVersion = Get-NumericVersion $Version
    $script:Target = Get-SafeInstallDir $InstallDir
    Write-DeployLog ("Action=$Action Product=$ProductId Version=$Version Target=$($script:Target)")
    if ($Action -in @('Validate', 'Install')) { Assert-RegisteredLocation }
    $existing = Get-ExistingInstallation
    $desired = $null
    if ($ManifestPath) {
        $desired = Read-Ownership $ManifestPath $false
        if ($desired.Document.version -cne $Version) { throw 'The payload manifest version does not match this installer.' }
    }
    if ($Action -eq 'Install') {
        if (!$desired) { throw 'Install requires an embedded ownership manifest.' }
        Assert-InstallLayout $existing $desired
        Invoke-Install $existing $desired
    } elseif ($Action -eq 'Uninstall') {
        Invoke-Uninstall $existing
    } elseif ($desired) {
        Assert-InstallLayout $existing $desired
    }
    [Console]::Out.WriteLine((@{ action = $Action; installDir = $script:Target; productId = $ProductId; version = $Version; status = 'ok' } | ConvertTo-Json -Compress))
    exit 0
} catch {
    Write-DeployLog ('ERROR: ' + $_.Exception.Message)
    exit 1
} finally {
    if ($script:Archive) { $script:Archive.Dispose() }
    if ($script:PayloadStream) { $script:PayloadStream.Dispose() }
    if ($script:StageRoot) {
        try { Remove-PrivateStage $script:StageRoot }
        catch { Write-DeployLog ('Staging cleanup was incomplete: ' + $_.Exception.Message) }
    }
}
