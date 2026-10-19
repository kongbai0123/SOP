param(
    [Parameter(Mandatory = $true)][string]$ReleaseDirectory,
    [string]$PayloadDirectory,
    [string]$OutputPath,
    [ValidateRange(1, 168)][int]$EventLookbackHours = 24,
    [ValidateRange(1, 500)][int]$MaximumEvents = 100
)

# Read-only inspection. The optional JSON report is the only write operation.
# Do not execute artifacts or change policies, certificates, ADS, or registry.
$ErrorActionPreference = 'Stop'

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function Get-CertificateSummary($Certificate) {
    if ($null -eq $Certificate) { return $null }
    return [ordered]@{
        subject = $Certificate.Subject
        issuer = $Certificate.Issuer
        thumbprint = $Certificate.Thumbprint
        validFromUtc = $Certificate.NotBefore.ToUniversalTime().ToString('o')
        validToUtc = $Certificate.NotAfter.ToUniversalTime().ToString('o')
    }
}

function Get-SignatureSummary([string]$Path, [string]$Role) {
    $result = [ordered]@{ role = $Role; path = $Path; status = 'Unavailable' }
    if (-not [IO.File]::Exists($Path)) {
        $result.status = 'Missing'
        return $result
    }
    try {
        $result.size = (Get-Item -LiteralPath $Path).Length
        $result.sha256 = Get-Sha256 $Path
        $signature = Get-AuthenticodeSignature -LiteralPath $Path -ErrorAction Stop
        $result.status = [string]$signature.Status
        $result.statusMessage = [string]$signature.StatusMessage
        $result.signatureType = [string]$signature.SignatureType
        $result.signerCertificate = Get-CertificateSummary $signature.SignerCertificate
        $result.timestampCertificate = Get-CertificateSummary $signature.TimeStamperCertificate
    } catch {
        $result.error = $_.Exception.Message
        $result.errorId = $_.FullyQualifiedErrorId
    }
    return $result
}

function Get-RelatedEvents([string]$LogName, [int[]]$Ids, $Names) {
    $result = [ordered]@{
        status = 'Unavailable'; logName = $LogName; eventIds = $Ids
        sinceUtc = [DateTime]::UtcNow.AddHours(-$EventLookbackHours).ToString('o')
        maximumScannedEvents = $MaximumEvents; matches = @()
        interpretation = 'Historical filename/ActivityID associations only; absence is not permission to run. Review XML, hashes, path, and policy for exact attribution.'
    }
    try {
        $events = @(Get-WinEvent -FilterHashtable @{
            LogName = $LogName; Id = $Ids
            StartTime = [DateTime]::UtcNow.AddHours(-$EventLookbackHours)
        } -MaxEvents $MaximumEvents -ErrorAction Stop)
        $result.scannedCount = $events.Count
        $result.scanLimitReached = $events.Count -ge $MaximumEvents
        $records = @()
        $matchedActivities = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($event in $events) {
            $rawXml = $event.ToXml()
            $xml = [xml]$rawXml
            $data = @()
            $nameMatch = $false
            foreach ($node in @($xml.Event.EventData.Data)) {
                if ($null -eq $node) { continue }
                $text = [string]$node.InnerText
                foreach ($name in $Names) {
                    if ($text.IndexOf($name, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $nameMatch = $true }
                }
                if ($text.Length -gt 4000) { $text = $text.Substring(0, 4000) }
                $data += [ordered]@{ name = [string]$node.Name; value = $text }
            }
            $activityId = [string]$xml.Event.System.Correlation.ActivityID
            if ($nameMatch -and $activityId) { [void]$matchedActivities.Add($activityId) }
            $records += [pscustomobject]@{
                id = $event.Id; recordId = $event.RecordId
                timeCreatedUtc = $event.TimeCreated.ToUniversalTime().ToString('o')
                activityId = $activityId; filenameMatched = $nameMatch; data = $data; xml = $rawXml
            }
        }
        $result.matches = @($records | Where-Object { $_.filenameMatched -or ($_.activityId -and $matchedActivities.Contains($_.activityId)) })
        $result.status = 'Collected'
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {
            $result.status = 'NoMatchingEventsInWindow'
            $result.scannedCount = 0
        } else {
            $result.error = $_.Exception.Message
            $result.errorId = $_.FullyQualifiedErrorId
        }
    }
    return $result
}

$releaseRoot = [IO.Path]::GetFullPath($ReleaseDirectory)
$releasePrefix = $releaseRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if (-not [IO.Directory]::Exists($releaseRoot)) { throw 'ReleaseDirectory must be an existing directory.' }
$manifestPath = Join-Path $releaseRoot 'release-manifest.json'
$protectedPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
[void]$protectedPaths.Add($manifestPath)
$report = [ordered]@{
    tool = 'check_windows_trust'
    schemaVersion = 1
    capturedAtUtc = [DateTime]::UtcNow.ToString('o')
    powershellVersion = [string]$PSVersionTable.PSVersion
    releaseDirectory = $releaseRoot
    inspectionOnly = $true
    artifactExecutionPerformed = $false
    policyOrTrustChangesPerformed = $false
    integrity = [ordered]@{ status = 'Unavailable'; manifestPath = $manifestPath; artifacts = @() }
    signatures = @()
    payloadSignatureScope = 'Not requested. Pass -PayloadDirectory to inspect its SOP.exe without executing it.'
    deviceGuard = [ordered]@{ status = 'Unavailable' }
    ciTool = [ordered]@{ status = 'Unavailable'; command = 'Windows System32/CiTool.exe -lp -json' }
    launchPermission = 'Not determined. A valid signature or a clean report does not guarantee execution on this or another PC.'
    limitations = @(
        'SHA-256 and size compare to this local manifest, not to an independently authenticated publisher.',
        'Signature status reflects this PC and current trust/revocation availability; it is not a target-policy decision.',
        'No EXE, DLL, script, or installer from the release is executed.',
        'DeviceGuard fields and bounded CodeIntegrity history do not enumerate every security control or predict launch permission.',
        'This report may contain local file paths and event data. Review it before sharing.'
    )
    sources = @(
        'https://learn.microsoft.com/powershell/module/microsoft.powershell.security/get-authenticodesignature',
        'https://learn.microsoft.com/windows/security/hardware-security/enable-virtualization-based-protection-of-code-integrity',
        'https://learn.microsoft.com/windows/security/application-security/application-control/app-control-for-business/operations/citool-commands',
        'https://learn.microsoft.com/windows/security/application-security/application-control/app-control-for-business/operations/event-id-explanations',
        'https://learn.microsoft.com/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting'
    )
}
$targetNames = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
[void]$targetNames.Add('SOP.exe')
foreach ($name in @('deploy.ps1', 'shortcuts.ps1', 'check_windows_trust.ps1')) { [void]$targetNames.Add($name) }
$integrityOk = $false
try {
    $manifest = [IO.File]::ReadAllText($manifestPath, [Text.Encoding]::UTF8) | ConvertFrom-Json
    $report.productId = [string]$manifest.productId
    $report.version = [string]$manifest.version
    $artifacts = @($manifest.artifacts)
    if ($artifacts.Count -eq 0) { throw 'Manifest contains no artifacts.' }
    $seenPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $integrityOk = $true
    foreach ($artifact in $artifacts) {
        $relativePath = [string]$artifact.path
        $item = [ordered]@{ path = $relativePath; status = 'Failed' }
        try {
            if (-not $relativePath -or [IO.Path]::IsPathRooted($relativePath) -or
                $relativePath.Contains(':') -or ($relativePath -split '[/\\]') -contains '..') {
                throw 'Artifact path must stay inside ReleaseDirectory.'
            }
            $fullPath = [IO.Path]::GetFullPath((Join-Path $releaseRoot $relativePath))
            if (-not $fullPath.StartsWith($releasePrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Artifact path escaped ReleaseDirectory.'
            }
            if (-not $seenPaths.Add($fullPath)) { throw 'Duplicate artifact path.' }
            [void]$protectedPaths.Add($fullPath)
            if ([string]$artifact.sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid expected SHA-256.' }
            $item.expectedSize = [long]$artifact.size
            if ($item.expectedSize -lt 0) { throw 'Invalid expected size.' }
            $item.expectedSha256 = ([string]$artifact.sha256).ToLowerInvariant()
            if (-not [IO.File]::Exists($fullPath)) { throw 'Artifact file is missing.' }
            $item.actualSize = (Get-Item -LiteralPath $fullPath).Length
            $item.actualSha256 = Get-Sha256 $fullPath
            $item.sizeMatches = $item.actualSize -eq $item.expectedSize
            $item.sha256Matches = $item.actualSha256 -eq $item.expectedSha256
            if ($item.sizeMatches -and $item.sha256Matches) { $item.status = 'Passed' }
            else { $integrityOk = $false }
            if ([IO.Path]::GetExtension($fullPath) -ieq '.exe') {
                [void]$targetNames.Add([IO.Path]::GetFileName($fullPath))
                $report.signatures += Get-SignatureSummary $fullPath 'release-executable'
            }
        } catch {
            $item.error = $_.Exception.Message
            $integrityOk = $false
        }
        $report.integrity.artifacts += $item
    }
    if ($integrityOk) { $report.integrity.status = 'Passed' }
    else { $report.integrity.status = 'Failed' }
} catch {
    $report.integrity.error = $_.Exception.Message
    $report.integrity.errorId = $_.FullyQualifiedErrorId
}

if ($PayloadDirectory) {
    $payloadRoot = [IO.Path]::GetFullPath($PayloadDirectory)
    $payloadExe = Join-Path $payloadRoot 'SOP.exe'
    [void]$protectedPaths.Add($payloadExe)
    $report.signatures += Get-SignatureSummary $payloadExe 'explicit-payload-entry'
    $report.payloadSignatureScope = 'Only SOP.exe in the explicitly supplied directory; no archive extraction or DLL signature sweep. Directory-to-ZIP equivalence is not established.'
}

try {
    $guard = @(Get-CimInstance -Namespace 'root\Microsoft\Windows\DeviceGuard' -ClassName 'Win32_DeviceGuard' -OperationTimeoutSec 10 -ErrorAction Stop)
    $values = @()
    foreach ($instance in $guard) {
        $value = [ordered]@{}
        foreach ($name in @('CodeIntegrityPolicyEnforcementStatus', 'UsermodeCodeIntegrityPolicyEnforcementStatus', 'VirtualizationBasedSecurityStatus', 'SecurityServicesConfigured', 'SecurityServicesRunning')) {
            if ($null -ne $instance.PSObject.Properties[$name]) { $value[$name] = $instance.$name }
        }
        $values += $value
    }
    $report.deviceGuard.status = 'Collected'
    $report.deviceGuard.rawValues = $values
    $report.deviceGuard.interpretation = 'Raw WMI fields only. No inference that the product is allowed or that all policies are visible.'
} catch {
    $report.deviceGuard.error = $_.Exception.Message
    $report.deviceGuard.errorId = $_.FullyQualifiedErrorId
}

$ciToolPath = Join-Path $env:SystemRoot 'System32\CiTool.exe'
if ([IO.File]::Exists($ciToolPath)) {
    $process = $null
    try {
        $startInfo = New-Object Diagnostics.ProcessStartInfo
        $startInfo.FileName = $ciToolPath
        $startInfo.Arguments = '-lp -json'
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = New-Object Diagnostics.Process
        $process.StartInfo = $startInfo
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(10000)) {
            $process.Kill()
            throw 'The read-only CiTool query timed out after 10 seconds; it was not retried.'
        }
        $outputText = $stdout.GetAwaiter().GetResult()
        $errorText = $stderr.GetAwaiter().GetResult()
        $report.ciTool.exitCode = $process.ExitCode
        if ($process.ExitCode -ne 0) { throw ('CiTool list query failed: ' + $errorText + $outputText) }
        $report.ciTool.rawResult = $outputText | ConvertFrom-Json
        $report.ciTool.status = 'Collected'
        $report.ciTool.interpretation = 'Policy listing only, including potentially inactive policies. No inference that this product is allowed.'
    } catch {
        $report.ciTool.error = $_.Exception.Message
        $report.ciTool.errorId = $_.FullyQualifiedErrorId
    } finally {
        if ($null -ne $process) { $process.Dispose() }
    }
} else {
    $report.ciTool.status = 'NotAvailable'
    $report.ciTool.reason = 'System CiTool was not found. It is not included in every Windows version.'
}

$report.codeIntegrityEvents = Get-RelatedEvents 'Microsoft-Windows-CodeIntegrity/Operational' @(3076, 3077, 3089) $targetNames
$report.scriptAndMsiEvents = Get-RelatedEvents 'Microsoft-Windows-AppLocker/MSI and Script' @(8028, 8029, 8036, 8038) $targetNames

$json = $report | ConvertTo-Json -Depth 12
if ($OutputPath) {
    $destination = [IO.Path]::GetFullPath($OutputPath)
    if ([IO.Path]::GetExtension($destination) -ine '.json') { throw 'OutputPath must end in .json.' }
    if ($protectedPaths.Contains($destination)) { throw 'OutputPath cannot overwrite a manifest or inspected artifact.' }
    if (-not [IO.Directory]::Exists([IO.Path]::GetDirectoryName($destination))) { throw 'OutputPath parent directory must already exist.' }
    if ([IO.File]::Exists($destination)) {
        $previous = [IO.File]::ReadAllText($destination, [Text.Encoding]::UTF8) | ConvertFrom-Json
        if ($previous.tool -ne 'check_windows_trust' -or $previous.schemaVersion -ne 1) {
            throw 'OutputPath already exists and is not a previous report from this tool.'
        }
    }
    [IO.File]::WriteAllText($destination, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding $false))
}
$json
if (-not $integrityOk) { exit 2 }
exit 0
