[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRepo,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9]+$')]
    [string]$JobId,

    [string]$ArtifactRelativeParent = 'out/multimodel/diagnostics/p6_random_control_smoke',

    [string]$SlurmJobName = 'mm-p6-random-smoke',

    [string]$LocalRepo = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$localRoot = (Resolve-Path -LiteralPath $LocalRepo).Path
$relativeParts = $ArtifactRelativeParent -split '/'
$localParent = $localRoot
foreach ($part in $relativeParts) {
    $localParent = Join-Path $localParent $part
}
$localJob = Join-Path $localParent "job_$JobId"
if (Test-Path -LiteralPath $localJob) {
    throw "Local result already exists; refusing to overwrite: $localJob"
}
New-Item -ItemType Directory -Path $localParent -Force | Out-Null

$remoteParent = $RemoteRepo.TrimEnd('/') + '/' + $ArtifactRelativeParent.Trim('/')
$remoteJob = "$remoteParent/job_$JobId"
& scp -r "${Remote}:$remoteJob" $localParent
if ($LASTEXITCODE -ne 0) {
    throw "Artifact transfer failed with scp exit code $LASTEXITCODE"
}

foreach ($extension in @('out', 'err')) {
    $remoteLog = $RemoteRepo.TrimEnd('/') + "/slurm/logs/$SlurmJobName-$JobId.$extension"
    & scp "${Remote}:$remoteLog" $localJob
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not transfer optional Slurm .$extension log"
    }
}

$summary = Join-Path $localJob 'smoke_summary.json'
if (-not (Test-Path -LiteralPath $summary)) {
    throw "Transfer completed but smoke_summary.json is missing: $localJob"
}
$checksumPath = Join-Path $localJob 'SHA256SUMS'
if (-not (Test-Path -LiteralPath $checksumPath)) {
    throw "Transfer completed but SHA256SUMS is missing: $localJob"
}
foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
        throw "Invalid SHA256SUMS line: $line"
    }
    $expectedHash = $Matches[1].ToLowerInvariant()
    $fileName = $Matches[2]
    $filePath = Join-Path $localJob $fileName
    if (-not (Test-Path -LiteralPath $filePath)) {
        throw "Checksummed artifact is missing: $filePath"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $filePath).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Checksum mismatch: $filePath"
    }
}
$payload = Get-Content -Raw -LiteralPath $summary | ConvertFrom-Json
Write-Host "Pulled: $localJob"
Write-Host "Checksums: PASS"
Write-Host "Status: $($payload.status); exit_code: $($payload.exit_code); commit: $($payload.git_commit)"
