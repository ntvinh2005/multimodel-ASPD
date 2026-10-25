[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRepo,

    [string]$LocalRepo = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$relativeRun = 'out/multimodel/runs/qwen25_aspd_original_100m'
$localRoot = (Resolve-Path -LiteralPath $LocalRepo).Path
$localRun = Join-Path $localRoot ($relativeRun -replace '/', [IO.Path]::DirectorySeparatorChar)
if (Test-Path -LiteralPath $localRun) {
    throw "Local result already exists; refusing to overwrite: $localRun"
}
$localParent = Split-Path -Parent $localRun
New-Item -ItemType Directory -Path $localParent -Force | Out-Null

$remoteRun = $RemoteRepo.TrimEnd('/') + '/' + $relativeRun
& scp -r "${Remote}:$remoteRun" $localParent
if ($LASTEXITCODE -ne 0) {
    throw "Artifact transfer failed with scp exit code $LASTEXITCODE"
}

$manifestPath = Join-Path $localRun 'result_package_manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Transfer completed but result_package_manifest.json is missing: $localRun"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.status -ne 'P6_HANDOFF_READY') {
    throw "Unexpected package status: $($manifest.status)"
}
foreach ($artifact in $manifest.artifacts) {
    $artifactPath = Join-Path $localRun ($artifact.path -replace '/', [IO.Path]::DirectorySeparatorChar)
    if (-not (Test-Path -LiteralPath $artifactPath)) {
        throw "Manifest artifact is missing: $artifactPath"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifactPath).Hash.ToLowerInvariant()
    if ($actual -ne $artifact.sha256.ToLowerInvariant()) {
        throw "Checksum mismatch: $artifactPath"
    }
}
Write-Host "Pulled: $localRun"
Write-Host "Manifest artifacts: $($manifest.artifacts.Count); checksums: PASS"
Write-Host "Status: $($manifest.status); evaluation: $($manifest.evaluation_status)"
