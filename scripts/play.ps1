<#
.SYNOPSIS
    Launches an unpacked title through Strata with optional profile, GPU adapter,
    resolution, FSR 4K preset, and environment overrides.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Exe,

    [ValidateSet("series-x", "series-s")]
    [string]$Console = "series-x",

    [string]$Adapter,

    [string]$Output,

    [ValidateSet("quality", "balanced", "performance")]
    [string]$Fsr4K,

    [double]$FrameIntervalMs,

    [ValidateSet("trap", "zero", "notimpl")]
    [string]$Unimplemented = "trap",

    [switch]$Safe,

    [switch]$Release,

    [hashtable]$More = @{}
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ProfileDir = if ($Release) { "release" } else { "diag" }
$Bin = Join-Path $RepoRoot "target\$ProfileDir\strata-cli.exe"

if (-not (Test-Path $Bin)) {
    Write-Host "Building strata-cli ($ProfileDir)..." -ForegroundColor Cyan
    Push-Location $RepoRoot
    try {
        if ($Release) {
            cargo build --release -p strata-cli
        } else {
            cargo build --profile diag -p strata-cli
        }
    } finally {
        Pop-Location
    }
}

$env:STRATA_CONSOLE = $Console
if ($Adapter) { $env:STRATA_GPU_ADAPTER = $Adapter }
if ($Output) { $env:STRATA_OUTPUT = $Output }
if ($Fsr4K) { $env:STRATA_FSR_4K = $Fsr4K }
if ($FrameIntervalMs -gt 0) { $env:STRATA_FRAME_INTERVAL_MS = "$FrameIntervalMs" }
if ($Safe) { $env:STRATA_SAFE = "1" }
foreach ($k in $More.Keys) {
    [Environment]::SetEnvironmentVariable($k, "$($More[$k])", "Process")
}

if (-not $Exe) {
    Write-Host "No -Exe specified; running strata-cli probe..." -ForegroundColor Yellow
    & $Bin probe
    exit $LASTEXITCODE
}

& $Bin run $Exe --unimplemented $Unimplemented
exit $LASTEXITCODE
