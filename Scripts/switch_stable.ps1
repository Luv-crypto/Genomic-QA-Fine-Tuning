# scripts/switch_stable.ps1 <run_name>
# Promote a candidate run to stable for the current family.
# Keeps previous stable pointer in archive for easy rollback.
param([Parameter(Mandatory)][string]$RunName)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Install-Module powershell-yaml -Scope CurrentUser -Force
}
Import-Module powershell-yaml -ErrorAction Stop

$params = ConvertFrom-Yaml (Get-Content 'params.yaml' -Raw)
$family = $params.family
$candidate = "models/hf/candidates/$family/${RunName}_merged"
if (-not (Test-Path $candidate)) {
    throw "Candidate not found: $candidate"
}

$stableFile = "models/hf/stable/$family.dvc"
$archiveDir = "models/hf/stable/archive/$family"
if (Test-Path $stableFile) {
    $ts = Get-Date -Format 'yyyy-MM-ddTHH-mm'
    New-Item -ItemType Directory $archiveDir -Force | Out-Null
    Move-Item $stableFile "$archiveDir/${ts}.dvc"
}

# create new pointer referencing the candidate
$dvcArgs = "add $candidate -f $stableFile"
Write-Host "dvc $dvcArgs"
dvc add $candidate -f $stableFile

git add $stableFile
$commitMsg = "chore(stable): set $RunName for $family"
git commit -m $commitMsg

Write-Host "`n✔ Stable updated to $RunName for $family"
Write-Host "Run: dvc push && git push"
