# scripts/run_train.ps1
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ─── 1) Determine project root based on this script’s location ───────────────────
# $MyInvocation.MyCommand.Definition gives the full path to this .ps1 file
$scriptPath  = $MyInvocation.MyCommand.Definition
$scriptDir   = Split-Path $scriptPath -Parent        # ...\Genomic-QA-Fine-Tuning\scripts
$projectRoot = Split-Path $scriptDir -Parent         # ...\Genomic-QA-Fine-Tuning
Set-Location $projectRoot

# ─── 2) Ensure powershell-yaml is available ─────────────────────────────────────
if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Write-Host 'Installing powershell-yaml module…'
    Install-Module powershell-yaml -Scope CurrentUser -Force -ErrorAction Stop
}
Import-Module powershell-yaml -ErrorAction Stop

# ─── 3) Load params.yaml from project root ───────────────────────────────────────
$params = ConvertFrom-Yaml (Get-Content 'params.yaml' -Raw)

# ─── 4) Compute the WIP folder path ──────────────────────────────────────────────
$stamp = Get-Date -Format 'yyyy-MM-ddTHH-mm'
$wip   = "$($params.output_root)/$stamp-$($params.run_name)"

# ▶ Add this:
$env:PYTHONPATH = Join-Path $projectRoot 'src'
# ─── 5) Run training ─────────────────────────────────────────────────────────────
Write-Host "`n▶ Starting training..."
& python -m model_lab.train
if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }

# ─── 6) Quick local evaluation ─────────────────────────────────────────────────
# Write-Host "`n▶ Running quick local eval..."
# & python -m model_lab.eval --model $wip --dataset $params.dataset.path `
#     --batch $params.eval.batch_size --max-len $params.eval.max_length --offload
# if ($LASTEXITCODE -ne 0) { throw 'Local eval failed.' }

# # ─── 7) Summary ─────────────────────────────────────────────────────────────────
# Write-Host ''
# Write-Host ("   wip folder : {0}" -f $wip)
# Write-Host '   metrics    : metrics/eval_local.json'
