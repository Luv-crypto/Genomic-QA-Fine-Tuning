# scripts/run_train.ps1  (update)

param()
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

Import-Module powershell-yaml -ErrorAction Stop
$params = ConvertFrom-Yaml (Get-Content params.yaml -Raw)

$stamp   = (Get-Date -Format "yyyy-MM-ddTHH-mm")
$runName = $params.run_name
$wipDir  = "models/hf/wip/$stamp-$runName"

# ─── CALL TRAINER (no CLI args required) ───────────────────────
python -m model_lab.train
if ($LASTEXITCODE) { throw "Training failed." }

# ─── quick local evaluation for sanity check ───────────────────
python eval.py `
  --model   $wipDir `
  --dataset $params.dataset.path `
  --batch   $params.eval.batch_size `
  --max-len $params.eval.max_length `
  --offload

Write-Host ""
Write-Host "✔  Finished local run"
Write-Host "   wip folder : $wipDir"
Write-Host "   metrics    : metrics/eval_local.json"
