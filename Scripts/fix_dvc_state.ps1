<#
Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts\fix_dvc_state.ps1
#>

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ─── 0) snapshot current git state ────────────────────────────────
git add -A
git commit -m "wip: pre-DVC-cleanup snapshot" -q 2>$null

Write-Host "`n Git snapshot created (or repo already clean)"

# ─── 1) manual artefacts to track (not stage outputs) ─────────────
$adds = @(
  'models/hf/wip/run-001',
  'models/hf/stable/qwen',
  'data/raw/genomic_qa_dataset_5000.json',
  'data/raw/eval.json'
)

foreach ($path in $adds) {
  if (Test-Path $path) {
    Write-Host "dvc add $path"
    dvc add $path | Out-Null
  } else {
    Write-Warning "path not found, skipped: $path"
  }
}

# ─── 2) register existing stage outputs (GGUF files) ──────────────
Write-Host "`nCommitting stage outputs…"
dvc commit gguf   | Out-Null
dvc commit quant  | Out-Null

# ─── 3) git-add pointer files produced above ─────────────────────
$pointerPaths = @(
  'models\hf\wip\run-001.dvc',
  'models\hf\stable\qwen.dvc',
  'data\raw\eval.json.dvc',
  'data\raw\genomic_qa_dataset_5000.json.dvc',
  'models\gguf\run-000_full.gguf.dvc',
  'models\gguf\run-000_q4_k_m.gguf.dvc'
)

foreach ($p in $pointerPaths) {
  if (Test-Path $p) { git add $p }
}

# also add gitignore files created by dvc
Get-ChildItem -Recurse -Filter '.gitignore' models | ForEach-Object { git add $_.FullName }

git add dvc.yaml
git commit -m "fix(dvc): register artefacts & commit gguf stage outs"

Write-Host "`n  Repository is now clean:"
dvc status -c
