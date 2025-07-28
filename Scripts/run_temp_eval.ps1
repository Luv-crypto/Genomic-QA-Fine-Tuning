# scripts/run_eval.ps1
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 1) cd to project root
$scriptPath  = $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path $scriptPath -Parent | Split-Path -Parent
Set-Location $projectRoot

# 2) ensure YAML parser
if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Install-Module powershell-yaml -Scope CurrentUser -Force
}
Import-Module powershell-yaml -ErrorAction Stop
$env:PYTHONPATH = Join-Path $projectRoot 'src'
# 3) load params and compute paths
$p = ConvertFrom-Yaml (Get-Content 'params.yaml' -Raw)
$run   = $p.run_name
$fam   = $p.family
$wip   = "$($p.output_root)/$fam/$run"

# 4) run eval
Write-Host "`n Evaluating run: $run"
& python -m model_lab.eval `
    --model   $wip `
    --dataset $p.eval.path `
    --batch   $p.eval.batch_size `
    --max-len $p.eval.max_length `
    --offload

if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }

# 5) summary
Write-Host "`n Evaluation complete"
Write-Host "   metrics to metrics/eval_ci.json"
