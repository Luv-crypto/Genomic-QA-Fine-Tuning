# scripts/promote.ps1  <timestamp-run-xxx>
# param([Parameter(Mandatory=$true)][string]$Wip)

# Import-Module powershell-yaml -ErrorAction Stop
# $repo = Split-Path $PSScriptRoot -Parent
# Set-Location $repo

# $params  = ConvertFrom-Yaml (Get-Content params.yaml -Raw)
# $runName = $params.run_name
# if (-not $Wip.EndsWith($runName)) {
#   throw "WIP folder '$Wip' does not match run_name '$runName'"
# }

# $candDir = "models/hf/candidates/$runName"
# if (Test-Path $candDir) { throw "Candidate $runName already exists" }

# Copy-Item "models/hf/wip/$Wip" $candDir -Recurse
# dvc add $candDir
# git add $candDir.dvc
# git commit -m "chore(promote): $runName to candidates"
# Write-Host "✔ Promoted. Now run:  dvc push  &&  git push"


# scripts/promote.ps1
param([Parameter(Mandatory)] [string]$WipFolder)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

Import-Module powershell-yaml -ErrorAction Stop
$params = ConvertFrom-Yaml (Get-Content params.yaml -Raw)
$runName = $params.run_name

if (-not $WipFolder.EndsWith($runName)) {
  throw "WIP folder '$WipFolder' does not match run_name '$runName'"
}
$candDir = "models/hf/candidates/$runName"
if (Test-Path $candDir) { throw "Candidate '$runName' already exists" }

# 1) copy wip → candidates
Copy-Item "models/hf/wip/$WipFolder" $candDir -Recurse

# 2) ask quantization
$answer = Read-Host "Quantize GGUF to Q4_0? (y/n)"
switch ($answer.ToLower()) {
  'y' { $params.ci.quantize = $true }
  'n' { $params.ci.quantize = $false }
  default { throw "Please answer 'y' or 'n'." }
}

# 3) embed llama.cpp venv info
#    adjust these paths to your local llama.cpp venv
$params.ci.env.LLAMA_CPP_DIR = "D:/LLM_prod/Genomic-QA-Fine-Tuning/llama.cpp"
$params.ci.env.LLAMA_PY      = "D:/LLM_prod/Genomic-QA-Fine-Tuning/weights_onv/Scripts/python.exe"


# 4) write back YAML
$params | ConvertTo-Yaml | Set-Content params.yaml

# 5) DVC + Git commit
dvc add $candDir
git add $candDir.dvc params.yaml
git commit -m "chore(promote): $runName (quantize=$($params.ci.quantize))"

Write-Host "`n✔ Promoted '$runName' with quantize=$($params.ci.quantize)"
Write-Host "Now run: dvc push && git push"

