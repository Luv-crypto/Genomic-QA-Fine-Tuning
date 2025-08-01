<#
Bootstrap the first stable container for any model family.
#>

param(
    [Parameter(Mandatory)][string]$Family,
    [string]$Run = 'run-000',
    [string]$Mode = 'q4_k_m'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- HELPER FUNCTION to replace the broken ConvertTo-Json cmdlet ---
function ConvertTo-SimpleJson {
    param($Hashtable)
    $lines = @("{")
    $entryCount = $Hashtable.Count
    $i = 0
    foreach ($key in $Hashtable.Keys) {
        $i++
        $familyData = $Hashtable[$key]
        
        # Corrected way to handle potential null values
        $stable = if ($familyData.stable -ne $null) { $familyData.stable } else { "" }
        $candidate = if ($familyData.candidate -ne $null) { $familyData.candidate } else { "" }

        # Add the family key
        $lines += "  `"$key`": {"

        # Add the nested key-value pairs
        $lines += "    `"stable`": `"$stable`","
        $lines += "    `"candidate`": `"$candidate`""

        # Add closing brace for the family, with a comma if it's not the last one
        if ($i -lt $entryCount) {
            $lines += "  },"
        } else {
            $lines += "  }"
        }
    }
    $lines += "}"
    return ($lines -join "`r`n")
}
# -----------------------------------------------------------------


# ── 0. repo root ──────────────────────────────────────────────────────────────
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot      # critical: puts us next to docker-compose.yml

# ── 0.5. AUTOMATED: Convert entrypoint.sh to use LF line endings ───────────────
Write-Host "`nSTEP 0.5 Converting entrypoint.sh to use LF line endings..."
$scriptPath = "$RepoRoot/Scripts/entrypoint.sh"
if (Test-Path $scriptPath) {
    # Read the file content, replacing Windows line endings with Unix line endings
    $content = Get-Content -Path $scriptPath -Raw
    $content = $content -replace "`r`n", "`n"
    # Write the file back with the correct line endings.
    # The `-Encoding utf8` is important for consistency.
    Set-Content -Path $scriptPath -Value $content -Encoding utf8
    Write-Host "entrypoint.sh converted successfully."
} else {
    Write-Host "Warning: entrypoint.sh not found at $scriptPath. Skipping line ending conversion."
}

# ── 1. locate image tag ───────────────────────────────────────────────────────
$imageFile = ".ci/image_${Family}_${Run}.txt"
if (-not (Test-Path $imageFile)) {
    throw " Tag file `$imageFile` not found in repo root."
}
$stableTag = Get-Content $imageFile | Select-Object -First 1
Write-Host "Tag → $stableTag"

# ── 1.5. AUTOMATED: Build the Docker image before using it ──────────────────────
Write-Host "`nSTEP 1.5  docker build (will run automatically if image is outdated)"
# We use the -f flag to specify the Dockerfile.serve file and use the current directory (.) as the build context
docker build -t $stableTag -f Dockerfile.serve .

# ── 2. write env + registry ──────────────────────────────────────────────────
$envFile      = ".env.$Family"
$registryFile = "models_registry.json"

# Create the .env file for docker-compose
@"
STABLE_IMAGE=$stableTag
CANDIDATE_IMAGE=
"@ | Set-Content $envFile

# Safely read and update the JSON registry file
if (-not (Test-Path $registryFile)) { '{}' | Set-Content $registryFile }
$registryContent = Get-Content $registryFile -Raw
$registryObject = $registryContent | ConvertFrom-Json

# Convert the PowerShell object to a regular Hashtable for safer modification
$registryHashtable = @{}
if ($registryObject) {
    $registryObject.PSObject.Properties | ForEach-Object {
        $registryHashtable[$_.Name] = $_.Value
    }
}

# Get or create the entry for the specific model family
$familyEntry = $registryHashtable[$Family]
if (-not $familyEntry) {
    $familyEntry = @{ stable = $null; candidate = $null }
}

# Set the stable and candidate values
$familyEntry.stable = $stableTag
$familyEntry.candidate = ''

# Update the main hashtable
$registryHashtable[$Family] = $familyEntry

# Use our new, safe function to write the JSON file
Write-Host "Safely updating registry file..."
$jsonOutput = ConvertTo-SimpleJson -Hashtable $registryHashtable
Set-Content -Path $registryFile -Value $jsonOutput


# ── 3. pull & start containers (no log-tail) ─────────────────────────────────
$project = "genqa_$Family"

Write-Host "`nSTEP 1  docker compose config"
docker compose --env-file $envFile --project-name $project config

Write-Host "`nSTEP 2  docker compose pull (can take time on first run)"
docker compose --env-file $envFile --project-name $project pull ui stable_model

Write-Host "`nSTEP 3  docker compose up"
docker compose --env-file $envFile --project-name $project up -d ui stable_model

Write-Host "`nSTEP 4  docker compose ps"
docker compose --project-name $project ps

Write-Host "`n✅ Bootstrap finished. UI → http://localhost:5000"
