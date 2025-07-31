# scripts/build_image_ci.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$Family, # e.g., 'qwen'

    [Parameter(Mandatory=$true)]
    [string]$Run, # e.g., 'run-000'

    [Parameter(Mandatory=$false)]
    [string]$Mode = 'q4_k_m' # e.g., 'q4_k_m' for quantized, 'full' for non-quantized
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# For local builds, GITHUB_REPOSITORY env var won't be set. Use a placeholder.
$githubRepo = if ($env:GITHUB_REPOSITORY) { $env:GITHUB_REPOSITORY.ToLower() } else { "local-user/genomic-qa-fine-tuning" }

# --- Determine Generic Model Name and Version ---

# Base model name should now include the quantization mode for distinct versioning
$baseModelName = "${Family}-${Mode}"

# Directory to store version tracking files
$versionDir = ".ci/versions"
if (-not (Test-Path $versionDir)) {
    New-Item $versionDir -ItemType Directory | Out-Null
}

# Version file path for this specific base model + mode combination
$versionFilePath = Join-Path $versionDir "$($baseModelName)_version.txt"

# Read current version, or start from 0 if not found
$currentVersion = 0
if (Test-Path $versionFilePath) {
    try {
        $currentVersion = [int](Get-Content $versionFilePath)
    } catch {
        Write-Warning "Could not read version from $versionFilePath. Starting with version 0."
        $currentVersion = 0
    }
}

# Increment version for the new build
$newVersion = $currentVersion + 1

# Store the new version
$newVersion | Set-Content -Path $versionFilePath -Force

# Construct the full generic model name for the image tag and inside the container
$genericModelName = "${baseModelName}_v${newVersion}"

Write-Host "Assigned generic model name: $genericModelName"

$image = "lovefadia/llm-dev:$genericModelName"

# --- End Generic Model Naming Logic ---


# Determine the correct GGUF file path, with a fallback
$ggufPath = "models/gguf/$Family/${Run}_${Mode}.gguf"
if (-not (Test-Path $ggufPath)) {
    $fallbackGgufPath = "models/gguf/$Family/${Run}_full.gguf"
    if (Test-Path $fallbackGgufPath) {
        $ggufPath = $fallbackGgufPath
        Write-Warning "Specific GGUF '$Mode' not found, falling back to '$fallbackGgufPath'."
    } else {
        throw "Could not find GGUF file at $ggufPath or the fallback path: $fallbackGgufPath."
    }
}

# Create a clean build context directory
$ctx = ".ci/build_${Family}_${Run}_${Mode}"
if (Test-Path $ctx) {
    Remove-Item $ctx -Recurse -Force
}
New-Item $ctx -ItemType Directory | Out-Null

# --- IMPORTANT CHANGES HERE ---

# Copy Dockerfile.serve to the build context
# Assuming Dockerfile.serve is in the project root
$sourceDockerfile = "Dockerfile.serve"
if (-not (Test-Path $sourceDockerfile)) {
    throw "Source Dockerfile.serve not found at: $sourceDockerfile. Please ensure it's in the project root."
}
Copy-Item -Path $sourceDockerfile -Destination (Join-Path $ctx "Dockerfile.serve") -Force

# Copy the PowerShell entrypoint script to the build context
# Assuming entrypoint.ps1 is now in 'Scripts/'
$sourceEntrypointScript = "Scripts/entrypoint.ps1"
if (-not (Test-Path $sourceEntrypointScript)) {
    throw "Source for entrypoint.ps1 not found at: $sourceEntrypointScript. Please ensure it's in the 'Scripts' folder."
}
Copy-Item -Path $sourceEntrypointScript -Destination (Join-Path $ctx "entrypoint.ps1") -Force # Copy to the root of context

# --- END IMPORTANT CHANGES ---


# Copy the GGUF file into the build context, renaming it to 'model.gguf'
Write-Host "Copying GGUF from '$ggufPath' to '$(Join-Path $ctx "model.gguf")'"
Copy-Item $ggufPath -Destination (Join-Path $ctx "model.gguf") -Force

# Copy the corresponding Modelfile into the build context, renaming it to "Modelfile"
$modelfilePath = "models/modelfiles/$Family/${Run}_${Mode}.Modelfile"
if (-not (Test-Path $modelfilePath)) {
    $fallbackModelfilePath = "models/modelfiles/$Family/${Run}_full.Modelfile"
    if (Test-Path $fallbackModelfilePath) {
        $modelfilePath = $fallbackModelfilePath
        Write-Warning "Specific Modelfile '$Mode' not found, falling back to '$fallbackModelfilePath'."
    } else {
        throw "Could not find Modelfile at $modelfilePath or the fallback path: $fallbackModelfilePath."
    }
}
Write-Host "Copying Modelfile from '$modelfilePath' to '$(Join-Path $ctx "Modelfile")'"
Copy-Item $modelfilePath -Destination (Join-Path $ctx "Modelfile") -Force


Write-Host "Building Docker image: $image"
Write-Host "Using GGUF: model.gguf (original: $(Split-Path $ggufPath -Leaf))"
Write-Host "Using Modelfile: Modelfile (original: $(Split-Path $modelfilePath -Leaf))"
Write-Host "Model will be created in Ollama as: $genericModelName"


# Build the Docker image from the context directory, passing genericModelName as a build-arg
docker build -f (Join-Path $ctx "Dockerfile.serve") `
             -t "$image" `
             --build-arg OLLAMA_MODEL_NAME="$genericModelName" `
             "$ctx"

Write-Host "Pushing image to Docker Hub registry..."
docker push "$image"

$outFile = ".ci/image_${Family}_${Run}.txt" # Consider including mode here too for specific image tracking
$image | Set-Content -Path $outFile

Write-Host "The Docker image built and tag saved to $outFile"

# Clean up the build context directory
# Remove-Item $ctx -Recurse -Force