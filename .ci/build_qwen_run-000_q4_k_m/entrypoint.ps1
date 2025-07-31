# Scripts/entrypoint.ps1
param (
    [string]$OLLAMA_MODEL_NAME
)

# Set error action preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "Starting Ollama server in background..."
Start-Process ollama -ArgumentList "serve" -NoNewWindow -PassThru | Out-Null

Write-Host "Waiting for Ollama server to start..."
$maxAttempts = 30
$attempt = 0
while ($true) {
    try {
        $response = Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get -TimeoutSec 5
        Write-Host "Ollama server is up."
        break
    } catch {
        $attempt++
        if ($attempt -ge $maxAttempts) {
            Write-Error "Ollama server did not start within the expected time."
            exit 1
        }
        Write-Host "Attempt $attempt/$maxAttempts: Ollama server not ready. Waiting..."
        Start-Sleep -Seconds 3
    }
}

Write-Host "Creating model ${OLLAMA_MODEL_NAME}..."
try {
    $ollamaCreateArgs = @("create", $OLLAMA_MODEL_NAME, "-f", "/workspace/Modelfile")
    & ollama $ollamaCreateArgs
} catch {
    Write-Error "Failed to create Ollama model: $($_.Exception.Message)"
    exit 1
}

Write-Host "Removing Modelfile..."
Remove-Item -Path "/workspace/Modelfile" -Force -ErrorAction SilentlyContinue

Write-Host "Model creation complete. Ollama server is running."

# The CMD in Dockerfile will typically take over as PID 1
# after this script completes.