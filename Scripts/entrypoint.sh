#!/usr/bin/env bash

set -e

# The OLLAMA_MODEL_NAME is passed as an argument to this script
OLLAMA_MODEL_NAME="$1"

if [ -z "$OLLAMA_MODEL_NAME" ]; then
    echo "Error: No OLLAMA_MODEL_NAME provided."
    exit 1
fi

echo "Starting model setup using PowerShell script..."

# This is where the switch happens. We call the PowerShell script.
# The `pwsh` command is now available because it's installed in the Dockerfile.
# We pass the OLLAMA_MODEL_NAME as an argument to the PowerShell script.
# The -File parameter ensures the script runs as a file.
pwsh -File "/usr/local/bin/entrypoint.ps1" -OLLAMA_MODEL_NAME "$OLLAMA_MODEL_NAME"

echo "Setup complete. Starting Ollama server in foreground."

# Now, we start the primary, long-running process for the container.
# This ensures the container stays alive after the setup is finished.
exec ollama serve
