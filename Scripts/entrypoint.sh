#!/usr/bin/env bash

set -e

# The OLLAMA_MODEL_NAME is passed as an argument to this script
OLLAMA_MODEL_NAME="$1"

if [ -z "$OLLAMA_MODEL_NAME" ]; then
    echo "Error: No OLLAMA_MODEL_NAME provided."
    exit 1
fi

echo "Starting temporary Ollama server..."
ollama serve &
SERVER_PID=$!


echo "Running PowerShell model setup..."

# Call the PowerShell script, which waits for the server and creates the model.
pwsh -File "/usr/local/bin/entrypoint.ps1" -OLLAMA_MODEL_NAME "$OLLAMA_MODEL_NAME"

echo "Setup complete. Starting Ollama server in foreground."
echo "Model setup complete. Stopping temporary server..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true

# Now, we start the primary, long-running process for the container.
# This ensures the container stays alive after the setup is finished.
echo "Starting Ollama server in foreground."
# Now start the long-running process to keep the container alive.
exec ollama serve








