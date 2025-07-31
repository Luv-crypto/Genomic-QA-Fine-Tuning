#!/usr/bin/env bash
set -euo pipefail

FAMILY=$1
RUN=$2
MODE=${3:-q4_k_m}
IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}/model-lab:${FAMILY}_${RUN}"

# Determine the correct GGUF file path
GGUF_PATH="models/gguf/${FAMILY}/${RUN}_${MODE}.gguf"
if [[ ! -f "$GGUF_PATH" ]]; then
  GGUF_PATH="models/gguf/${FAMILY}/${RUN}_full.gguf"
fi

# Get just the filename of the GGUF model
GGUF_FILENAME=$(basename "$GGUF_PATH")

# Create a clean build context directory
CTX=".ci/build_${FAMILY}_${RUN}"
rm -rf "$CTX" # Ensure the directory is clean before use
mkdir -p "$CTX"

# Copy the GGUF file *with its original name* into the build context
cp "$GGUF_PATH" "$CTX/$GGUF_FILENAME"

# Copy the corresponding Modelfile into the build context
# Note: The Modelfile name in dvc.yaml should match this.
cp "models/modelfiles/${FAMILY}/${RUN}_${MODE}.Modelfile" "$CTX/Modelfile"

echo "Building Docker image: $IMAGE"
echo "Using GGUF: $GGUF_FILENAME"
echo "Using Modelfile: $CTX/Modelfile"

# Build the Docker image from the context directory
docker build -f Dockerfile.serve -t "$IMAGE" "$CTX"

# Push the image and save its tag
docker push "$IMAGE"
echo "$IMAGE" > ".ci/image_${FAMILY}_${RUN}.txt"
