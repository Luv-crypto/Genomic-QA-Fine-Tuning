#!/usr/bin/env bash
set -euo pipefail

FAMILY=$1
RUN=$2
MODE=${3:-q4_k_m}
IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}/model-lab:${FAMILY}_${RUN}"

GGUF="models/gguf/${FAMILY}/${RUN}_${MODE}.gguf"
if [[ ! -f "$GGUF" ]]; then
  GGUF="models/gguf/${FAMILY}/${RUN}_full.gguf"
fi

CTX=".ci/build_${FAMILY}_${RUN}"
mkdir -p "$CTX"
cp "$GGUF" "$CTX/model.gguf"
cp "models/modelfiles/${FAMILY}/${RUN}.Modelfile" "$CTX/Modelfile"

docker build -f Dockerfile.serve -t "$IMAGE" "$CTX"
docker push "$IMAGE"
echo "$IMAGE" > ".ci/image_${FAMILY}_${RUN}.txt"
