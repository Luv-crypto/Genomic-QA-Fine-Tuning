#!/usr/bin/env bash
set -euo pipefail
RUN=$1
IMAGE="ghcr.io/${GITHUB_REPOSITORY,,}/model-lab:${RUN}"

docker build --build-arg GGUF=models/gguf/${RUN}_full.gguf \
             -f Dockerfile.serve \
             -t $IMAGE .
docker push $IMAGE
echo $IMAGE > .ci/image_${RUN}.txt
