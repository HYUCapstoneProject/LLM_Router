#!/usr/bin/env bash
set -euo pipefail

task_image_tag="${1:-llm-router-modernbert-training:cu124-v1}"
task_platform="${PLATFORM:-linux/amd64}"
task_revision="${VCS_REF:-$(git describe --always --dirty)}"

docker buildx build \
  --platform "${task_platform}" \
  --file ml/docker/modernbert-training.Dockerfile \
  --build-arg "VCS_REF=${task_revision}" \
  --tag "${task_image_tag}" \
  --load \
  .
