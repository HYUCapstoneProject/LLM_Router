#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_output="${1:-${task_repo_root}/ml/artifacts/modernbert-colab-smoke-input.tar.gz}"
task_training_path="ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet"

task_inputs=(
  "ml/pyproject.toml"
  "ml/src"
  "ml/config/modernbert_smoke.json"
  "ml/config/modernbert_runtime_cu124_v1.json"
  "ml/requirements-training.lock"
  "ml/requirements-training-cu124.txt"
  "${task_training_path}"
)

for task_input in "${task_inputs[@]}"; do
  if [[ ! -e "${task_repo_root}/${task_input}" ]]; then
    echo "Missing Colab smoke input: ${task_input}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${task_output}")"
tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "${task_output}" \
  -C "${task_repo_root}" \
  "${task_inputs[@]}"
echo "Created ${task_output}"
