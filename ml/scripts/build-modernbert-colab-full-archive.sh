#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_output="${1:-${task_repo_root}/ml/artifacts/modernbert-colab-full-input.tar.gz}"
task_training_path="ml/artifacts/routerbench-training/withmartian-routerbench-2024-03-27__minimum-sufficient-tier-v1__routerbench-training-preprocess-v1/training.parquet"
task_routerarena_dir="ml/artifacts/routerarena-evaluation-full"

task_inputs=(
  "ml/pyproject.toml"
  "ml/src"
  "ml/config/modernbert_t4_full.json"
  "ml/config/modernbert_runtime_cu124_v1.json"
  "ml/config/routerarena_evaluation.json"
  "ml/requirements-training.lock"
  "ml/requirements-training-cu124.txt"
  "${task_training_path}"
  "${task_routerarena_dir}/evaluation.parquet"
  "${task_routerarena_dir}/metadata.parquet"
)

for task_input in "${task_inputs[@]}"; do
  if [[ ! -e "${task_repo_root}/${task_input}" ]]; then
    echo "Missing Colab full-training input: ${task_input}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${task_output}")"
tar \
  --exclude='*/__pycache__' \
  --exclude='*.egg-info' \
  --exclude='*.pyc' \
  -czf "${task_output}" \
  -C "${task_repo_root}" \
  "${task_inputs[@]}"
echo "Created ${task_output}"
