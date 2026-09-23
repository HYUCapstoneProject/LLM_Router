#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_path="${1:-${repo_root}/ml/artifacts/routing-objective-colab-code.tar.gz}"
staging_dir="$(mktemp -d)"
trap 'rm -rf "${staging_dir}"' EXIT

mkdir -p "${staging_dir}/LLM_Router/ml"
cp -R "${repo_root}/ml/src" "${staging_dir}/LLM_Router/ml/src"
cp -R "${repo_root}/ml/config" "${staging_dir}/LLM_Router/ml/config"
cp "${repo_root}/ml/pyproject.toml" "${staging_dir}/LLM_Router/ml/pyproject.toml"
cp "${repo_root}/ml/requirements-lock.txt" "${staging_dir}/LLM_Router/ml/requirements-lock.txt"
cp "${repo_root}/ml/requirements-comparison.lock" "${staging_dir}/LLM_Router/ml/requirements-comparison.lock"
find "${staging_dir}/LLM_Router" -type d -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$(dirname "${output_path}")"
tar -C "${staging_dir}" -czf "${output_path}" LLM_Router
printf '%s\n' "${output_path}"
