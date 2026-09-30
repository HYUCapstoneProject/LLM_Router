"""Prepare the frozen external evaluation sets (evaluation only; no model calls).

Example:
  python ml/scripts/prepare-external-evaluation.py \
    --config ml/config/external_evaluation_v1.json --repo-root . \
    --llmrouterbench-root ml/artifacts/external-eval-raw/llmrouterbench/bench-release \
    --sprout-dir ml/artifacts/external-eval-raw/sprout \
    --tokenizer ml/artifacts/r0-release-20260927/approved \
    --output-dir ml/artifacts/external-evaluation-v1
"""
import argparse
import json
from pathlib import Path

from llm_router_labeling.evaluation.external_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--llmrouterbench-root", type=Path, required=True)
    parser.add_argument("--sprout-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare(args.config, args.repo_root, args.llmrouterbench_root, args.sprout_dir, args.output_dir, args.tokenizer)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
