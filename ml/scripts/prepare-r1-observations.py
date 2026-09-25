"""Prepare an internal, trusted R1 observation export; no HTTP or cloud mutation."""

import argparse
import json
from pathlib import Path

from llm_router_labeling.shared.dataset_io import sha256_file
from llm_router_labeling.training.r1_observations import project_observations
from llm_router_labeling.training.routing_policy import load_routing_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-file-sha256", required=True)
    parser.add_argument("--protected-question-hashes", type=Path, required=True)
    parser.add_argument("--router-model-version", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new immutable output directory")
    policy = load_routing_policy(
        args.policy, expected_file_sha256=args.policy_file_sha256
    )
    protected = json.loads(args.protected_question_hashes.read_text())
    if not isinstance(protected, list) or any(
        not isinstance(x, str) or len(x) != 64 for x in protected
    ):
        parser.error(
            "Protected question hashes must be an explicit JSON list of SHA256 strings"
        )
    rows, manifest = project_observations(
        json.loads(args.observations.read_text()),
        policy,
        protected_question_hashes=protected,
        allowed_router_model_versions=args.router_model_version,
    )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "feedback-training.parquet"
    rows.to_parquet(output, index=False)
    manifest.update(
        input_export_sha256=sha256_file(args.observations),
        protected_hashes_sha256=sha256_file(args.protected_question_hashes),
        policy_file_sha256=sha256_file(args.policy),
        output_sha256=sha256_file(output),
        deployment_ready=False,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
