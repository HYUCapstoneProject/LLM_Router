"""Plan or run the four-arm R1/R3 generalization research experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .training import ARMS, plan_four_arms, run_four_arms


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train"))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS))
    args = parser.parse_args(argv)
    kwargs = {
        "dataset_dir": args.dataset_dir,
        "config_path": args.config,
        "runtime_config_path": args.runtime_contract,
    }
    if args.command == "train":
        if args.output_dir is None:
            parser.error("train requires --output-dir")
        result = run_four_arms(
            **kwargs, output_dir=args.output_dir, arms_to_run=args.arms
        )
    else:
        if args.arms is not None:
            parser.error("--arms applies to train; plan always describes all four arms")
        result = plan_four_arms(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
