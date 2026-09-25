"""Apply the G1 calibration -> selection lock -> holdout protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_holdout, infer_holdout, lock_selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock", "infer", "evaluate"))
    for name in ("dataset-dir", "runs-dir", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    kwargs = {
        "dataset_dir": args.dataset_dir,
        "runs_dir": args.runs_dir,
        "config_path": args.config,
        "output_dir": args.output_dir,
    }
    if args.command == "lock":
        result = lock_selection(**kwargs)
    else:
        if args.lock_dir is None:
            parser.error("infer/evaluate requires --lock-dir")
        kwargs["lock_dir"] = args.lock_dir
        if args.command == "infer":
            result = infer_holdout(**kwargs, device=args.device)
        else:
            if args.scores_dir is None:
                parser.error("evaluate requires --scores-dir")
            result = evaluate_holdout(**kwargs, scores_dir=args.scores_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
