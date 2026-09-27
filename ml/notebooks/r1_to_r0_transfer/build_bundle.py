"""Build a source-only Colab ZIP from this worktree, including uncommitted files.

Usage: python build_bundle.py --output /path/to/transfer-source.zip
No models, data, credentials, Git internals or cloud transfers are included.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [ROOT / "ml" / "pyproject.toml", ROOT / "docs/ml/README.md",
             ROOT / "docs/ml/R1_TO_R0_TRANSFER_EXPERIMENT.md"]
    paths += list((ROOT / "ml/src/llm_router_labeling").rglob("*.py"))
    paths += list((ROOT / "ml/config").glob("*.json"))
    public_files = ["README.md", "experiment.py", "experiment_config.json",
                    "source_pins.example.json", "requirements-colab.txt", "build_bundle.py",
                    "prepare_local_data.py", "prepare_models.py",
                    "00_prepare.ipynb", "01_r0_continued.ipynb", "02_r1_to_r0.ipynb",
                    "03_r1_frozen.ipynb", "04_compare_results.ipynb"]
    paths += [Path(__file__).parent / name for name in public_files]
    records = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(paths)):
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
            records[name] = hashlib.sha256(data).hexdigest()
        info = zipfile.ZipInfo("source-bundle-manifest.json", date_time=(2026, 9, 27, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(info, json.dumps(records, sort_keys=True, indent=2))
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".sha256").write_text(checksum + "  " + args.output.name + "\n")
    print(json.dumps({"file": str(args.output), "sha256": checksum, "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
