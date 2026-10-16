"""Build the Colab source bundle for the router-final notebooks.

Usage: python build_kit.py --output-dir <dir>
Writes router-final-source.zip (ml/src, ml/config, ml/pyproject.toml, this folder's code), its
SHA-256 file and an in-zip manifest of per-file hashes. No data, models or credentials.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    paths = [ROOT / "ml/pyproject.toml"]
    paths += sorted((ROOT / "ml/src/llm_router_labeling").rglob("*.py"))
    paths += sorted((ROOT / "ml/config").glob("*.json"))
    paths += sorted(q for q in HERE.iterdir() if q.suffix in {".py", ".json", ".txt"})
    # 07_lock compares RT-08 with the operational R0's recorded 512-token scores (committed results)
    paths += sorted((ROOT / "experiments/results/context-length-qualification-20260925/r0/scores").glob("*-512.parquet"))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    target = a.output_dir / "router-final-source.zip"
    records = {}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            records[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        info = zipfile.ZipInfo("source-bundle-manifest.json", date_time=(2026, 10, 16, 0, 0, 0))
        bundle.writestr(info, json.dumps(records, sort_keys=True, indent=2))
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".sha256").write_text(f"{checksum}  {target.name}\n")
    print(target, checksum, len(records), "files")


if __name__ == "__main__":
    main()
