"""EV-10 01_embed: one BGE-large-en-v1.5 embedding per question for every role (and RouterArena).

Reads the 00_prepare outputs (train/dev/test.parquet) plus the RouterArena questions and writes
float16 L2-normalized embeddings with a row-aligned id table. The model revision is pinned; the
manifest records input and output SHA-256 so later notebooks can verify what they read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "BAAI/bge-large-en-v1.5"
REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items(prepare_dir: Path, arena_path: Path | None) -> pd.DataFrame:
    frames = []
    for role in ("train", "dev", "test"):
        part = pd.read_parquet(prepare_dir / f"{role}.parquet", columns=["sample_id", "source_dataset", "model_input_text"])
        frames.append(part.assign(role=role))
    if arena_path is not None:
        arena = pd.read_parquet(arena_path)
        frames.append(pd.DataFrame({"sample_id": "routerarena:" + arena["Global Index"].astype(str),
                                    "source_dataset": "routerarena", "model_input_text": arena["Question"].astype(str),
                                    "role": "diagnostic"}))
    items = pd.concat(frames, ignore_index=True)
    if items.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique across roles")
    return items


def embed(texts: list[str], batch_size: int, device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=REVISION, device=device)
    model.max_seq_length = 512
    if model.device.type == "cuda":
        model.half()
    out = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    return out.astype(np.float16)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-dir", required=True)
    p.add_argument("--routerarena", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke test only")
    a = p.parse_args(argv)
    prep, out = Path(a.prepare_dir), Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory must be new and empty")
    prep_manifest = json.loads((prep / "manifest.json").read_text())
    for name, spec in prep_manifest["outputs"].items():
        if sha256_file(prep / name) != spec["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    items = load_items(prep, Path(a.routerarena) if a.routerarena else None)
    if a.limit:
        items = items.groupby("role", group_keys=False).head(a.limit).reset_index(drop=True)
    vectors = embed(items["model_input_text"].tolist(), a.batch_size, a.device)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    items[["sample_id", "source_dataset", "role"]].to_parquet(out / "ids.parquet", index=False)
    manifest = {"model": MODEL, "revision": REVISION, "max_seq_length": 512, "normalized": True, "dtype": "float16",
                "rows": int(len(items)), "dim": int(vectors.shape[1]), "limit": a.limit,
                "prepare_manifest_sha256": sha256_file(prep / "manifest.json"),
                "outputs": {n: sha256_file(out / n) for n in ("embeddings.npy", "ids.parquet")},
                "rows_by_role": items.groupby(["role", "source_dataset"]).size().unstack(fill_value=0).to_dict("index")}
    if a.routerarena:
        manifest["routerarena_sha256"] = sha256_file(Path(a.routerarena))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ("rows", "dim", "rows_by_role")}, indent=1))


if __name__ == "__main__":
    main()
