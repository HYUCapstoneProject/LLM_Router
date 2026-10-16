"""pgvector replay and per-stage latency for the cache experiment. Local only.

The SQL is read from backend CacheService.java and the cache_entry DDL from
infra/postgres/init/01-schema.sql at run time, so this module cannot drift from
the operational statements. PostgreSQL runs in a throwaway local Docker
container; nothing connects to the operational database.
"""
from __future__ import annotations

import re
import resource
import secrets
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CACHE_SERVICE = REPO / "backend/src/main/java/com/llmrouter/service/CacheService.java"
SCHEMA = REPO / "infra/postgres/init/01-schema.sql"
CONTAINER = "llm-router-cache-experiment-pg"


def operational_sql():
    """SELECT_SQL, SELECT_EXACT_SQL, INSERT_SQL and recordHit from CacheService.java."""
    source = CACHE_SERVICE.read_text()
    blocks = dict(re.findall(r'static final String (\w+) = """(.*?)""";', source, flags=re.S))
    hit = re.search(r'jdbc\.update\("(UPDATE cache_entry SET hit_count = hit_count \+ 1[^"]*)"', source)
    missing = {"SELECT_SQL", "SELECT_EXACT_SQL", "INSERT_SQL"} - set(blocks)
    if missing or not hit:
        raise RuntimeError(f"CacheService.java changed; cannot find {missing or 'recordHit'}")
    to_psycopg = lambda sql: " ".join(sql.split()).replace("%", "%%").replace("?", "%s")
    return {name: to_psycopg(blocks[name]) for name in ("SELECT_SQL", "SELECT_EXACT_SQL", "INSERT_SQL")} | {
        "RECORD_HIT": to_psycopg(hit.group(1))}


def cache_entry_ddl():
    text = SCHEMA.read_text()
    table = re.search(r"CREATE TABLE cache_entry \(.*?\n\);", text, flags=re.S)
    indexes = re.findall(r"CREATE INDEX idx_cache_entry_\w+\s+ON cache_entry[^;]*;", text, flags=re.S)
    if not table or len(indexes) < 2:
        raise RuntimeError("01-schema.sql changed; cache_entry DDL not found")
    strip = lambda sql: re.sub(r"--[^\n]*", "", sql)
    return ["CREATE EXTENSION IF NOT EXISTS vector;", "DROP TABLE IF EXISTS cache_entry CASCADE;", strip(table.group(0))] + [
        strip(i) for i in indexes]


# ------------------------------------------------------------------ docker --

def start_postgres(config):
    lat = config["latency"]
    password = secrets.token_urlsafe(16)
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--cpus", str(lat["docker_cpus"]),
                    "-e", f"POSTGRES_PASSWORD={password}", "-e", "POSTGRES_DB=cache_experiment",
                    "-p", f"127.0.0.1:{lat['pgvector_port']}:5432", lat["pgvector_image"]], check=True, capture_output=True)
    dsn = f"postgresql://postgres:{password}@127.0.0.1:{lat['pgvector_port']}/cache_experiment"
    import psycopg
    for _ in range(60):
        try:
            with psycopg.connect(dsn, connect_timeout=2):
                return dsn
        except psycopg.OperationalError:
            time.sleep(1)
    raise RuntimeError("Local pgvector container did not become ready")


def stop_postgres():
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def container_info(config):
    image = config["latency"]["pgvector_image"]
    out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True)
    return {"image": image, "image_id": out.stdout.strip() or None, "cpus": config["latency"]["docker_cpus"]}


# --------------------------------------------------------------- pg cache ---

def _literal(vector):
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PgCache:
    """MemoryCache interface backed by the operational SQL on real pgvector."""

    def __init__(self, dsn, candidate_limit, ef_search=None):
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.sql = operational_sql()
        self.limit = candidate_limit
        with self.conn.cursor() as cur:
            for statement in cache_entry_ddl():
                cur.execute(statement)
            if ef_search:
                cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")
            cur.execute("SHOW hnsw.ef_search")
            self.ef_search = cur.fetchone()[0]
        self.rows, self.evictions = {}, 0

    def close(self):
        self.conn.close()

    def __len__(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM cache_entry")
            return int(cur.fetchone()[0])

    def exact(self, key):
        # SELECT_EXACT_SQL binds the hash that Java computes; ours is pinned to it.
        with self.conn.cursor() as cur:
            cur.execute(self.sql["SELECT_EXACT_SQL"], (key,))
            row = cur.fetchone()
        return None if row is None else int(row[0])

    def candidates(self, vector, k):
        if k != self.limit:
            raise ValueError("PgCache is created for one candidate limit")
        literal = _literal(vector)
        with self.conn.cursor() as cur:
            cur.execute(self.sql["SELECT_SQL"], (literal, literal, k))
            return [(int(r[0]), float(r[6])) for r in cur.fetchall()]

    def touch(self, entry_id):
        with self.conn.cursor() as cur:
            cur.execute(self.sql["RECORD_HIT"], (entry_id,))

    def insert(self, row, key, vector, label, threshold):
        # replay() passes the pool row; question/response text come from bind_pool().
        with self.conn.cursor() as cur:
            cur.execute(self.sql["INSERT_SQL"], (self.texts[row], key, _literal(vector), self.answers[row], label,
                                                 self.models[row], "bge-large-en-v1.5", threshold))
            entry_id = int(cur.fetchone()[0])
        self.rows[entry_id] = int(row)
        return entry_id

    def bind_pool(self, pool):
        self.texts = pool.request_text.tolist()
        self.answers = pool.routed_answer.fillna("").tolist()
        self.models = [f"{s}:{t}" for s, t in zip(pool.source_dataset, pool.routed_tier)]
        return self


def compare_decisions(memory_log, pg_log):
    """Per-request agreement between in-memory exact search and pgvector HNSW."""
    a = memory_log[["position", "decision", "source_sample_id"]].rename(columns=lambda c: c if c == "position" else f"memory_{c}")
    b = pg_log[["position", "decision", "source_sample_id"]].rename(columns=lambda c: c if c == "position" else f"pgvector_{c}")
    merged = a.merge(b, on="position", validate="one_to_one")
    same = (merged.memory_decision == merged.pgvector_decision) & (
        merged.memory_source_sample_id.fillna("") == merged.pgvector_source_sample_id.fillna(""))
    return {"requests": int(len(merged)), "agreement": float(same.mean()), "disagreements": int((~same).sum())}, merged[~same]


# ---------------------------------------------------------------- timing ----

def timed(fn, inputs, warmup, repeats):
    for x in inputs[:warmup]:
        fn(x)
    samples = []
    for i in range(repeats):
        x = inputs[(warmup + i) % len(inputs)]
        start = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(samples)


def summarize_ms(stage, samples, **extra):
    return {"stage": stage, "n": int(len(samples)), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "mean_ms": float(samples.mean()),
            "max_ms": float(samples.max()), "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
            **extra}


def measure_models(texts, embedder, reranker, config, router=None):
    import torch
    lat = config["latency"]
    torch.set_num_threads(lat["torch_threads"])
    rows = []
    rows.append(summarize_ms("embed", timed(embedder.embed, texts, lat["warmup"], lat["requests"]),
                             threads=torch.get_num_threads(), device=str(embedder.device)))
    batch = lat["rerank_batch"]
    pairs = [(texts[i], [texts[(i + j + 1) % len(texts)] for j in range(batch)]) for i in range(len(texts))]
    rows.append(summarize_ms(f"rerank_batch{batch}", timed(lambda p: reranker.score(*p), pairs, lat["warmup"], lat["requests"]),
                             threads=torch.get_num_threads(), device=str(reranker.device)))
    if router is not None:
        rows.append(summarize_ms("route", timed(router, texts, lat["warmup"], lat["requests"]),
                                 threads=torch.get_num_threads()))
    return pd.DataFrame(rows)


def measure_pgvector(dsn, pool, vectors, config, sizes=None):
    """Exact-key and semantic lookup latency on caches of several sizes."""
    lat = config["latency"]
    sizes = sizes or lat["cache_sizes"]
    k = config["policies"]["C3"]["candidate_limit"]
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    rows = []
    for size in sizes:
        if size > len(pool):
            raise ValueError(f"cache size {size} exceeds the pool")
        cache = PgCache(dsn, k, lat["hnsw_ef_search"]).bind_pool(pool)
        members = rng.permutation(len(pool))[:size]
        for row in members:
            cache.insert(int(row), pool.key_hash.iat[row], vectors[row], pool.routed_label.iat[row], 0.80)
        with cache.conn.cursor() as cur:
            cur.execute("ANALYZE cache_entry")
        probes = rng.permutation(len(pool))[: lat["requests"] + lat["warmup"]]
        keys = [pool.key_hash.iat[i] for i in probes]
        vecs = [vectors[i] for i in probes]
        rows.append(summarize_ms("pg_exact_lookup", timed(cache.exact, keys, lat["warmup"], lat["requests"]),
                                 cache_size=size, ef_search=cache.ef_search))
        rows.append(summarize_ms("pg_vector_search", timed(lambda v: cache.candidates(v, k), vecs, lat["warmup"], lat["requests"]),
                                 cache_size=size, ef_search=cache.ef_search))
        cache.close()
    return pd.DataFrame(rows)


def hnsw_recall(dsn, pool, vectors, config, size, queries=500):
    """Share of queries whose HNSW top-k equals exact top-k (before thresholds)."""
    k = config["policies"]["C3"]["candidate_limit"]
    rng = np.random.default_rng([config["bootstrap"]["seed"], size])
    cache = PgCache(dsn, k, config["latency"]["hnsw_ef_search"]).bind_pool(pool)
    members = rng.permutation(len(pool))[:size]
    ids = []
    for row in members:
        ids.append(cache.insert(int(row), pool.key_hash.iat[row], vectors[row], pool.routed_label.iat[row], -1.0))
    matrix = vectors[members]
    same = []
    for q in rng.permutation(len(pool))[:queries]:
        with cache.conn.cursor() as cur:
            cur.execute("SELECT id FROM cache_entry ORDER BY embedding <=> %s::vector LIMIT %s", (_literal(vectors[q]), k))
            hnsw = [cache.rows[int(r[0])] for r in cur.fetchall()]
        exact = [int(members[i]) for i in np.argsort(-(matrix @ vectors[q]), kind="stable")[:k]]
        same.append(set(hnsw) == set(exact))
    cache.close()
    return {"cache_size": size, "queries": queries, "topk_set_recall": float(np.mean(same)), "ef_search": cache.ef_search}
