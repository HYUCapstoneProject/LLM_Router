"""RF-05 planning: how much of the new-generation gain does a frozen A boundary lose? (dev only, no test)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(sys.argv[1]); RUN = Path(sys.argv[2])
sys.path.insert(0, str(REPO / "ml/notebooks/router_final"))
import lock as L

data = L.load_outcomes(RUN)
excluded, _ = L.evaluation_exclusions(REPO)
dev = data["dev"]; dev = dev[~dev.sample_id.isin(excluded)]
cols = [f"p_{t}" for t in L.TIERS]
dev = dev[dev[cols].notna().all(axis=1)].reset_index(drop=True)
BODIES = {"r1_group_a": "scores-modernbert/r1_group_a/scores.parquet", "knn_a": "common/knn_a/n_neighbors=50/scores.parquet",
          "r1_group": "scores-modernbert/r1_group/scores.parquet", "knn": "common/knn/n_neighbors=40/scores.parquet"}
NEW, A = ("llmrouterbench", "sprout"), ("routerbench", "xroutebench")
sc = [f"score_{t}" for t in L.TIERS]
rows = []
for body, rel in BODIES.items():
    s = pd.read_parquet(RUN / rel).drop_duplicates("sample_id").set_index("sample_id")
    a = dev[dev.source_dataset.isin(A)]
    Xa, Ya = s.loc[a.sample_id, sc].to_numpy(float), a[cols].to_numpy(float)
    rho_a = L.estimate_rho(Ya)
    for src in NEW:
        d = dev[dev.source_dataset.eq(src)]
        X, Y = s.loc[d.sample_id, sc].to_numpy(float), d[cols].to_numpy(float)
        targets, rho = L.cost_targets(Y), L.estimate_rho(Y)
        variants = {"frozen_A_boundary": (L.fit_predict(Xa, Ya, X), rho_a),
                    "new_source_boundary_crossfit": (L.cross_fit(X, Y, L.fold_of(d.sample_id)), rho)}
        for name, (Phat, r) in variants.items():
            t = L.lambda_table(Y, Phat, r, targets)
            esc = pd.DataFrame(t["escalation"])
            rows.append({"body": body, "source": src, "variant": name, "rows": len(d), "mean_gain": esc.dev_gain.mean(),
                         **{f"gain_k{k:.2f}": esc.loc[np.isclose(esc.kappa, k), "dev_gain"].mean() for k in (0.0, 0.3, 0.5, 1.0)},
                         "cal_err_low": float(np.mean(Phat[:, 0]) - np.mean(Y[:, 0])),
                         "cal_err_med": float(np.mean(Phat[:, 1]) - np.mean(Y[:, 1])),
                         "cal_err_high": float(np.mean(Phat[:, 2]) - np.mean(Y[:, 2]))})
out = pd.DataFrame(rows)
out.to_csv(Path(sys.argv[3]), index=False)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
