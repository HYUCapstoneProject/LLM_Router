"""Where R0-family routing stops: ceilings on the sealed test (v1 outcomes).

Reads only existing artifacts (no training, no new inference):
  * sealed test 7,202 rows with recorded outcomes (the F4/F5 kit input),
  * R0/R1 8192-token test logits from the context-length qualification,
  * F0/F4/F5 final predictions from the personal-Drive runs,
  * initial train rows and raw RouterBench outcomes for train-only calibration.

Every curve is a test-set *description* of an achievable envelope, not a policy
chosen on test. Nothing here selects a model, a threshold or a lambda.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression

TIERS = ("low", "medium", "high")
LABELS = ("easy", "medium", "hard")
LAMBDAS = np.concatenate([[0.0], np.logspace(-4, 2, 301)])
R1_LOCKED = {"routerbench": 0.1, "xroutebench": 0.03}  # sealed-test policy, chosen on D_dev
BUDGETS = (0.1, 0.3, 0.5)
ARM_MODELS = {"mistralai/mistral-7b-chat", "zero-one-ai/Yi-34B-Chat", "gpt-4-1106-preview"}


def arrays(frame):
    return (frame[[f"quality_{t}" for t in TIERS]].to_numpy(float),
            frame[[f"success_{t}" for t in TIERS]].to_numpy(float),
            frame[[f"actual_cost_{t}" for t in TIERS]].to_numpy(float),
            frame[[f"decision_cost_{t}" for t in TIERS]].iloc[0].to_numpy(float))


def mixing_frontier(points, x):
    best = max([q for c, q in points if c <= x + 1e-12], default=np.nan)
    for (c1, q1), (c2, q2) in itertools.combinations(sorted(points), 2):
        if c1 <= x <= c2 and c2 > c1:
            best = max(best, q1 + (q2 - q1) * (x - c1) / (c2 - c1))
    return best


class Source:
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.Q, self.S, self.C, self.dc = arrays(self.frame)
        self.lo, self.hi = self.C[:, 0].mean(), self.C[:, 2].mean()
        self.points = [(self.rel(self.C[:, t].mean()), self.Q[:, t].mean()) for t in range(3)]

    def rel(self, cost):
        return (cost - self.lo) / (self.hi - self.lo)

    def first_try(self, tier):
        i = np.arange(len(tier))
        return float(self.Q[i, tier].mean()), float(self.rel(self.C[i, tier].mean()))

    def cascade(self, tier):
        """Start at `tier`, escalate on recorded failure; charge every call made."""
        n = len(tier)
        solved, cost, depth = np.zeros(n, bool), np.zeros(n), np.zeros(n)
        for t in range(3):
            active = ~solved & (tier <= t)
            cost[active] += self.C[active, t]
            depth[active] += 1
            solved |= active & (self.S[:, t] == 1)
        i = np.arange(n)
        return {"first_try_success": float(self.S[i, tier].mean()), "final_resolution": float(solved.mean()),
                "mean_calls": float(depth.mean()), "cascade_cost_vs_always_high": float(cost.mean() / self.hi)}

    def curve(self, scores):
        """Score - lambda * decision cost over the lambda grid -> (relative cost, first-try quality)."""
        rows = []
        for lam in LAMBDAS:
            q, c = self.first_try(np.argmax(scores - lam * self.dc, 1))
            rows.append((c, q))
        return sorted(set(rows))

    def oracle_curve(self):
        rel = self.rel(self.C)
        return sorted({self.first_try(np.argmax(self.Q - lam * rel, 1))[::-1] for lam in LAMBDAS})


def summarize(name, source, curve):
    pts = pd.DataFrame(curve, columns=["relative_cost", "quality"])
    gain = pts.quality - pts.relative_cost.map(lambda x: mixing_frontier(source.points, x))
    row = {"method": name, "max_gain_over_mixing": float(gain.max())}
    for b in BUDGETS:
        within = pts[pts.relative_cost <= b]
        row[f"best_quality_cost_le_{b}"] = float(within.quality.max()) if len(within) else np.nan
    return row


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def r0_tau(probs, tau):
    """Cheapest tier t with P(difficulty <= t) >= tau."""
    cum = np.cumsum(probs, 1)
    return np.where(cum[:, 0] >= tau, 0, np.where(cum[:, 1] >= tau, 1, 2))


def label_of(rate):
    return np.where(rate >= .64, 0, np.where(rate >= .27, 1, 2))


def label_noise_ceiling(train, test):
    """Bayes accuracy against the observed 3-class label for a predictor that knows each question's
    true solve rate p, with p ~ Beta(a, b) fitted on train and k ~ Binomial(n, p) models solving it.
    Exchangeable models are assumed; heterogeneous model abilities would make labels a little steadier."""
    k_tr = train.accepted_model_count.to_numpy(int)
    n_tr = train.valid_model_count.to_numpy(int)

    def nll(log_ab):
        a, b = np.exp(log_ab)
        return -stats.betabinom.logpmf(k_tr, n_tr, a, b).sum()

    a, b = np.exp(optimize.minimize(nll, [0.0, 0.0], method="Nelder-Mead").x)
    grid = np.linspace(1e-4, 1 - 1e-4, 4001)
    weight = stats.beta.pdf(grid, a, b)
    weight /= weight.sum()
    ceilings = {}
    for n in np.unique(test.valid_model_count.to_numpy(int)):
        k = np.arange(n + 1)
        label = label_of(k / n)
        pmf = stats.binom.pmf(k[None, :], n, grid[:, None])
        per_label = np.stack([pmf[:, label == y].sum(1) for y in range(3)], 1)
        ceilings[int(n)] = float((weight * per_label.max(1)).sum())
    values = test.valid_model_count.map(ceilings)
    return float(values.mean()), {"beta_a": float(a), "beta_b": float(b), "by_model_count": ceilings}


def peer_scores(train, test, raw_dir):
    """RouterBench only: per-arm success from the 8 non-arm models' recorded outcomes + task (train-fit)."""
    raw = pd.read_pickle(Path(raw_dir) / "routerbench_5shot.pkl").drop_duplicates("sample_id").set_index("sample_id")
    zero = pd.read_pickle(Path(raw_dir) / "routerbench_0shot.pkl").drop_duplicates("sample_id").set_index("sample_id")
    models = [c for c in raw.columns if "|" not in c and c not in ("prompt", "eval_name", "oracle_model_to_route_to")]
    peers = [m for m in models if m not in ARM_MODELS]
    outcomes = (raw[peers].astype(float).combine_first(zero[peers].astype(float)) >= 0.70).astype(float)

    def features(frame):
        ids = frame.sample_id.str.replace("routerbench:", "", regex=False)
        return outcomes.reindex(ids).fillna(0.5).to_numpy()

    tasks = sorted(set(train.source_task) | set(test.source_task))
    onehot = lambda f: (f.source_task.to_numpy()[:, None] == np.array(tasks)[None, :]).astype(float)
    x_train, x_test = np.c_[features(train), onehot(train)], np.c_[features(test), onehot(test)]
    return np.stack([LogisticRegression(max_iter=3000).fit(x_train, (train[f"success_{t}"] >= 1).astype(int))
                     .predict_proba(x_test)[:, 1] for t in TIERS], 1)


def load_final(run, arm, seeds, role="test"):
    preds = []
    for seed in seeds:
        frame = pd.read_parquet(Path(run) / "scores" / arm / f"seed-{seed}" / "final" / role / "predictions.parquet")
        preds.append(frame.set_index("sample_id").prediction.map(LABELS.index))
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--scores-root", required=True, help="experiments/results/context-length-qualification-20260925")
    parser.add_argument("--f4-run", required=True)
    parser.add_argument("--f5-run", required=True)
    parser.add_argument("--routerbench-raw", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(args.test)
    train = pd.read_parquet(args.train)
    scores_root = Path(args.scores_root)
    f0 = load_final(args.f5_run, "f0_frozen", [0])[0]
    f4 = load_final(args.f4_run, "f4_bidirectional", [42, 43, 44])
    f5 = load_final(args.f5_run, "f5_matched_prior", [42, 43, 44])
    curves, summaries, cascades, labels, notes = [], [], [], [], {}
    for source in ("routerbench", "xroutebench"):
        frame = test[test.source_dataset.eq(source)].sort_values("sample_id").reset_index(drop=True)
        tr = train[train.source_dataset.eq(source)]
        src = Source(frame)
        ids = frame.sample_id
        logits = {}
        for model, folder in (("r0", "r0/scores"), ("r1", "r1/a-scores")):
            s = pd.read_parquet(scores_root / folder / f"test_{source}-8192.parquet").set_index("sample_id")
            logits[model] = s.loc[ids, ["score_low", "score_medium", "score_high"]].to_numpy(float)
        r0_prob, r1_prob = softmax(logits["r0"]), special.expit(logits["r1"])
        y = frame.difficulty_label.map(LABELS.index).to_numpy()
        label_means = np.stack([tr.groupby("difficulty_label")[f"quality_{t}"].mean().reindex(LABELS).to_numpy()
                                for t in TIERS], 1)
        task_means = np.stack([tr.groupby("source_task")[f"quality_{t}"].mean() for t in TIERS], 1)
        task_index = {k: i for i, k in enumerate(tr.groupby("source_task").size().index)}
        task_rows = np.array([task_means[task_index[k]] if k in task_index else label_means.mean(0)
                              for k in frame.source_task])
        method_curves = {
            "oracle (knows every tier's outcome)": src.oracle_curve(),
            "true difficulty label + train calibration": src.curve(label_means[y]),
            "task identity + train calibration": src.curve(task_rows),
            "R1 success logits": src.curve(r1_prob),
            "R0-tau (cumulative threshold)": sorted({src.first_try(r0_tau(r0_prob, t))[::-1]
                                                    for t in np.linspace(0, 1, 201)}),
        }
        if source == "routerbench":
            method_curves["peer bound (8 other LLMs' outcomes + task)"] = src.curve(
                peer_scores(tr, frame, args.routerbench_raw))
        for name, curve in method_curves.items():
            curves += [{"source": source, "method": name, "relative_cost": c, "quality": q} for c, q in curve]
            summaries.append({"source": source, **summarize(name, src, curve)})
        points = {
            "F0 (operational R0 1.0)": [f0.loc[ids].to_numpy()],
            "F4 bidirectional": [p.loc[ids].to_numpy() for p in f4],
            "F5 matched prior": [p.loc[ids].to_numpy() for p in f5],
            "R1 locked policy": [np.argmax(r1_prob - R1_LOCKED[source] * src.dc, 1)],
            "always-low": [np.zeros(len(frame), int)], "always-medium": [np.ones(len(frame), int)],
            "always-high": [np.full(len(frame), 2)],
        }
        oracle_direct = np.where(src.S.max(1) == 1, src.S.argmax(1), 0)
        points["oracle direct (cheapest solving tier)"] = [oracle_direct]
        for name, preds in points.items():
            fq = [src.first_try(p) for p in preds]
            cas = [src.cascade(p) for p in preds]
            q, c = float(np.mean([a for a, _ in fq])), float(np.mean([b for _, b in fq]))
            row = {"source": source, "method": name, "first_try_quality": q, "first_try_relative_cost": c,
                   "gain_over_mixing": q - mixing_frontier(src.points, c)}
            row.update({k: float(np.mean([x[k] for x in cas])) for k in cas[0]})
            cascades.append(row)
        # Label accuracy against the original 3-class label and its noise ceiling.
        ceiling, info = label_noise_ceiling(tr, frame)
        notes[f"label_noise_{source}"] = info
        majority = tr.difficulty_label.value_counts().idxmax()
        task_major = tr.groupby("source_task").difficulty_label.agg(lambda s: s.value_counts().idxmax())
        label_rows = {
            "always the majority label": [np.full(len(frame), LABELS.index(majority))],
            "task majority label (train)": [frame.source_task.map(task_major).fillna(majority).map(LABELS.index).to_numpy()],
            "F0 (operational R0 1.0)": [f0.loc[ids].to_numpy()],
            "F4 bidirectional": [p.loc[ids].to_numpy() for p in f4],
            "F5 matched prior": [p.loc[ids].to_numpy() for p in f5],
        }
        for name, preds in label_rows.items():
            labels.append({"source": source, "method": name, "accuracy": float(np.mean([(p == y).mean() for p in preds]))})
        labels.append({"source": source, "method": "label-noise ceiling (Bayes, knows true solve rate)", "accuracy": ceiling})
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "curve-summary.csv", index=False)
    pd.DataFrame(cascades).to_csv(out / "methods-resolution-cost.csv", index=False)
    pd.DataFrame(labels).to_csv(out / "label-accuracy-ceiling.csv", index=False)
    (out / "notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
