"""Prototype of a router-agnostic evaluation: calibrate any router's scores into per-tier success
probabilities on D_dev, then evaluate one-shot and escalation-aware start-tier rules on test with
service utility U = resolved - lam * cost - kappa * failed_attempts (v2 outcomes for both splits)."""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import special
from sklearn.isotonic import IsotonicRegression

M = (sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[3])).rstrip('/') + '/'  # repo root holding ml/artifacts
sys.path.insert(0, M + 'ml/src')
from llm_router_labeling.shared.structured_text import structured_duplicate_key  # noqa: E402
from llm_router_labeling.routing_data.outcome_audit import xroute_cell_key  # noqa: E402
TIERS = ('low', 'medium', 'high'); LABELS = ('easy', 'medium', 'hard')
rb_fix = pd.read_parquet(M + 'ml/artifacts/benchmark-outcome-audit-v1/routerbench-corrections.parquet').set_index(['sample_id', 'model']).verified_accepted
rb_exc = set(pd.read_parquet(M + 'ml/artifacts/benchmark-outcome-audit-v1/routerbench-exclusions.parquet').sample_id)
xc = pd.read_parquet(M + 'ml/artifacts/benchmark-outcome-audit-v1/xroutebench-corrections.parquet').set_index('cell_key').verified_accepted
def xmap(split):
    raw = pd.read_parquet(M + f'ml/data/raw/xroutebench/ea4b6e1b29d9a734f55f0a637baf326bad6aa681/llmrouter_generic-{split}.parquet')
    ident = raw.task_id.where(raw.task_id.notna(), raw.embedding_id).astype(str)
    raw['sid'] = (raw.task_name.astype(str) + '|' + ident + '|' + raw['query'].map(structured_duplicate_key)).map(lambda k: 'xroutebench:' + hashlib.sha256(k.encode()).hexdigest()[:20])
    raw['fix'] = xroute_cell_key(raw).map(xc)
    return raw.dropna(subset=['fix']).drop_duplicates(['sid', 'model_name']).set_index(['sid', 'model_name']).fix
XFIX = pd.concat([xmap('train'), xmap('test')])
def v2_success(frame):
    S = frame[[f'success_{t}' for t in TIERS]].to_numpy(float).copy(); keep = np.ones(len(frame), bool)
    for i, r in enumerate(frame.itertuples()):
        if r.source_dataset == 'routerbench':
            rid = r.sample_id.replace('routerbench:', ''); keep[i] = rid not in rb_exc
            for j, t in enumerate(TIERS):
                k = (rid, getattr(r, f'model_{t}'))
                if k in rb_fix.index: S[i, j] = float(rb_fix[k])
        else:
            for j, t in enumerate(TIERS):
                k = (r.sample_id, getattr(r, f'model_{t}'))
                if k in XFIX.index: S[i, j] = max(S[i, j], float(XFIX[k]))
    return S, keep

dev = pd.read_parquet(M + 'ml/artifacts/routing-objective-continual-v1/dataset/validation.parquet')
test = pd.read_parquet(M + 'ml/artifacts/retraining-effect-f4-user-kit/retraining-effect-f4/inputs/test.parquet')
train = pd.read_parquet(M + 'ml/artifacts/routing-objective-continual-v1/dataset/train.parquet')
SC = M + 'experiments/results/context-length-qualification-20260925/'
def scores(model, split_files, ids):
    s = pd.concat([pd.read_parquet(SC + f'{model}/{f}') for f in split_files]).drop_duplicates('sample_id').set_index('sample_id')
    return s.loc[ids, ['score_low', 'score_medium', 'score_high']].to_numpy(float)

rows = []
LAMS = (0.05, 0.2, 0.5); KAPPAS = (0.0, 0.05, 0.15, 0.4)
for source in ('routerbench', 'xroutebench'):
    d = dev[dev.source_dataset.eq(source)].reset_index(drop=True); t = test[test.source_dataset.eq(source)].reset_index(drop=True)
    Sd, kd = v2_success(d); St, kt = v2_success(t)
    d, Sd = d[kd].reset_index(drop=True), Sd[kd]; t, St = t[kt].reset_index(drop=True), St[kt]
    Cd = d[[f'actual_cost_{x}' for x in TIERS]].to_numpy(float); Ct = t[[f'actual_cost_{x}' for x in TIERS]].to_numpy(float)
    unit = Ct[:, 2].mean()  # cost unit: always-high mean cost on this source's test
    c_hat = Cd.mean(0) / Cd[:, 2].mean()  # decision cost per tier (dev mean, high = 1)
    tr = train[train.source_dataset.eq(source)]
    # --- adapters: every router -> calibrated per-tier success probabilities ---
    raw = {}
    r0d = special.softmax(scores('r0/scores', ['validation-8192.parquet'], d.sample_id), 1)
    r0t = special.softmax(scores('r0/scores', [f'test_{source}-8192.parquet'], t.sample_id), 1)
    trS, _ = v2_success(tr.reset_index(drop=True))
    calib = np.stack([trS[tr.difficulty_label.eq(l).to_numpy()].mean(0) for l in LABELS])  # P(success_t | label) on train
    raw['R0 (label probs x train table)'] = (r0d @ calib, r0t @ calib)
    raw['R1 (success logits)'] = (special.expit(scores('r1/a-scores', ['validation-8192.parquet'], d.sample_id)),
                                  special.expit(scores('r1/a-scores', [f'test_{source}-8192.parquet'], t.sample_id)))
    calibrated = {}
    for name, (pd_, pt_) in raw.items():
        P = np.zeros_like(pt_)
        for j in range(3):
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pd_[:, j], Sd[:, j])
            P[:, j] = iso.predict(pt_[:, j])
        calibrated[name] = P
    # dependence between tier failures, fitted on dev: P(fa & fb) = rho*min + (1-rho)*prod
    fd = 1 - Sd
    pairs = [(0, 1), (1, 2), (0, 2)]
    num = sum((fd[:, a] * fd[:, b]).mean() - fd[:, a].mean() * fd[:, b].mean() for a, b in pairs)
    # Largest covariance two Bernoulli failures can have is min(fa, fb) - fa * fb (nested failures).
    den = sum(min(fd[:, a].mean(), fd[:, b].mean()) - fd[:, a].mean() * fd[:, b].mean() for a, b in pairs)
    rho = float(np.clip(num / den, 0, 1))
    def chain(P, s):
        f = 1 - P[:, s:]  # failure probs along the chain
        joint = [f[:, 0]]
        for k in range(1, f.shape[1]):
            prev = joint[-1]
            joint.append(rho * np.minimum(prev, f[:, k]) + (1 - rho) * prev * f[:, k])
        exp_cost = c_hat[s] + sum(joint[k - 1] * c_hat[s + k] for k in range(1, f.shape[1]))
        return 1 - joint[-1], exp_cost, sum(joint)
    def realized(start):
        n = len(start); solved = np.zeros(n, bool); cost = np.zeros(n); fails = np.zeros(n)
        for k in range(3):
            act = ~solved & (start <= k)
            cost[act] += Ct[act, k]; ok = act & (St[:, k] == 1); fails[act & ~ok] += 1; solved |= ok
        i = np.arange(n)
        return dict(first=St[i, start].mean(), resolved=solved.mean(), cost=cost.mean() / unit, fails=fails.mean())
    policies = {'start low': lambda lam, kap: np.zeros(len(t), int), 'start medium': lambda lam, kap: np.ones(len(t), int),
                'high only': lambda lam, kap: np.full(len(t), 2),
                'R0 argmax (as deployed)': lambda lam, kap: r0t.argmax(1)}
    for name, P in calibrated.items():
        policies[f'{name} | one-shot'] = (lambda P: lambda lam, kap: np.argmax(P - lam * c_hat, 1))(P)
        def ea(P):
            def rule(lam, kap):
                U = np.stack([(lambda r, c, fl: r - lam * c - kap * fl)(*chain(P, s)) for s in range(3)], 1)
                return U.argmax(1)
            return rule
        policies[f'{name} | escalation-aware'] = ea(P)
    best = np.where(St.max(1) == 1, St.argmax(1), 0)
    policies['oracle'] = lambda lam, kap: best
    for lam in LAMS:
        for kap in KAPPAS:
            for name, rule in policies.items():
                r = realized(rule(lam, kap))
                rows.append(dict(source=source, lam=lam, kappa=kap, policy=name, rho=rho, **r,
                                 utility=r['resolved'] - lam * r['cost'] - kap * r['fails']))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).resolve().parent / 'utility-grid.csv', index=False)
pd.set_option('display.width', 250)
print('rho', out.groupby('source').rho.first().round(3).to_dict())
for (src, lam), g in out.groupby(['source', 'lam']):
    if lam != 0.2: continue
    piv = g.pivot(index='policy', columns='kappa', values='utility').round(3)
    print(f'\n== {src}  lambda={lam}: utility by kappa'); print(piv.sort_values(0.15, ascending=False).to_string())
g = out[(out.lam.eq(0.2)) & (out.kappa.eq(0.15))]
print('\n== detail lambda=0.2 kappa=0.15'); print(g[['source', 'policy', 'first', 'resolved', 'cost', 'fails', 'utility']].round(3).to_string(index=False))
