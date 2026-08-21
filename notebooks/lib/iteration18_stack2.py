"""Second-stage candidate ranker over every expert's per-class opinion.

The log-pool is a fixed log-linear combination, so it can only follow a weighted vote.
The crosstab of "how many experts are right" against "is the pool right" shows ~350 cells
where the experts split 4-15 against 27 and the pool loses more than half of them: a
non-linear gate could plausibly recover some.  This ranker sees, for each (cell, class)
candidate, every expert's probability and rank for exactly that class, anchored on the
pool's own log-score through XGBoost `base_margin`.

Protocol: the pool exponents come from fold partitions the stacker is not scored on, and
the stacker itself is cross-fitted over cells.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_submit as S

EPS = 1e-9
TOPK = 6
PARAMS = dict(objective="rank:pairwise", eta=0.03, max_depth=5, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=15, reg_lambda=8.0,
              tree_method="hist", nthread=11)


def _rank_desc(a):
    o = np.argsort(-a, axis=1)
    r = np.empty_like(o)
    np.put_along_axis(r, o, np.arange(a.shape[1])[None, :], axis=1)
    return r.astype(np.float32)


def pool_scores(seed, used, fits, glia, classes, y):
    d = np.load(B.OUT / f"experts_oof_seed{seed}.npz", allow_pickle=True)
    allow = d["allow"]
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    lp = np.log(prior)
    logs = np.stack([np.log(np.maximum(d[n], EPS)) for n in used])
    z = np.zeros((len(y), len(classes)))
    z[glia] = LP.apply(logs[:, glia], *fits["glia"], lp, allow[glia])
    z[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], lp, allow[~glia])
    return z, logs, allow, d


def build_features(z, logs, allow, data, used, offset=0):
    n, C = z.shape
    K = min(TOPK, C)
    order = np.argsort(-z, axis=1)[:, :K]
    zs = z - z.max(1, keepdims=True)
    p_pool = np.exp(zs) / np.exp(zs).sum(1, keepdims=True)

    feats, names = [], []
    def add(mat, nm):
        feats.append(np.take_along_axis(mat, order, 1).astype(np.float32)[..., None])
        names.append(nm)

    add(z, "pool_z"); add(p_pool, "pool_p"); add(_rank_desc(z), "pool_rank")
    add(np.where(allow, 1.0, 0.0), "allow")
    rel = [np.take_along_axis(z, order, 1) - np.take_along_axis(z, order[:, :1], 1)]
    for m, nm in enumerate(used):
        p = np.exp(logs[m])
        add(p, f"p_{nm}")
        add(_rank_desc(p), f"r_{nm}")
        rel.append(np.take_along_axis(p, order, 1)
                   - np.take_along_axis(p, order[:, :1], 1))
    P = np.concatenate(feats, axis=2)
    names += ["rel_pool"] + [f"rel_{nm}" for nm in used]
    P = np.concatenate([P, np.stack(rel, axis=2)], axis=2)

    meta = data["meta_train"] if offset == 0 else data["meta_test"]
    counts = (data["counts_train"] if offset == 0 else data["counts_test"]).to_numpy()
    votes = np.stack([np.eye(C, dtype=np.float32)[np.where(allow, np.exp(l), -1).argmax(1)]
                      for l in logs]).sum(0)
    add(votes / len(used), "vote_share")
    names_v = names[-1]
    cell = np.hstack([
        np.log1p(counts.sum(1))[:, None], (counts > 0).sum(1)[:, None].astype(np.float32),
        np.log1p(meta["volume"].to_numpy())[:, None],
        meta["Region"].isna().to_numpy()[:, None].astype(np.float32),
        -(p_pool * np.log(np.maximum(p_pool, EPS))).sum(1, keepdims=True),
        np.take_along_axis(p_pool, order[:, :1], 1),
    ]).astype(np.float32)
    P = np.concatenate([P, feats[-1], np.repeat(cell[:, None, :], K, axis=1)], axis=2)
    names = names + [f"cell{j}" for j in range(cell.shape[1])]
    return P, order, names


def run(fit_seeds=(59, 83), eval_seeds=(18, 41), rounds=350, seeds=(0, 1)):
    import xgboost as xgb
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    glia = data["meta_train"]["Region"].isna().to_numpy()
    used, fits, _ = S.frozen_weights(partitions=fit_seeds)
    out = []
    for seed in eval_seeds:
        z, logs, allow, d = pool_scores(seed, used, fits, glia, classes, y)
        base_pred = classes[z.argmax(1)]
        base = float(np.mean(base_pred == y))
        P, order, names = build_features(z, logs, allow, data, used)
        n, K, Fdim = P.shape
        cand = classes[order]
        label = (cand == y[:, None]).astype(int)
        margin = np.take_along_axis(z, order, 1).astype(np.float32)
        score = np.zeros((n, K), np.float32)
        t0 = time.time()
        for fit, val in StratifiedKFold(5, shuffle=True, random_state=7).split(
                np.zeros(n), y):
            dtr = xgb.DMatrix(P[fit].reshape(-1, Fdim), label=label[fit].reshape(-1),
                              feature_names=names)
            dtr.set_group(np.full(len(fit), K)); dtr.set_base_margin(margin[fit].reshape(-1))
            dva = xgb.DMatrix(P[val].reshape(-1, Fdim), feature_names=names)
            dva.set_group(np.full(len(val), K)); dva.set_base_margin(margin[val].reshape(-1))
            acc = np.zeros(len(val) * K, np.float32)
            for s in seeds:
                acc += xgb.train({**PARAMS, "seed": s}, dtr, num_boost_round=rounds
                                 ).predict(dva, output_margin=True)
            score[val] = (acc / len(seeds)).reshape(len(val), K)
        pred = cand[np.arange(n), score.argmax(1)]
        acc = float(np.mean(pred == y))
        w = int(((pred == y) & (base_pred != y)).sum())
        l = int(((pred != y) & (base_pred == y)).sum())
        print(f"partition {seed}: pool {base:.4f} -> stacked {acc:.4f} "
              f"({100*(acc-base):+.2f} pt) wins {w} losses {l} "
              f"changed {int((pred!=base_pred).sum())} [{time.time()-t0:.0f}s]", flush=True)
        out.append(100 * (acc - base))
    print(f"mean stacker gain over the pool: {np.mean(out):+.2f} pt")


if __name__ == "__main__":
    run()
