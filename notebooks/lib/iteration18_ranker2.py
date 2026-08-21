"""Residual re-ranker: score = log p_incumbent + learned correction.

Anchoring the ranker to the incumbent's own log-posterior (XGBoost `base_margin`) means
the trees only have to model where the flat 60-way argmax is wrong, instead of first
re-deriving an ordering it already has.  Everything else is as in `iteration18_ranker`.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_classfeat as CF
import iteration18_ranker as R1
import iteration5_features as F

PARAMS = dict(objective="rank:pairwise", eta=0.03, max_depth=5, subsample=0.8,
              colsample_bytree=0.6, min_child_weight=10, reg_lambda=5.0,
              tree_method="hist", nthread=11)


def cell_pcs(data, n_components=24):
    counts = np.vstack([data["counts_train"].to_numpy(), data["counts_test"].to_numpy()])
    expr = F.log_cpm(counts.astype(np.float32))
    return PCA(n_components=n_components, random_state=0).fit_transform(expr).astype(np.float32)


def assemble(data, probs, allow, cf, offset, pcs):
    P, order, names = R1.build_pair_features(data, probs, allow, None, cf, offset)
    n, K, _ = P.shape
    sl = slice(offset, offset + n)
    P = np.concatenate([P, np.repeat(pcs[sl][:, None, :], K, axis=1)], axis=2)
    names = names + [f"pc{j}" for j in range(pcs.shape[1])]
    margin = np.log(np.maximum(np.take_along_axis(probs, order, 1), 1e-9)).astype(np.float32)
    return P, order, names, margin


def run(seed=18, rounds=400, folds=5, seeds=(0, 1, 2), verbose=True):
    import xgboost as xgb
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
    if seed == 18:
        probs, allow = B.prior_correct(c["oof_raw"], y, classes), c["oof_allow"]
    else:
        raw, allow = B.oof_probabilities(data, seed=seed, raw=True)
        probs = B.prior_correct(raw, y, classes)
    cf = CF.load()
    pcs = cell_pcs(data)
    P, order, names, margin = assemble(data, probs, allow, cf, 0, pcs)
    n, K, Fdim = P.shape
    cand = classes[order]
    label = (cand == y[:, None]).astype(int)
    base_pred = cand[:, 0]
    base = float(np.mean(base_pred == y))

    oof = np.zeros((n, K), np.float32)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    t0 = time.time()
    for fold, (fit, val) in enumerate(skf.split(np.zeros(n), y)):
        dtr = xgb.DMatrix(P[fit].reshape(-1, Fdim), label=label[fit].reshape(-1),
                          feature_names=names)
        dtr.set_group(np.full(len(fit), K))
        dtr.set_base_margin(margin[fit].reshape(-1))
        dva = xgb.DMatrix(P[val].reshape(-1, Fdim), feature_names=names)
        dva.set_group(np.full(len(val), K))
        dva.set_base_margin(margin[val].reshape(-1))
        acc = np.zeros(len(val) * K, np.float32)
        for s in seeds:
            bst = xgb.train({**PARAMS, "seed": s}, dtr, num_boost_round=rounds)
            acc += bst.predict(dva, output_margin=True)
        oof[val] = (acc / len(seeds)).reshape(len(val), K)
    pred = cand[np.arange(n), oof.argmax(1)]
    acc = float(np.mean(pred == y))
    wins = int(((pred == y) & (base_pred != y)).sum())
    loss = int(((pred != y) & (base_pred == y)).sum())
    if verbose:
        print(f"partition {seed}: incumbent {base:.4f} -> ranker {acc:.4f} "
              f"({100*(acc-base):+.2f} pt) changed {int((pred!=base_pred).sum())} "
              f"wins {wins} losses {loss}  [{time.time()-t0:.0f}s]", flush=True)
    return dict(seed=seed, base=base, acc=acc, gain=100 * (acc - base),
                wins=wins, losses=loss, oof=oof, order=order, pred=pred)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [18]
    for s in args:
        run(seed=s)
