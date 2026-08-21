"""Candidate re-ranking over (cell, class) pairs.

The incumbent's top-2 candidate set already contains the truth for 93.0% of cells while
its top-1 is right for 80.3%.  The gap is a decision problem, not a representation
problem, so the 60-way argmax is replaced by a learned ranker over the top-K candidates
whose features are indexed by (cell, class) - atlas multinomial likelihood, per-class
atlas kNN distance, multi-scale per-class neighbourhood composition, and every expert's
posterior for that specific class.

Training uses only released training cells and out-of-fold incumbent probabilities.
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
import iteration18_classfeat as CF
import iteration5_features as F

TOPK = 8
BLOCKS = {"BASE": (0, 371), "EXT": (371, 431), "SPA": (431, 439), "NIC": (439, 469),
          "COMP": (469, 530), "ANIC": (530, 560), "ATL": (560, 620),
          "ATL_ET": (620, 680), "COARSE": (680, 694)}


def _rank_desc(a):
    o = np.argsort(-a, axis=1)
    r = np.empty_like(o)
    np.put_along_axis(r, o, np.arange(a.shape[1])[None, :], axis=1)
    return r.astype(np.float32)


def _rank_asc(a):
    return _rank_desc(-a)


def build_pair_features(data, probs, allow, rows, cf, offset):
    """rows: indices into the 10,000-cell challenge order (train first, then test)."""
    classes = data["classes"]
    n, C = probs.shape
    K = min(TOPK, C)
    masked = np.where(allow, probs, -1.0)
    order = np.argsort(-masked, axis=1)[:, :K]

    x = data["x_train"] if offset == 0 else data["x_test"]
    ext, atl = x[:, slice(*BLOCKS["EXT"])], x[:, slice(*BLOCKS["ATL"])]
    aet, coarse = x[:, slice(*BLOCKS["ATL_ET"])], x[:, slice(*BLOCKS["COARSE"])]

    h = np.load(B.OUT / "hierarchy_maps.npz", allow_pickle=True)
    g = h["r1"].astype(str)
    groups = np.array(sorted(set(g)))
    gcol = np.array([list(groups).index(t) for t in g])
    Pg = np.zeros((n, len(groups)), np.float32)
    for j in range(C):
        Pg[:, gcol[j]] += probs[:, j]

    sl = slice(offset, offset + n)
    loglik_n = cf["loglik_n"][sl]
    cos, corr = cf["cos"][sl], cf["corr"][sl]
    knn, comp = cf["knn"][sl], cf["comp"][sl]

    lp = np.log(np.maximum(probs, 1e-9))
    ent = -(probs * lp).sum(1, keepdims=True)
    top1 = np.take_along_axis(probs, order[:, :1], 1)
    top2 = np.take_along_axis(probs, order[:, 1:2], 1)

    r_ll = _rank_desc(loglik_n); r_cos = _rank_desc(cos); r_corr = _rank_desc(corr)
    r_knn = [_rank_asc(knn[:, :, t]) for t in range(knn.shape[2])]
    r_p = _rank_desc(masked)

    meta = data["meta_train"] if offset == 0 else data["meta_test"]
    counts = (data["counts_train"] if offset == 0 else data["counts_test"]).to_numpy()
    cell = np.hstack([
        np.log1p(counts.sum(1))[:, None],
        (counts > 0).sum(1)[:, None].astype(np.float32),
        np.log1p(meta["volume"].to_numpy())[:, None],
        meta["Region"].isna().to_numpy()[:, None].astype(np.float32),
        ent, top1, top2, (top1 - top2),
    ]).astype(np.float32)

    prior = pd.Series(data["y"]).value_counts(normalize=True).reindex(classes).fillna(
        1e-9).to_numpy().astype(np.float32)

    feats, names = [], []
    def add(mat, name):
        feats.append(np.take_along_axis(mat, order, 1).astype(np.float32)[..., None])
        names.append(name)

    add(probs, "p_inc"); add(lp, "logp_inc"); add(r_p, "rank_inc")
    add(np.where(allow, 1.0, 0.0), "allow")
    add(ext, "p_sni"); add(atl, "p_atlas_lr"); add(aet, "p_atlas_et")
    add(loglik_n, "loglik_n"); add(r_ll, "rank_loglik")
    add(cos, "cos"); add(r_cos, "rank_cos"); add(corr, "corr"); add(r_corr, "rank_corr")
    for t, k in enumerate(cf["knn_k"]):
        add(knn[:, :, t], f"knn{k}"); add(r_knn[t], f"rank_knn{k}")
    for t, k in enumerate(cf["comp_k"]):
        add(comp[:, :, t], f"comp{k}")
    add(np.tile(prior[None, :], (n, 1)), "prior")
    add(Pg[:, gcol], "p_coarse_inc")
    add(coarse[:, gcol], "p_coarse_atlas")
    P = np.concatenate(feats, axis=2)                        # (n, K, F)

    # relative-to-best versions of the strongest channels
    rel = []
    for j, nm in enumerate(names):
        if nm in ("p_inc", "loglik_n", "cos", "corr", "knn1", "knn5", "knn15",
                  "knn50", "p_sni", "p_atlas_lr", "p_atlas_et", "p_coarse_inc"):
            rel.append(P[:, :, j] - P[:, :1, j])
            names.append(nm + "_rel")
    P = np.concatenate([P, np.stack(rel, axis=2)], axis=2)

    cellrep = np.repeat(cell[:, None, :], K, axis=1)
    names += [f"cell{j}" for j in range(cell.shape[1])]
    P = np.concatenate([P, cellrep], axis=2)
    return P, order, names


def fit_ranker(Xtr, ytr, gtr, seed=0, n_estimators=500):
    import xgboost as xgb
    m = xgb.XGBRanker(objective="rank:pairwise", n_estimators=n_estimators,
                      learning_rate=0.05, max_depth=6, subsample=0.85,
                      colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
                      random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(Xtr, ytr, group=gtr, verbose=False)
    return m


def main(seed=18, n_estimators=500, ranker_folds=5):
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
    probs = B.prior_correct(c["oof_raw"], y, classes)
    allow = c["oof_allow"]
    cf = CF.load()

    t0 = time.time()
    P, order, names = build_pair_features(data, probs, allow, None, cf, 0)
    n, K, Fdim = P.shape
    print(f"pair tensor {P.shape} ({time.time()-t0:.0f}s), {len(names)} named features")

    cand = classes[order]
    label = (cand == y[:, None]).astype(int)
    base_pred = classes[np.where(allow, probs, -1.0).argmax(1)]
    base = float(np.mean(base_pred == y))
    print(f"incumbent OOF {base:.4f} | candidate coverage {label.max(1).mean():.4f}")

    oof_score = np.zeros((n, K), np.float32)
    skf = StratifiedKFold(ranker_folds, shuffle=True, random_state=seed)
    for fold, (fit, val) in enumerate(skf.split(np.zeros(n), y)):
        Xtr = P[fit].reshape(-1, Fdim)
        ytr = label[fit].reshape(-1)
        gtr = np.full(len(fit), K)
        m = fit_ranker(Xtr, ytr, gtr, seed=fold)
        oof_score[val] = m.predict(P[val].reshape(-1, Fdim)).reshape(len(val), K)
        print(f"  ranker fold {fold+1}/{ranker_folds}", flush=True)

    pred = cand[np.arange(n), oof_score.argmax(1)]
    acc = float(np.mean(pred == y))
    print(f"\nranker OOF accuracy {acc:.4f}   gain {100*(acc-base):+.2f} pt   "
          f"changed {int((pred != base_pred).sum())}")
    wins = int(((pred == y) & (base_pred != y)).sum())
    loss = int(((pred != y) & (base_pred == y)).sum())
    print(f"wins {wins} losses {loss}")

    np.savez_compressed(B.OUT / f"ranker_oof_seed{seed}.npz",
                        score=oof_score, order=order, pred=pred, y=y, base=base_pred)
    return acc, base


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 18)
