"""Selective local calibration in ensemble-opinion space.

Cells are neighbours only when the *entire expert bank* has a similar pattern of class
probabilities, uncertainty and disagreement.  Within each frozen-pool predicted class,
released labels of nearby design cells estimate the local confusion distribution.  The
router switches only when that local distribution prefers another metadata-compatible
class by a frozen margin.

The 60/40 cell split is shared with the pair-rule experiment.  All preprocessing and
neighbour labels are fitted on the 3,000 design cells.  Hyperparameters are selected by
cross-partition self-excluded queries on those design cells; the remaining 2,000 labels
are read only once for confirmation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, str(Path(__file__).parent))
import iteration22_router_common as C


SPLIT_SEED = 20260821
K_VALUES = (9, 15, 25, 40)
DELTA_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)
SHARE_VALUES = (0.35, 0.45, 0.55, 0.65)
N_COMPONENTS = 48


def opinion_features(bank: dict, z: np.ndarray) -> np.ndarray:
    from scipy.special import softmax
    p = bank["probs"].astype(np.float32)
    n, c = z.shape
    rows = np.arange(n)
    votes = np.zeros((n, c), np.float32)
    ep = np.where(bank["allow"][None], p, -1).argmax(2)
    for m in range(len(bank["names"])):
        votes[rows, ep[m]] += 1.0 / len(bank["names"])
    strong_names = ("etaug4_0.25_3", "etaug3", "atlaslam_lin", "atlasftlam",
                    "atlaslam_rf_0.1", "atlaslam_md")
    strong = [p[bank["names"].index(s)] for s in strong_names]
    pool = softmax(z, axis=1).astype(np.float32)
    order = np.argsort(-z, axis=1)
    margin = (z[rows, order[:, 0]] - z[rows, order[:, 1]])[:, None]
    entropy = (-(pool * np.log(np.maximum(pool, 1e-9))).sum(1))[:, None]
    x = np.column_stack([pool, p.mean(0), p.std(0), votes] + strong
                        + [margin, entropy]).astype(np.float32)
    return x


def fit_geometry(ref_x: np.ndarray) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler().fit(ref_x)
    xs = np.clip(scaler.transform(ref_x), -8, 8)
    pca = PCA(n_components=min(N_COMPONENTS, len(ref_x) - 1), random_state=SPLIT_SEED,
              whiten=True).fit(xs)
    return scaler, pca, normalize(pca.transform(xs)).astype(np.float32)


def transform(x: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return normalize(pca.transform(np.clip(scaler.transform(x), -8, 8))).astype(np.float32)


def local_distribution(ref_vec: np.ndarray, ref_base: np.ndarray, ref_y: np.ndarray,
                       query_vec: np.ndarray, query_base: np.ndarray,
                       query_allow: np.ndarray, classes: np.ndarray, k: int,
                       ref_ids: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    out = np.zeros((len(query_vec), len(classes)), np.float32)
    ci = {c: i for i, c in enumerate(classes)}
    for label in np.unique(query_base):
        rr = np.flatnonzero(ref_base == label)
        qq = np.flatnonzero(query_base == label)
        if len(rr) < 3:
            out[qq, ci[label]] = 1.0
            continue
        # Ask for one spare neighbour so identical cell IDs can be excluded during
        # design-set selection even though opinions come from a different OOF partition.
        kk = min(k + 1, len(rr))
        nn = NearestNeighbors(n_neighbors=kk, metric="cosine", algorithm="brute").fit(
            ref_vec[rr])
        dist, ind = nn.kneighbors(query_vec[qq])
        for a, qi in enumerate(qq):
            chosen = []
            dd = []
            for d, loc in zip(dist[a], ind[a]):
                ri = rr[loc]
                if ref_ids[ri] == query_ids[qi]:
                    continue
                chosen.append(ri); dd.append(d)
                if len(chosen) == k:
                    break
            if not chosen:
                out[qi, ci[label]] = 1.0
                continue
            dd = np.asarray(dd)
            weight = np.exp(-(dd - dd.min()) / 0.12)
            for w, ri in zip(weight, chosen):
                out[qi, ci[ref_y[ri]]] += float(w)
            out[qi] *= query_allow[qi]
            s = out[qi].sum()
            if s <= 0:
                out[qi, ci[label]] = 1.0
            else:
                out[qi] /= s
    return out


def route(base: np.ndarray, local: np.ndarray, classes: np.ndarray,
          delta: float, share: float) -> tuple[np.ndarray, np.ndarray]:
    ci = {c: i for i, c in enumerate(classes)}
    rows = np.arange(len(base))
    bi = np.array([ci[v] for v in base])
    ai = local.argmax(1)
    gain = local[rows, ai] - local[rows, bi]
    take = (ai != bi) & (gain >= delta) & (local[rows, ai] >= share)
    pred = base.copy(); pred[take] = classes[ai[take]]
    return pred, gain


def main() -> None:
    t0 = time.time()
    b18, b41 = C.load_experts(18), C.load_experts(41)
    y = b18["y"]
    idx = np.arange(len(y))
    design, confirm = train_test_split(idx, test_size=0.40, random_state=SPLIT_SEED,
                                       stratify=y)
    z18, z41 = C.pool_logits(b18), C.pool_logits(b41)
    base18 = b18["classes"][z18.argmax(1)]
    base41 = b41["classes"][z41.argmax(1)]
    scaler, pca, ref_all = fit_geometry(opinion_features(b18, z18)[design])
    # Geometry indices are local to the design reference bank.
    ref_vec = ref_all
    ref_base, ref_y = base18[design], y[design]
    ref_ids = design
    q41 = transform(opinion_features(b41, z41), scaler, pca)
    glia = b18["meta"]["Region"].isna().to_numpy()

    rows = []
    locals_by_k = {}
    for k in K_VALUES:
        local = local_distribution(ref_vec, ref_base, ref_y, q41[design], base41[design],
                                   b41["allow"][design], b41["classes"], k,
                                   ref_ids, design)
        locals_by_k[k] = local
        for delta in DELTA_VALUES:
            for share in SHARE_VALUES:
                pred, score = route(base41[design], local, b41["classes"], delta, share)
                r = C.metric_row("opinion_knn", pred, base41[design], y[design],
                                 glia[design], score)
                r.update(k=k, delta=delta, share=share, partition=41,
                         split="design_selection")
                rows.append(r)
    grid = pd.DataFrame(rows)
    grid.to_csv(C.OUT / "opinion_knn_screen.csv", index=False)
    eligible = grid[(grid.net > 0) & (grid.changed >= 10)].copy()
    if eligible.empty:
        best = grid.sort_values(["net", "wins"], ascending=False).iloc[0]
    else:
        eligible["safety"] = eligible.net / np.sqrt(eligible.changed)
        best = eligible.sort_values(["safety", "net"], ascending=False).iloc[0]
    k, delta, share = int(best.k), float(best.delta), float(best.share)
    print("screen winner:\n" + best.to_string())

    confirm_rows = []
    for seed in (59, 83):
        bank = C.load_experts(seed)
        z = C.pool_logits(bank)
        base = bank["classes"][z.argmax(1)]
        q = transform(opinion_features(bank, z), scaler, pca)
        local = local_distribution(ref_vec, ref_base, ref_y, q[confirm], base[confirm],
                                   bank["allow"][confirm], bank["classes"], k,
                                   ref_ids, confirm)
        pred, score = route(base[confirm], local, bank["classes"], delta, share)
        r0 = C.metric_row("iteration21_pool", base[confirm], base[confirm], y[confirm],
                          glia[confirm])
        r1 = C.metric_row("opinion_knn", pred, base[confirm], y[confirm], glia[confirm], score)
        alt = bank["classes"][local.argmax(1)]
        changed = int((pred != base[confirm]).sum())
        eligible = np.flatnonzero(alt != base[confirm])
        rng = np.random.default_rng(22100 + seed)
        random_idx = rng.choice(eligible, min(changed, len(eligible)), replace=False)
        pr = base[confirm].copy(); pr[random_idx] = alt[random_idx]
        rr = C.metric_row("random_same_coverage", pr, base[confirm], y[confirm],
                          glia[confirm], score)
        order = np.argsort(-z[confirm], axis=1)
        margin = (z[confirm][np.arange(len(confirm)), order[:, 0]]
                  - z[confirm][np.arange(len(confirm)), order[:, 1]])
        low = eligible[np.argsort(margin[eligible])[:changed]]
        pm = base[confirm].copy(); pm[low] = alt[low]
        rm = C.metric_row("low_margin_same_coverage", pm, base[confirm], y[confirm],
                          glia[confirm], score)
        for r in (r0, r1, rr, rm):
            r.update(partition=seed, k=k, delta=delta, share=share,
                     split="cell_disjoint_confirmation", runtime_sec=time.time() - t0,
                     device="cpu (PCA/cosine kNN; MPS not applicable)")
        confirm_rows.extend([r0, r1, rr, rm])
        print(f"confirm {seed}: pool={r0['accuracy']:.4f} router={r1['accuracy']:.4f} "
              f"net={r1['net']:+d} wins/losses={r1['wins']}/{r1['losses']} "
              f"p={r1['mcnemar_p']:.4g} coverage={r1['coverage']:.3f}")
    cf = pd.DataFrame(confirm_rows)
    cf.to_csv(C.OUT / "opinion_knn_confirmation.csv", index=False)
    r = cf[cf.candidate == "opinion_knn"].reset_index(drop=True)
    b = cf[cf.candidate == "iteration21_pool"].reset_index(drop=True)
    gains = 100 * (r.accuracy - b.accuracy)
    confirmed_ok = bool((gains > 0).all() and gains.mean() >= 0.10 and r.net.sum() >= 4)
    freeze = {
        "mechanism": "cell-disjoint local calibration in ensemble-opinion space",
        "design_cells": len(design), "confirmation_cells": len(confirm),
        "n_components": N_COMPONENTS, "k": k, "delta": delta, "share": share,
        "fit_partition": 18, "selection_partition": 41,
        "confirmation_partitions": [59, 83],
        "confirmation_gain_pt": gains.tolist(),
        "mean_confirmation_gain_pt": float(gains.mean()),
        "worst_confirmation_gain_pt": float(gains.min()),
        "confirmed": confirmed_ok, "test_scoring_authorized": confirmed_ok,
        "test_truth_read": False, "runtime_sec": time.time() - t0,
        "device": "cpu", "device_reason": "PCA/cosine kNN; MPS not applicable",
    }
    (C.OUT / "opinion_knn_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
