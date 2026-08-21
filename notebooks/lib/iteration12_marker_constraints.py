"""Iteration 12 - PRISM-inspired positive/inverse marker constraints.

PRISM (Bioinformatics 2026) improves spatial cell mapping by rewarding positive marker
support and explicitly penalising inverse-marker contradictions.  Here the mechanism is
adapted without pseudo-labels or query-label access: for each of 60 classes, the eight
largest and eight smallest atlas effect-size genes define one signed marker-contrast
feature.  Effects are estimated from non-challenge parent-atlas cells over the released
200 genes only.

The null randomly reassigns the same signed effect magnitudes to genes within each class,
preserving sparsity, scale and dimensionality.  Screen: partition 887, five ET seeds;
advance only for >0.30 point gain, p<0.05, and >0.20 point advantage over the null.  No
challenge test label or withheld expression is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "marker_constraints.npz"
PARTITION = 887
SEEDS = tuple(range(5))
ALPHA = 0.45
N_MARKERS = 8


def decode(handle: h5py.File, key: str) -> np.ndarray:
    categories = [x.decode() for x in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.asarray([F._normalise_label(categories[c]) if c >= 0 else "NA"
                       for c in codes])


def build(counts_train: pd.DataFrame, counts_test: pd.DataFrame,
          meta_train: pd.DataFrame, meta_test: pd.DataFrame,
          classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    genes = list(counts_train.columns)
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        ids = np.asarray([x.decode() for x in handle["obs/_index"][:]])
        atlas_genes = [x.decode() for x in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )[:, [lookup[g] for g in genes]]
        labels = decode(handle, "MERFISH cell type annotation")
    challenge = set(meta_train.index.astype(str)) | set(meta_test.index.astype(str))
    keep = np.asarray([cell not in challenge for cell in ids]) & np.isin(labels, classes)
    keep &= np.asarray(matrix.sum(1)).ravel() > 0
    atlas = F.log_cpm(np.asarray(matrix[keep].todense(), np.float32))
    atlas_labels = labels[keep]
    mean = atlas.mean(0); std = atlas.std(0) + 1e-5
    z = (atlas - mean) / std
    total = z.sum(0)

    weights = np.zeros((len(classes), len(genes)), np.float32)
    for j, label in enumerate(classes):
        rows = atlas_labels == label
        class_mean = z[rows].mean(0)
        rest_mean = (total - z[rows].sum(0)) / max((~rows).sum(), 1)
        effect = class_mean - rest_mean
        pos = np.argsort(effect)[-N_MARKERS:]
        neg = np.argsort(effect)[:N_MARKERS]
        selected = np.r_[pos, neg]
        weights[j, selected] = effect[selected]
        weights[j] /= np.linalg.norm(weights[j]) + 1e-8

    challenge_x = F.log_cpm(np.vstack([counts_train.to_numpy(np.float32),
                                       counts_test.to_numpy(np.float32)]))
    challenge_z = (challenge_x - mean) / std
    real = challenge_z @ weights.T
    rng = np.random.default_rng(20260819)
    null_weights = np.zeros_like(weights)
    for j in range(len(classes)):
        nonzero = weights[j][weights[j] != 0].copy()
        columns = rng.choice(len(genes), len(nonzero), replace=False)
        rng.shuffle(nonzero)
        null_weights[j, columns] = nonzero
    null = challenge_z @ null_weights.T

    # Put every class score on the same scale without using a response label.
    real = (real - real.mean(0)) / (real.std(0) + 1e-5)
    null = (null - null.mean(0)) / (null.std(0) + 1e-5)
    np.savez_compressed(CACHE, real=real.astype(np.float32), null=null.astype(np.float32),
                        weights=weights, genes=np.asarray(genes), classes=np.asarray(classes))
    return real.astype(np.float32), null.astype(np.float32)


def oof(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    correct = np.zeros(len(y), bool); labels = np.asarray(classes)
    for train, valid in StratifiedKFold(5, shuffle=True, random_state=PARTITION).split(y, y):
        p = M.fit_extra_trees(x[train], pd.Series(y[train]), classes, x[valid], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        correct[valid] = labels[p.argmax(1)] == y[valid]
    return correct


def main() -> None:
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy(); classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    if CACHE.exists():
        cached = np.load(CACHE); real, null = cached["real"], cached["null"]
    else:
        real, null = build(counts_train, counts_test, meta_train, meta_test, classes)
    baseline = current_stack(meta_all, classes, list(counts_train.columns))
    configs = {
        "baseline_694": baseline,
        "+ marker constraints": np.hstack([baseline, real[:len(y)]]).astype(np.float32),
        "+ randomized-marker null": np.hstack([baseline, null[:len(y)]]).astype(np.float32),
    }
    print(f"marker block={real.shape[1]} partition={PARTITION}", flush=True)
    results = {}; t0 = time.time()
    for name, x in configs.items():
        results[name] = oof(x, y, classes)
        print(f"finished {name} ({time.time()-t0:.1f}s)", flush=True)
    base = results["baseline_694"]; rows = []
    for name, ok in results.items():
        if name == "baseline_694": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, base)
            wins = int((ok & ~base).sum()); losses = int((base & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100 * (ok.mean() - base.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "marker_constraints_screen.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real_row, null_row = rows[1], rows[2]
    passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
              real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
