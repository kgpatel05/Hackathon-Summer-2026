"""Iteration 12 - local atlas subcluster ecology beyond cell-type composition.

The adopted neighborhood block counts 60 final cell types among the nearest atlas donors.
The parent atlas also has a 24-way second-round subcluster annotation that is not fully
determined by final type (mean within-type purity 0.918).  This experiment records its
10- and 50-neighbor histograms, preserving local developmental/state structure that the
final labels collapse.

The matched null permutes subcluster identities *within final cell type* among donor
cells.  It therefore preserves every signal derivable from the existing 60-way neighbor
histogram and destroys only spatial organization within a type.  All 10,000 challenge
cells are excluded from the donor pool.  Screen: partition 809, five ET seeds; advance
only for >0.30 point gain, p<0.05, and >0.20 point advantage over the null.  No challenge
test label or withheld expression is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "subcluster_niche.npz"
PARTITION = 809
SEEDS = tuple(range(5))
ALPHA = 0.45


def decode(handle: h5py.File, key: str) -> np.ndarray:
    categories = [x.decode() for x in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.asarray([categories[c] if c >= 0 else "NA" for c in codes])


def build(meta_all: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        aux = decode(handle, "2nd round subcluster")
    categories = sorted(set(aux[donors]))
    index = {name: j for j, name in enumerate(categories)}
    donor_aux = aux[donors]
    donor_labels = labels[donors]
    donor_sections = sections[donors]
    rng = np.random.default_rng(20260819)
    null_aux = donor_aux.copy()
    # Retain P(subcluster | final type) exactly while removing spatial localization.
    for label in np.unique(donor_labels):
        rows = np.flatnonzero(donor_labels == label)
        null_aux[rows] = donor_aux[rng.permutation(rows)]

    q_sections = meta_all["Section_ID"].astype(str).to_numpy()
    q_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    real = np.zeros((len(meta_all), 2 * len(categories)), np.float32)
    null = np.zeros_like(real)
    for section in np.unique(q_sections):
        qrows = np.flatnonzero(q_sections == section)
        drows = np.flatnonzero(donor_sections == section)
        donor_xy = np.column_stack([ax[donors[drows]], ay[donors[drows]]])
        k = min(50, len(drows))
        _, nearest = cKDTree(donor_xy).query(q_xy[qrows], k=k)
        if nearest.ndim == 1:
            nearest = nearest[:, None]
        selected = drows[nearest]
        for i, row in enumerate(qrows):
            for block, width in enumerate((min(10, k), k)):
                cols = slice(block * len(categories), (block + 1) * len(categories))
                rcodes = [index[x] for x in donor_aux[selected[i, :width]]]
                ncodes = [index[x] for x in null_aux[selected[i, :width]]]
                real[row, cols] = np.bincount(rcodes, minlength=len(categories)) / width
                null[row, cols] = np.bincount(ncodes, minlength=len(categories)) / width
    np.savez_compressed(CACHE, real=real, null=null, categories=np.asarray(categories))
    return real, null


def oof(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    correct = np.zeros(len(y), bool); labels = np.asarray(classes)
    for train, valid in StratifiedKFold(5, shuffle=True, random_state=PARTITION).split(y, y):
        p = M.fit_extra_trees(x[train], pd.Series(y[train]), classes, x[valid], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        correct[valid] = labels[p.argmax(1)] == y[valid]
    return correct


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy(); classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    if CACHE.exists():
        cached = np.load(CACHE); real, null = cached["real"], cached["null"]
    else:
        real, null = build(meta_all)
    baseline = current_stack(meta_all, classes, list(counts_train.columns))
    configs = {
        "baseline_694": baseline,
        "+ subcluster niche": np.hstack([baseline, real[:len(y)]]).astype(np.float32),
        "+ permuted subcluster null": np.hstack([baseline, null[:len(y)]]).astype(np.float32),
    }
    print(f"subcluster block={real.shape[1]} features; partition={PARTITION}", flush=True)
    t0 = time.time(); results = {}
    for name, x in configs.items():
        results[name] = oof(x, y, classes)
        print(f"finished {name} ({time.time()-t0:.1f}s)", flush=True)
    base = results["baseline_694"]
    rows = []
    for name, ok in results.items():
        if name == "baseline_694": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, base)
            wins = int((ok & ~base).sum()); losses = int((base & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100 * (ok.mean() - base.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "subcluster_niche_screen.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real_row, null_row = rows[1], rows[2]
    passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
              real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
