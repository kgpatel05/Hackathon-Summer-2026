"""Iteration 10 - multiscale class-conditional spatial fields.

The adopted atlas-composition block records only the labels among the ten nearest
non-challenge atlas cells.  It discards both distance and the evidence for classes
that happen not to occur in those ten cells.  This experiment treats the labelled
external atlas cells as a spatial point process instead.  For each challenge cell
and each of the 60 classes it emits:

  * relative distance to the nearest external cell of that class;
  * Gaussian log-density at 30, 60 and 120 coordinate units; and
  * an adaptive Gaussian log-density whose bandwidth is the local 100-neighbour
    radius.

All densities are normalised across classes, giving 5 x 60 = 300 features.  The
donor pool excludes all 10,000 challenge cells before any calculation.  The null
permutes atlas labels within Section_ID, preserving coordinates, section class
frequencies, dimensionality and feature marginals while destroying the spatial
class field.

Pre-registered screen (before results): one stratified 5-fold partition, seed 131,
five ExtraTrees seeds.  Advance only if the real field gains >0.30 points over the
current 694-feature stack, exceeds the null gain by >0.20 points, and paired exact
McNemar p < 0.05.  Confirmation uses untouched partition seed 157 and 20 estimator
seeds; adopt only if gain >0.20 points and p < 0.05.  The hidden test labels are
never read by this script.

Usage:
    python3 notebooks/lib/iteration10_spatial_fields.py screen
    python3 notebooks/lib/iteration10_spatial_fields.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "spatial_fields.npz"
BASE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
ATLAS_ET_CACHE = Path("outputs/iteration9/atlas_et_block.npz")
ALPHA = 0.45
SCREEN_PARTITION = 131
CONFIRM_PARTITION = 157
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
FIXED_BANDWIDTHS = (30.0, 60.0, 120.0)
K_CLASS = 16


def spatial_field(meta_all: pd.DataFrame, classes: list[str], *, permute: bool) -> np.ndarray:
    """Return a 300-column class-conditional spatial field for every query cell."""
    _, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    donor_labels = labels[donors].copy()
    donor_sections = sections[donors]
    if permute:
        rng = np.random.default_rng(20260819)
        for section in np.unique(donor_sections):
            rows = np.flatnonzero(donor_sections == section)
            donor_labels[rows] = donor_labels[rng.permutation(rows)]

    query_sections = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    n, c = len(meta_all), len(classes)
    nearest = np.zeros((n, c), np.float32)
    log_density = [np.zeros((n, c), np.float32) for _ in range(4)]

    for section in np.unique(query_sections):
        qrows = np.flatnonzero(query_sections == section)
        drows = np.flatnonzero(donor_sections == section)
        donor_xy = np.column_stack([ax[donors[drows]], ay[donors[drows]]])
        local_radius = cKDTree(donor_xy).query(
            query_xy[qrows], k=min(100, len(drows))
        )[0]
        if local_radius.ndim == 1:
            local_radius = local_radius[:, None]
        radius100 = np.maximum(local_radius[:, -1], 1.0)

        for j, cls in enumerate(classes):
            class_rows = drows[donor_labels[drows] == cls]
            if len(class_rows) == 0:
                # A large finite value keeps the feature matrix tree-friendly.
                distance = 10.0 * radius100[:, None]
            else:
                k = min(K_CLASS, len(class_rows))
                class_xy = np.column_stack(
                    [ax[donors[class_rows]], ay[donors[class_rows]]]
                )
                distance = cKDTree(class_xy).query(query_xy[qrows], k=k)[0]
                if distance.ndim == 1:
                    distance = distance[:, None]

            nearest[qrows, j] = -np.log1p(distance[:, 0] / radius100)
            for b, bandwidth in enumerate(FIXED_BANDWIDTHS):
                density = np.exp(-0.5 * (distance / bandwidth) ** 2).sum(axis=1)
                log_density[b][qrows, j] = np.log(density + 1e-8)
            adaptive = np.exp(-0.5 * (distance / radius100[:, None]) ** 2).sum(axis=1)
            log_density[3][qrows, j] = np.log(adaptive + 1e-8)

    # Convert each scale to log P(class | local field).  Normalisation removes
    # section-level cell-density variation while retaining relative class evidence.
    normalised = [x - logsumexp(x, axis=1, keepdims=True) for x in log_density]
    return np.hstack([nearest, *normalised]).astype(np.float32)


def current_stack(meta_all: pd.DataFrame, classes: list[str], genes: list[str]) -> np.ndarray:
    """Reconstruct the adopted 694-column training stack from cached source blocks."""
    n_train = 5000
    base = np.load(BASE_CACHE, allow_pickle=True)
    atlas_et = np.load(ATLAS_ET_CACHE, allow_pickle=True)
    comp = F.atlas_composition(meta_all, classes, k=10)[:n_train]
    niche = F.atlas_niche(meta_all, genes, k=50, n_components=30)[:n_train]
    blocks = [
        base["BASE_TR"], base["EXT_TR"], base["SPA_TR"], base["NIC_TR"],
        comp, niche, base["ATL_TR"], atlas_et["ATL_ET_TR"], atlas_et["COARSE_TR"],
    ]
    out = np.hstack(blocks).astype(np.float32)
    assert out.shape == (n_train, 694), out.shape
    return out


def oof_correct(x: np.ndarray, y: np.ndarray, classes: list[str], partition: int,
                seeds: tuple[int, ...]) -> np.ndarray:
    correct = np.zeros(len(y), bool)
    class_array = np.asarray(classes)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for train, valid in folds.split(y, y):
        probs = M.fit_extra_trees(
            x[train], pd.Series(y[train]), classes, x[valid], seeds=seeds
        )
        probs = M.correct_prior(
            probs, M.prior_vector(pd.Series(y[train]), classes), ALPHA
        )
        correct[valid] = class_array[probs.argmax(1)] == y[valid]
    return correct


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

    if CACHE.exists():
        cached = np.load(CACHE)
        real, null = cached["real"], cached["null"]
        print(f"loaded {CACHE}: {real.shape}", flush=True)
    else:
        t0 = time.time()
        real = spatial_field(meta_all, classes, permute=False)
        null = spatial_field(meta_all, classes, permute=True)
        np.savez_compressed(CACHE, real=real, null=null, classes=np.asarray(classes))
        print(f"built {CACHE}: {real.shape} in {time.time()-t0:.1f}s", flush=True)

    baseline = current_stack(meta_all, classes, list(counts_train.columns))
    n = len(y)
    configs = {
        "baseline_694": baseline,
        "+ spatial field": np.hstack([baseline, real[:n]]).astype(np.float32),
        "+ shuffled field (null)": np.hstack([baseline, null[:n]]).astype(np.float32),
    }
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    print(f"mode={mode} partition={partition} estimator_seeds={len(seeds)}", flush=True)
    for name, x in configs.items():
        print(f"  {name:26s} {x.shape[1]} features", flush=True)

    results: dict[str, np.ndarray] = {}
    for name, x in configs.items():
        t0 = time.time()
        results[name] = oof_correct(x, y, classes, partition, seeds)
        print(f"finished {name} in {time.time()-t0:.1f}s", flush=True)

    base_ok = results["baseline_694"]
    rows = []
    for name, ok in results.items():
        gain = ok.mean() - base_ok.mean()
        if name == "baseline_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": ok.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:26s} acc={ok.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    result_path = OUT / f"spatial_fields_{mode}.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    real_row = rows[1]
    null_row = rows[2]
    if mode == "screen":
        passed = (real_row["gain_pt"] > 0.30 and
                  real_row["gain_pt"] - null_row["gain_pt"] > 0.20 and
                  real_row["p"] < 0.05)
        verdict = "ADVANCE TO CONFIRM" if passed else "REJECT"
    else:
        passed = real_row["gain_pt"] > 0.20 and real_row["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
    print(f"VERDICT: {verdict}; wrote {result_path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
