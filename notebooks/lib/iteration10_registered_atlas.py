"""Iteration 10 - registered cross-section anatomical label map.

Raw MERFISH coordinates live in unrelated imaging frames.  The existing local atlas
composition uses only a query's own section, while the existing atlas ExtraTrees sees
raw position plus a section identifier and therefore cannot share an anatomical map
across serial sections.  Here each section is centered, PCA-rotated, dorsally oriented
and radius-scaled from non-challenge atlas donors.  Queries then vote from the 200
nearest registered atlas cells across sections with the same C/L/S/T axial suffix.

Only public atlas donor labels are used; all 10,000 challenge cells are excluded from
transform estimation and voting.  A section-wise label permutation is the width-matched
null.  No challenge test label or withheld transcript is read.

Pre-registered screen: partition seed 251, five estimator seeds.  Advance only if the
real 61-column block gains >0.30 points over the 694-feature incumbent, paired exact
McNemar p<0.05, and exceeds the null gain by >0.20 points.  Confirmation uses untouched
partition seed 269 and 20 estimator seeds; adopt only for >0.20 points and p<0.05.

Usage:
    python3 notebooks/lib/iteration10_registered_atlas.py screen
    python3 notebooks/lib/iteration10_registered_atlas.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "registered_atlas.npz"
K_REGISTERED = 200
ALPHA = 0.45
SCREEN_PARTITION = 251
CONFIRM_PARTITION = 269
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def register_and_vote(meta_all: pd.DataFrame, classes: list[str],
                      neuron_classes: set[str]) -> tuple[np.ndarray, np.ndarray]:
    ids, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    donor_sections = sections[donors]
    donor_labels = labels[donors]
    donor_xy = np.column_stack([ax[donors], ay[donors]])
    query_sections = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    donor_registered = np.zeros_like(donor_xy, dtype=np.float32)
    query_registered = np.zeros_like(query_xy, dtype=np.float32)

    # Estimate every transform from external donors only, then apply it unchanged
    # to the challenge queries in that section.
    for section in np.unique(query_sections):
        drows = np.flatnonzero(donor_sections == section)
        qrows = np.flatnonzero(query_sections == section)
        points = donor_xy[drows]
        centre = points.mean(axis=0)
        centred = points - centre
        _, _, components = np.linalg.svd(centred, full_matrices=False)
        donor_rotated = centred @ components.T
        query_rotated = (query_xy[qrows] - centre) @ components.T
        dorsal = np.isin(donor_labels[drows], list(neuron_classes))
        if dorsal.any() and donor_rotated[dorsal, 1].mean() < 0:
            donor_rotated[:, 1] *= -1
            query_rotated[:, 1] *= -1
        scale = np.percentile(np.linalg.norm(donor_rotated, axis=1), 95) + 1e-9
        donor_registered[drows] = donor_rotated / scale
        query_registered[qrows] = query_rotated / scale

    class_index = {name: i for i, name in enumerate(classes)}
    other = len(classes)
    donor_code = np.asarray([class_index.get(name, other) for name in donor_labels])
    rng = np.random.default_rng(20260819)
    null_code = donor_code.copy()
    for section in np.unique(donor_sections):
        rows = np.flatnonzero(donor_sections == section)
        null_code[rows] = donor_code[rng.permutation(rows)]

    real = np.zeros((len(meta_all), other + 1), np.float32)
    null = np.zeros_like(real)
    donor_suffix = np.asarray([section.rsplit("_", 1)[-1] for section in donor_sections])
    query_suffix = np.asarray([section.rsplit("_", 1)[-1] for section in query_sections])
    for suffix in np.unique(query_suffix):
        drows = np.flatnonzero(donor_suffix == suffix)
        qrows = np.flatnonzero(query_suffix == suffix)
        k = min(K_REGISTERED, len(drows))
        distance, nn = cKDTree(donor_registered[drows]).query(query_registered[qrows], k=k)
        if nn.ndim == 1:
            nn = nn[:, None]
            distance = distance[:, None]
        selected = drows[nn]
        # Mild distance weighting prevents one dense section from dominating while
        # retaining a broad, low-variance anatomical neighbourhood.
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-5)
        weight = np.exp(-0.5 * (distance / scale) ** 2)
        weight /= weight.sum(axis=1, keepdims=True)
        for i, row in enumerate(qrows):
            real[row] = np.bincount(
                donor_code[selected[i]], weights=weight[i], minlength=other + 1
            )
            null[row] = np.bincount(
                null_code[selected[i]], weights=weight[i], minlength=other + 1
            )
    return real, null


def oof_correct(x: np.ndarray, y: np.ndarray, classes: list[str], partition: int,
                seeds: tuple[int, ...]) -> np.ndarray:
    correct = np.zeros(len(y), bool)
    class_array = np.asarray(classes)
    for train, valid in StratifiedKFold(5, shuffle=True, random_state=partition).split(y, y):
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
    class_array = np.asarray(classes)
    neuron_classes = set(meta_train.loc[meta_train["Region"].notna(), F.TARGET].astype(str))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    if CACHE.exists():
        cached = np.load(CACHE)
        real, null = cached["real"], cached["null"]
        print(f"loaded {CACHE}: {real.shape}", flush=True)
    else:
        t0 = time.time()
        real, null = register_and_vote(meta_all, classes, neuron_classes)
        np.savez_compressed(CACHE, real=real, null=null, classes=np.asarray(classes))
        print(f"built {CACHE}: {real.shape} in {time.time()-t0:.1f}s", flush=True)
    print(f"standalone train accuracy: real="
          f"{(class_array[real[:len(y), :len(classes)].argmax(1)] == y).mean():.4f} "
          f"null={(class_array[null[:len(y), :len(classes)].argmax(1)] == y).mean():.4f}",
          flush=True)

    baseline = current_stack(meta_all, classes, list(counts_train.columns))
    configs = {
        "baseline_694": baseline,
        "+ registered atlas": np.hstack([baseline, real[:len(y)]]).astype(np.float32),
        "+ registered shuffled (null)": np.hstack([baseline, null[:len(y)]]).astype(np.float32),
    }
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    print(f"mode={mode} partition={partition} estimator_seeds={len(seeds)}", flush=True)
    results = {}
    for name, x in configs.items():
        t0 = time.time()
        results[name] = oof_correct(x, y, classes, partition, seeds)
        print(f"finished {name} ({x.shape[1]} features) in {time.time()-t0:.1f}s", flush=True)

    base_ok = results["baseline_694"]
    rows = []
    for name, correct in results.items():
        gain = correct.mean() - base_ok.mean()
        if name == "baseline_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_ok)
            wins = int((correct & ~base_ok).sum())
            losses = int((base_ok & ~correct).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": correct.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:30s} acc={correct.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    real_row, null_row = rows[1], rows[2]
    if mode == "screen":
        passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
                  real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
        verdict = "ADVANCE TO CONFIRM" if passed else "REJECT"
    else:
        passed = real_row["gain_pt"] > 0.20 and real_row["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
    path = OUT / f"registered_atlas_{mode}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
