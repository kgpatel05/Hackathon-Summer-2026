"""Iteration 12 - stratified subagging without replacement.

Bootstrap-0.80 retained only ~55% unique cells and failed.  This distinct estimator gives
each ensemble member a stratified 80% subset *without replacement*, preserving rare-class
coverage while decorrelating trees across row-level annotation noise.  Total tree count,
features, prior correction and hard metadata mask match the incumbent.

Screen is one pre-registered partition (seed 743), five ensemble members.  Advance only
for >0.30 point gain with exact paired McNemar p<0.05.  No test label is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_quota as Q
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
PARTITION = 743
SEEDS = tuple(range(5))
ALPHA = 0.45
FRACTION = 0.80


def stratified_subset(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        keep = max(1, int(round(fraction * len(members))))
        rows.extend(rng.choice(members, keep, replace=False))
    return np.asarray(sorted(rows), dtype=int)


def fit(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
        classes: list[str], subag: bool) -> np.ndarray:
    out = np.zeros((len(x_eval), len(classes)), np.float32)
    for seed in SEEDS:
        rows = stratified_subset(y_train, FRACTION, seed + 74300) if subag else np.arange(len(y_train))
        model = ExtraTreesClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        ).fit(x_train[rows], y_train[rows])
        out += M.align_proba(model, x_eval, classes)
    return out / len(SEEDS)


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y)); class_array = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    folds = StratifiedKFold(5, shuffle=True, random_state=PARTITION)
    predictions = {name: np.empty(len(y), object) for name in ("incumbent", "subag_0.80")}
    probabilities = {name: np.zeros((len(y), len(classes)), np.float32)
                     for name in predictions}
    t0 = time.time()
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        allow = Q.compatibility_mask(meta_train.iloc[train], y[train],
                                     meta_train.iloc[valid], classes)
        for name, subag in (("incumbent", False), ("subag_0.80", True)):
            p = fit(x[train], y[train], x[valid], classes, subag)
            p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
            probabilities[name][valid] = p
            predictions[name][valid] = class_array[np.where(allow, p, -1.0).argmax(1)]
        print(f"fold {fold}/5", flush=True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)

    base_ok = predictions["incumbent"] == y
    candidate_ok = predictions["subag_0.80"] == y
    p_value, _ = M.paired_mcnemar(candidate_ok, base_ok)
    wins = int((candidate_ok & ~base_ok).sum())
    losses = int((base_ok & ~candidate_ok).sum())
    gain_pt = 100 * (candidate_ok.mean() - base_ok.mean())
    frame = pd.DataFrame([
        {"config": "incumbent", "accuracy": base_ok.mean(), "gain_pt": 0.0,
         "wins": 0, "losses": 0, "p": 1.0},
        {"config": "subag_0.80", "accuracy": candidate_ok.mean(), "gain_pt": gain_pt,
         "wins": wins, "losses": losses, "p": p_value},
    ])
    frame.to_csv(OUT / "subsample_et_screen.csv", index=False)
    np.savez_compressed(OUT / "subsample_et_screen_oof.npz", y=y,
                        classes=class_array, **probabilities)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    passed = gain_pt > 0.30 and p_value < 0.05
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
