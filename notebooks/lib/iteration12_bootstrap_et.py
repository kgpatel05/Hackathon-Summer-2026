"""Iteration 12 - row-bagged ExtraTrees for noisy cell annotations.

The incumbent ExtraTrees uses ``bootstrap=False``: every tree sees every labelled cell.
That is a poor default when (a) atlas annotators agree with their consensus only 57-80%,
and (b) the challenge learning curve is flat above ~3,500 rows.  This experiment changes
one axis only: each tree receives a bootstrap sample of 80% of the fold-training rows.

Screen: partition 701, five estimator seeds.  Advance only for >0.30 point gain and exact
paired McNemar p<0.05.  Confirm: untouched partition 727, 20 seeds; adopt only for >0.20
point and p<0.05.  Features, leaf size, feature subsampling, prior correction and metadata
mask are unchanged.  No test label is read and the script never writes a submission.
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
ALPHA = 0.45
SCREEN_PARTITION, CONFIRM_PARTITION = 701, 727
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def probabilities(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                  classes: list[str], seeds: tuple[int, ...], bootstrap: bool) -> np.ndarray:
    out = np.zeros((len(x_eval), len(classes)), np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=bootstrap,
            max_samples=0.80 if bootstrap else None,
            n_jobs=-1,
            random_state=seed,
        ).fit(x_train, y_train)
        out += M.align_proba(model, x_eval, classes)
    return out / len(seeds)


def oof(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, classes: list[str],
        partition: int, seeds: tuple[int, ...], bootstrap: bool) -> tuple[np.ndarray, np.ndarray]:
    probs = np.zeros((len(y), len(classes)), np.float32)
    pred = np.empty(len(y), dtype=object)
    class_array = np.asarray(classes)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        p = probabilities(x[train], y[train], x[valid], classes, seeds, bootstrap)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        allow = Q.compatibility_mask(meta.iloc[train], y[train], meta.iloc[valid], classes)
        probs[valid] = p
        pred[valid] = class_array[np.where(allow, p, -1.0).argmax(1)]
        print(f"  {'bagged' if bootstrap else 'incumbent'} fold {fold}/5", flush=True)
    return probs, pred


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    print(f"mode={mode} partition={partition} features={x.shape[1]} seeds={len(seeds)}",
          flush=True)

    t0 = time.time()
    base_p, base_pred = oof(x, y, meta_train, classes, partition, seeds, False)
    bag_p, bag_pred = oof(x, y, meta_train, classes, partition, seeds, True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)
    base_ok, bag_ok = base_pred == y, bag_pred == y
    p_value, _ = M.paired_mcnemar(bag_ok, base_ok)
    wins = int((bag_ok & ~base_ok).sum())
    losses = int((base_ok & ~bag_ok).sum())
    gain_pt = 100 * (bag_ok.mean() - base_ok.mean())
    frame = pd.DataFrame([
        {"config": "incumbent", "accuracy": base_ok.mean(), "gain_pt": 0.0,
         "wins": 0, "losses": 0, "p": 1.0},
        {"config": "bootstrap_0.80", "accuracy": bag_ok.mean(), "gain_pt": gain_pt,
         "wins": wins, "losses": losses, "p": p_value},
    ])
    path = OUT / f"bootstrap_et_{mode}.csv"
    frame.to_csv(path, index=False)
    np.savez_compressed(OUT / f"bootstrap_et_{mode}_oof.npz", base=base_p, bagged=bag_p,
                        y=y, classes=np.asarray(classes), base_pred=base_pred,
                        bagged_pred=bag_pred)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    passed = (gain_pt > (0.30 if mode == "screen" else 0.20) and p_value < 0.05)
    verdict = ("ADVANCE TO CONFIRM" if mode == "screen" else "ADOPT") if passed else "REJECT"
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be screen or confirm")
    main(mode)
