"""Iteration 10 - CatBoost model-family complementarity on the 694-feature stack.

XGBoost and sklearn HGB were tested on the older 529-column stack and lost.  CatBoost's
ordered boosting, symmetric trees and stronger categorical-style regularisation are a
different boosting regime and have not been tested in this repository.  This script
tests one frozen numeric configuration plus one fixed probability blend:

    0.80 * adopted ExtraTrees + 0.20 * CatBoost

There is deliberately no blend-weight grid.  Screen uses stratified partition seed 307,
five ET seeds and one CatBoost seed.  Advance only if the fixed blend gains >0.30 points
and paired exact McNemar p<0.05.  Confirmation uses untouched partition seed 331, 20 ET
seeds and three averaged CatBoost seeds; adopt only for >0.20 points and p<0.05.  This
script never reads hidden test labels.

Usage:
    python3 notebooks/lib/iteration10_catboost.py screen
    python3 notebooks/lib/iteration10_catboost.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Import the repository's numerical stack first.  CatBoost was installed into an
# ignored target directory that also contains wheel dependencies; keeping already-loaded
# NumPy/pandas avoids shadowing the environment used by the rest of the repository.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.45
BLEND_WEIGHT = 0.20
SCREEN_PARTITION = 307
CONFIRM_PARTITION = 331
SCREEN_ET_SEEDS = tuple(range(5))
CONFIRM_ET_SEEDS = tuple(range(20))
SCREEN_CAT_SEEDS = (0,)
CONFIRM_CAT_SEEDS = (0, 1, 2)


def cat_probabilities(x_train: np.ndarray, y_train: np.ndarray,
                      x_valid: np.ndarray, classes: list[str],
                      seeds: tuple[int, ...]) -> np.ndarray:
    total = np.zeros((len(x_valid), len(classes)), np.float32)
    index = {name: i for i, name in enumerate(classes)}
    for seed in seeds:
        model = CatBoostClassifier(
            iterations=700,
            depth=7,
            learning_rate=0.05,
            loss_function="MultiClass",
            l2_leaf_reg=6.0,
            random_strength=1.0,
            bootstrap_type="Bayesian",
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
        raw = model.predict_proba(x_valid)
        aligned = np.zeros_like(total)
        for j, name in enumerate(model.classes_.astype(str)):
            aligned[:, index[name]] = raw[:, j]
        total += aligned
    return total / len(seeds)


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    et_seeds = SCREEN_ET_SEEDS if mode == "screen" else CONFIRM_ET_SEEDS
    cat_seeds = SCREEN_CAT_SEEDS if mode == "screen" else CONFIRM_CAT_SEEDS
    print(f"mode={mode} partition={partition} x={x.shape} ET={len(et_seeds)} "
          f"CatBoost={len(cat_seeds)} blend={BLEND_WEIGHT:.2f}", flush=True)

    et_oof = np.zeros((len(y), len(classes)), np.float32)
    cat_oof = np.zeros_like(et_oof)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        t0 = time.time()
        et = M.fit_extra_trees(
            x[train], pd.Series(y[train]), classes, x[valid], seeds=et_seeds
        )
        et_oof[valid] = M.correct_prior(
            et, M.prior_vector(pd.Series(y[train]), classes), ALPHA
        )
        cat_oof[valid] = cat_probabilities(
            x[train], y[train], x[valid], classes, cat_seeds
        )
        print(f"fold {fold}/5 finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et_oof + BLEND_WEIGHT * cat_oof
    configurations = {
        "ExtraTrees incumbent": et_oof,
        "CatBoost standalone": cat_oof,
        "0.80 ET + 0.20 CatBoost": blend,
    }
    base_ok = class_array[et_oof.argmax(1)] == y
    rows = []
    for name, probabilities in configurations.items():
        correct = class_array[probabilities.argmax(1)] == y
        gain = correct.mean() - base_ok.mean()
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_ok)
            wins = int((correct & ~base_ok).sum())
            losses = int((base_ok & ~correct).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": correct.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:28s} acc={correct.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    candidate = rows[2]
    if mode == "screen":
        passed = candidate["gain_pt"] > 0.30 and candidate["p"] < 0.05
        verdict = "ADVANCE TO CONFIRM" if passed else "REJECT"
    else:
        passed = candidate["gain_pt"] > 0.20 and candidate["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
    path = OUT / f"catboost_{mode}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    np.savez_compressed(OUT / f"catboost_{mode}_oof.npz", et=et_oof, cat=cat_oof,
                        y=y, classes=np.asarray(classes))
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
