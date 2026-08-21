"""Iteration 10 - ExtraTrees split geometry after expanding to 694 features.

The frozen ExtraTrees parameters were last retested at 529 columns.  The accepted atlas
context and nonlinear transfer added 165 columns, including 74 high-value probability
features, while ``max_features='sqrt'`` grew only from 23 to 26 columns per split.  This
screen tests whether each split now needs a larger view of the stack.

Pre-registered screen: partition seed 401, five estimator seeds, four candidates versus
the frozen sqrt/leaf-2 incumbent, Holm correction, and a >0.30-point effect floor.  The
single best passing candidate advances to untouched partition seed 419 with 20 seeds;
adopt only for >0.20 points and p<0.05.  No test label is read.

Usage:
    python3 notebooks/lib/iteration10_et_geometry.py screen
    python3 notebooks/lib/iteration10_et_geometry.py confirm
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
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN_RESULT = OUT / "et_geometry_screen.csv"
ALPHA = 0.45
SCREEN_PARTITION = 401
CONFIRM_PARTITION = 419
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
CONFIGS = {
    "sqrt_leaf2": ("sqrt", 2),
    "sqrt_leaf4": ("sqrt", 4),
    "mf0.1_leaf2": (0.1, 2),
    "mf0.1_leaf4": (0.1, 4),
    "mf0.2_leaf4": (0.2, 4),
}


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
                classes: list[str], seeds: tuple[int, ...], max_features,
                leaf: int) -> np.ndarray:
    total = np.zeros((len(x_valid), len(classes)), np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=600, max_features=max_features, min_samples_leaf=leaf,
            n_jobs=-1, random_state=seed,
        ).fit(x_train, y_train)
        total += M.align_proba(model, x_valid, classes)
    return total / len(seeds)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))

    if mode == "confirm":
        screen = pd.read_csv(SCREEN_RESULT)
        eligible = screen[(screen["config"] != "sqrt_leaf2") & screen["advance"]]
        if len(eligible) != 1:
            raise RuntimeError("confirmation requires exactly one advanced screen candidate")
        names = ["sqrt_leaf2", str(eligible.iloc[0]["config"])]
        partition, seeds = CONFIRM_PARTITION, CONFIRM_SEEDS
    else:
        names = list(CONFIGS)
        partition, seeds = SCREEN_PARTITION, SCREEN_SEEDS
    print(f"mode={mode} partition={partition} x={x.shape} seeds={len(seeds)} "
          f"configs={names}", flush=True)

    results: dict[str, np.ndarray] = {}
    for name in names:
        max_features, leaf = CONFIGS[name]
        correct = np.zeros(len(y), bool)
        t0 = time.time()
        folds = StratifiedKFold(5, shuffle=True, random_state=partition)
        for train, valid in folds.split(y, y):
            probs = fit_predict(
                x[train], y[train], x[valid], classes, seeds, max_features, leaf
            )
            probs = M.correct_prior(
                probs, M.prior_vector(pd.Series(y[train]), classes), ALPHA
            )
            correct[valid] = class_array[probs.argmax(1)] == y[valid]
        results[name] = correct
        print(f"finished {name:14s} acc={correct.mean():.4f} "
              f"in {time.time()-t0:.1f}s", flush=True)

    base = results["sqrt_leaf2"]
    rows = []
    raw_p = []
    for name in names:
        ok = results[name]
        gain = ok.mean() - base.mean()
        if name == "sqrt_leaf2":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base)
            raw_p.append(p_value)
            wins = int((ok & ~base).sum())
            losses = int((base & ~ok).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": ok.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})

    if mode == "screen":
        adjusted = holm_adjust(raw_p)
        for row, p_adjusted in zip(rows[1:], adjusted):
            row["p_holm"] = p_adjusted
            row["passes"] = row["gain_pt"] > 0.30 and p_adjusted < 0.05
        rows[0].update({"p_holm": 1.0, "passes": False})
        passing = [row for row in rows if row["passes"]]
        winner = max(passing, key=lambda row: row["gain_pt"])["config"] if passing else None
        for row in rows:
            row["advance"] = row["config"] == winner
        verdict = f"ADVANCE {winner}" if winner else "REJECT ALL"
        path = SCREEN_RESULT
    else:
        for row in rows:
            row["advance"] = False
        candidate = rows[1]
        passed = candidate["gain_pt"] > 0.20 and candidate["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
        path = OUT / "et_geometry_confirm.csv"

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False), flush=True)
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
