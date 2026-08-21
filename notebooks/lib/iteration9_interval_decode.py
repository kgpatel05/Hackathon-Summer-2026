"""Iteration 9b - sampling-interval cohort decoder.

Iteration 9a rejected fixed train-derived quotas: even 25% shrinkage toward the labelled
histogram lost 0.36 point.  This follow-up is not a finer hyperparameter search.  It tests
one theoretically specified rule: keep the model's predicted class totals unless they
fall outside a 95% two-sample IID count interval (z=1.96), then project only to the nearest
interval boundary and reassign the minimum-cost cells jointly.

The first partition (seed 7) is a screen.  Adoption requires a positive gain there and on
an independent partition (seed 23); test truth remains sealed until both pass.

Usage:
    python3 notebooks/lib/iteration9_interval_decode.py screen
    python3 notebooks/lib/iteration9_interval_decode.py confirm
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_models as M
import iteration9_quota as Q
from iteration9_quota_decode import ALPHA, N_SPLITS, OUT, load_problem


Z_SCORE = 1.96


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        partition_seed, model_seeds = 7, tuple(range(5))
    elif mode == "confirm":
        partition_seed, model_seeds = 23, tuple(range(10, 20))
    else:
        raise SystemExit("mode must be 'screen' or 'confirm'")

    meta, y, classes, X = load_problem()
    class_array = np.asarray(classes)
    glia = meta["Region"].isna().to_numpy()
    baseline = np.empty(len(y), dtype=object)
    candidate = np.empty(len(y), dtype=object)

    import time
    t0 = time.time()
    for fold, (train_rows, valid_rows) in enumerate(
        KFold(N_SPLITS, shuffle=True, random_state=partition_seed).split(X), start=1
    ):
        probabilities = M.fit_extra_trees(
            X[train_rows], pd.Series(y[train_rows]), classes, X[valid_rows],
            seeds=model_seeds,
        )
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[train_rows]), classes), ALPHA
        )
        allowed = Q.compatibility_mask(
            meta.iloc[train_rows], y[train_rows], meta.iloc[valid_rows], classes
        )
        baseline[valid_rows] = class_array[
            np.where(allowed, probabilities, -1.0).argmax(axis=1)
        ]
        candidate[valid_rows] = class_array[Q.interval_decode(
            probabilities,
            meta.iloc[train_rows],
            y[train_rows],
            meta.iloc[valid_rows],
            classes,
            z_score=Z_SCORE,
        )]
        print(f"fold {fold}/{N_SPLITS} complete ({time.time()-t0:.0f}s)", flush=True)

    base_ok, candidate_ok = baseline == y, candidate == y
    p_value, table = M.paired_mcnemar(candidate_ok, base_ok)
    result = pd.DataFrame([{
        "mode": mode,
        "partition_seed": partition_seed,
        "z_score": Z_SCORE,
        "baseline_accuracy": base_ok.mean(),
        "candidate_accuracy": candidate_ok.mean(),
        "gain": candidate_ok.mean() - base_ok.mean(),
        "baseline_glia": base_ok[glia].mean(),
        "candidate_glia": candidate_ok[glia].mean(),
        "baseline_neuron": base_ok[~glia].mean(),
        "candidate_neuron": candidate_ok[~glia].mean(),
        "changed_cells": int((baseline != candidate).sum()),
        "baseline_only_correct": table[1][0],
        "candidate_only_correct": table[0][1],
        "mcnemar_p": p_value,
    }])
    result.to_csv(OUT / f"interval_{mode}_partition{partition_seed}.csv", index=False)
    np.savez_compressed(
        OUT / f"interval_{mode}_partition{partition_seed}_predictions.npz",
        truth=y, baseline=baseline, candidate=candidate,
    )
    print("\n" + result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
