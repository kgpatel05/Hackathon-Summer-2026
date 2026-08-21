"""Iteration 9a - optimal cohort decoding with train-derived soft class quotas.

WHY THIS IS DIFFERENT
---------------------
Every model through Iteration 8 predicts each cell independently.  Yet the challenge
train and test sets are equal-sized IID halves of the same 108 tissue sections.  The
labelled half therefore contains information about the *cohort composition* of the
unlabelled half, even though it contains no information about any test cell's answer.

Iteration 8b found the corresponding decision failure: several rare dorsal-horn classes
have one-vs-rest AUC 0.95-0.99, but probability shrinkage keeps them below common classes
at argmax.  Raising rare-class probabilities independently caused too many false
positives.  A joint assignment can recover the strongest rare candidates while capping
how many are selected.

HONESTY AND PRE-REGISTRATION
----------------------------
* no validation/test label enters a quota or metadata mask;
* folds use shuffled KFold, NOT stratification, so validation class counts fluctuate as
  they do in the real IID split;
* probabilities are the frozen 529-feature, alpha=0.45 Extra Trees stack;
* screen strengths are fixed at 0.25, 0.50, 0.75 and 1.00;
* strength 0.00 is an exact masked-argmax control;
* select the largest screen gain, then require positive gain on a second fold partition;
* only a confirmed candidate may be fitted to all training rows and test-labelled once.

The 5-fold validation/train size ratio is 1:4 rather than the final 1:1 ratio.  Therefore
the confirmation also reports count error by class and the chosen strength must be less
than 1 unless the hard quota wins both partitions.  This guards against treating the
training class histogram as if it were the unknown test histogram.

Usage:
    python3 notebooks/lib/iteration9_quota_decode.py screen
    python3 notebooks/lib/iteration9_quota_decode.py confirm 0.50
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_quota as Q


OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
SCREEN_STRENGTHS = (0.0, 0.25, 0.50, 0.75, 1.00)
ALPHA = 0.45
N_SPLITS = 5


def load_problem():
    _, meta_train, _, _ = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    cache = np.load(FEATURE_CACHE, allow_pickle=True)
    X = np.hstack([
        cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"], cache["NIC_TR"],
        cache["ATL_TR"],
    ]).astype(np.float32)
    return meta_train, y, classes, X


def evaluate_partition(partition_seed: int, strengths, model_seeds):
    meta, y, classes, X = load_problem()
    class_array = np.asarray(classes)
    glia = meta["Region"].isna().to_numpy()
    folds = KFold(N_SPLITS, shuffle=True, random_state=partition_seed).split(X)
    predictions = {strength: np.empty(len(y), dtype=object) for strength in strengths}
    fold_rows = []
    t0 = time.time()

    for fold, (train_rows, valid_rows) in enumerate(folds, start=1):
        probabilities = M.fit_extra_trees(
            X[train_rows], pd.Series(y[train_rows]), classes, X[valid_rows],
            seeds=model_seeds,
        )
        probabilities = M.correct_prior(
            probabilities,
            M.prior_vector(pd.Series(y[train_rows]), classes),
            ALPHA,
        )
        for strength in strengths:
            decoded = Q.quota_decode(
                probabilities,
                meta.iloc[train_rows],
                y[train_rows],
                meta.iloc[valid_rows],
                classes,
                strength,
            )
            predictions[strength][valid_rows] = class_array[decoded]
        fold_rows.append({
            "partition_seed": partition_seed,
            "fold": fold,
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "seconds_elapsed": time.time() - t0,
        })
        print(f"fold {fold}/{N_SPLITS} complete ({time.time()-t0:.0f}s)", flush=True)

    baseline_correct = predictions[0.0] == y
    rows = []
    for strength in strengths:
        correct = predictions[strength] == y
        p_value, table = M.paired_mcnemar(correct, baseline_correct)
        rows.append({
            "partition_seed": partition_seed,
            "strength": strength,
            "accuracy": correct.mean(),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "gain": correct.mean() - baseline_correct.mean(),
            "changed_cells": int((predictions[strength] != predictions[0.0]).sum()),
            "mcnemar_p_vs_baseline": p_value,
            "baseline_only_correct": table[1][0],
            "candidate_only_correct": table[0][1],
        })
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), predictions


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        strengths = SCREEN_STRENGTHS
        partition_seed = 7
        model_seeds = tuple(range(5))
    elif mode == "confirm":
        if len(sys.argv) != 3:
            raise SystemExit("usage: iteration9_quota_decode.py confirm STRENGTH")
        strengths = (0.0, float(sys.argv[2]))
        partition_seed = 23
        model_seeds = tuple(range(10, 20))
    else:
        raise SystemExit("mode must be 'screen' or 'confirm'")

    results, folds, predictions = evaluate_partition(
        partition_seed, strengths, model_seeds
    )
    stem = f"quota_{mode}_partition{partition_seed}"
    results.to_csv(OUT / f"{stem}.csv", index=False)
    folds.to_csv(OUT / f"{stem}_folds.csv", index=False)
    np.savez_compressed(
        OUT / f"{stem}_predictions.npz",
        truth=load_problem()[1],
        **{f"strength_{strength:.2f}": value for strength, value in predictions.items()},
    )
    print("\n" + results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if mode == "screen":
        candidate = results.loc[results.strength > 0].sort_values(
            ["gain", "strength"], ascending=[False, True]
        ).iloc[0]
        decision = {
            "screen_partition": partition_seed,
            "selected_strength": float(candidate.strength),
            "screen_gain": float(candidate.gain),
            "screen_p": float(candidate.mcnemar_p_vs_baseline),
            "confirmation_required": bool(candidate.gain > 0),
        }
        (OUT / "quota_screen_decision.json").write_text(json.dumps(decision, indent=2))
        verdict = "confirmation required" if candidate.gain > 0 else "screen rejected"
        print(f"\nselected strength={candidate.strength:.2f}; "
              f"gain={candidate.gain:+.5f}; {verdict}", flush=True)


if __name__ == "__main__":
    main()
