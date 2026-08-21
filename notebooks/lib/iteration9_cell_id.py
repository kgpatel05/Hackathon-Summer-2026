"""Iteration 9c - recover acquisition structure from the provided Cell_ID.

The 19-digit challenge IDs are not random UUIDs.  They decompose into an 8-digit
acquisition prefix, a four-digit field-of-view-like value, a one-digit index, and a
six-digit object code.  All prefix and nearly all field values occur in both train and
test.  The current model discards the ID even though it is explicitly supplied.

This experiment asks whether ID structure contains acquisition/spatial information not
already captured by Section_ID and center_x/center_y.  It compares compact numeric digits,
coarse categorical components, their combination, and a row-permuted combined null with
identical dimensionality.  The permutation is fixed before CV and never touches labels.

Pre-registered screen: one 5-fold OOF prediction per cell, partition seed 7, five model
seeds.  A candidate must improve on the baseline, beat the permuted null, and then retain
positive gain on partition seed 23 with independent model seeds before test evaluation.

Usage:
    python3 notebooks/lib/iteration9_cell_id.py screen
    python3 notebooks/lib/iteration9_cell_id.py confirm combined
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_quota as Q


OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
ALPHA = 0.45


def id_blocks(train_index, test_index):
    ids = np.concatenate([
        pd.Index(train_index).astype(str).to_numpy(),
        pd.Index(test_index).astype(str).to_numpy(),
    ])
    if any(len(value) != 19 or not value.isdigit() for value in ids):
        raise ValueError("expected every Cell_ID to contain exactly 19 digits")

    digits = np.asarray([[int(char) for char in value] for value in ids], np.float32)
    numeric = np.column_stack([
        digits,
        np.asarray([int(value[8:12]) for value in ids], np.float32) / 1000.0,
        np.asarray([int(value[8:13]) for value in ids], np.float32) / 10000.0,
        np.asarray([int(value[13:16]) for value in ids], np.float32) / 100.0,
        np.asarray([int(value[16:19]) for value in ids], np.float32) / 1000.0,
    ])

    # Relative order within acquisition and within field captures segmentation order
    # without treating a 19-digit identifier as an imprecise floating-point scalar.
    components = pd.DataFrame({
        "prefix8": [value[:8] for value in ids],
        "fov4": [value[8:12] for value in ids],
        "index1": [value[12] for value in ids],
        "object2": [value[13:15] for value in ids],
        "object3": [value[13:16] for value in ids],
        "tail6": [int(value[13:19]) for value in ids],
    })
    rank_acquisition = components.groupby("prefix8")["tail6"].rank(pct=True).to_numpy()
    rank_field = components.groupby(["prefix8", "fov4"])["tail6"].rank(pct=True).to_numpy()
    numeric = np.column_stack([numeric, rank_acquisition, rank_field]).astype(np.float32)

    categorical_frame = components[["prefix8", "fov4", "index1", "object2", "object3"]]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical = encoder.fit_transform(categorical_frame).astype(np.float32)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "combined": np.hstack([numeric, categorical]).astype(np.float32),
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    selected = sys.argv[2] if mode == "confirm" and len(sys.argv) > 2 else "combined"

    _, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    cache = np.load(CACHE, allow_pickle=True)
    base = np.hstack([
        cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"], cache["NIC_TR"],
        cache["ATL_TR"],
    ]).astype(np.float32)
    blocks = id_blocks(meta_train.index, meta_test.index)
    rng = np.random.default_rng(99173)
    permutation = rng.permutation(len(y))

    if mode == "screen":
        variants = {
            "baseline": base,
            "id_numeric": np.hstack([base, blocks["numeric"][:len(y)]]),
            "id_categorical": np.hstack([base, blocks["categorical"][:len(y)]]),
            "id_combined": np.hstack([base, blocks["combined"][:len(y)]]),
            "permuted_combined_id_null": np.hstack([
                base, blocks["combined"][:len(y)][permutation]
            ]),
        }
        partition_seed, model_seeds = 7, tuple(range(5))
    else:
        if selected not in blocks:
            raise SystemExit(f"unknown ID block: {selected}")
        variants = {
            "baseline": base,
            f"id_{selected}": np.hstack([base, blocks[selected][:len(y)]]),
            "permuted_id_null": np.hstack([
                base, blocks[selected][:len(y)][permutation]
            ]),
        }
        partition_seed, model_seeds = 23, tuple(range(10, 20))

    splits = list(StratifiedKFold(
        5, shuffle=True, random_state=partition_seed
    ).split(base, y))
    glia = meta_train["Region"].isna().to_numpy()
    predictions = {name: np.empty(len(y), dtype=object) for name in variants}
    t0 = time.time()

    for name, matrix in variants.items():
        for fold, (train_rows, valid_rows) in enumerate(splits, start=1):
            probabilities = M.fit_extra_trees(
                matrix[train_rows], pd.Series(y[train_rows]), classes,
                matrix[valid_rows], seeds=model_seeds,
            )
            probabilities = M.correct_prior(
                probabilities, M.prior_vector(pd.Series(y[train_rows]), classes), ALPHA
            )
            allow = Q.compatibility_mask(
                meta_train.iloc[train_rows], y[train_rows], meta_train.iloc[valid_rows],
                classes,
            )
            predictions[name][valid_rows] = class_array[
                np.where(allow, probabilities, -1.0).argmax(axis=1)
            ]
        print(f"{name}: 5 folds complete ({time.time()-t0:.0f}s)", flush=True)

    baseline_ok = predictions["baseline"] == y
    rows = []
    for name, prediction in predictions.items():
        correct = prediction == y
        p_value, table = M.paired_mcnemar(correct, baseline_ok)
        rows.append({
            "mode": mode,
            "partition_seed": partition_seed,
            "variant": name,
            "n_features": variants[name].shape[1],
            "accuracy": correct.mean(),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "gain": correct.mean() - baseline_ok.mean(),
            "changed_cells": int((prediction != predictions["baseline"]).sum()),
            "baseline_only_correct": table[1][0],
            "candidate_only_correct": table[0][1],
            "mcnemar_p": p_value,
        })
    results = pd.DataFrame(rows)
    stem = f"cell_id_{mode}_partition{partition_seed}"
    results.to_csv(OUT / f"{stem}.csv", index=False)
    np.savez_compressed(OUT / f"{stem}_predictions.npz", truth=y, **predictions)
    print("\n" + results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
