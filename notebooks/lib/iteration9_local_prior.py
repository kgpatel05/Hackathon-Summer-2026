"""Iteration 9e - atlas-neighbour composition as a Bayesian local prior.

Iteration 9's composition *feature* produced a consistent +0.52 / +0.38 point on two
fold partitions, but confirmation p=0.067 missed its pre-registered adoption threshold.
That form asks ExtraTrees to discover how 61 new columns should reweight already-good
class probabilities.  Here the mechanism is encoded directly.

For each cell, the k=10 external-neighbour histogram is smoothed with ten pseudo-neighbours
drawn from the global atlas composition.  The ratio of this local posterior composition to
the global composition is a spatial likelihood ratio.  Frozen ExtraTrees probabilities
are multiplied by ratio**beta and renormalised before the adopted metadata mask.

Registered screen: beta in {0.25, 0.50, 1.00}, tau=10, partition seed 7, five ET seeds;
Holm correction across the three betas and a within-section row-shuffled composition null
at the winning beta.  Confirmation: one winning beta, partition 23, 20 ET seeds.  Adopt
only if the screen passes Holm and beats its null, then confirmation gain > 0 and p < .05.
No challenge test label or withheld gene is read.

Usage:
    python3 notebooks/lib/iteration9_local_prior.py screen
    python3 notebooks/lib/iteration9_local_prior.py confirm
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_freshidea as C
import iteration9_quota as Q


OUT = Path("outputs/iteration9")
FEATURE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
COMPOSITION_CACHE = OUT / "atlas_composition_cache.npz"
SCREEN_JSON = OUT / "local_prior_screen.json"
BETAS = (0.25, 0.50, 1.00)
TAU = 10.0
K_NEIGHBOURS = 10.0
ALPHA = 0.45


def adjust_with_local_prior(probabilities, composition, global_composition, beta):
    local = composition[:, :probabilities.shape[1]].astype(float)
    global_composition = np.clip(np.asarray(global_composition, float), 1e-7, None)
    smoothed = (
        K_NEIGHBOURS * local + TAU * global_composition[None, :]
    ) / (K_NEIGHBOURS + TAU)
    enrichment = np.clip(smoothed / global_composition[None, :], 1e-3, 1e3)
    adjusted = probabilities * enrichment**beta
    total = adjusted.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return adjusted / total


def holm_pass(p_values, alpha=0.05):
    """Return a boolean per hypothesis under Holm's step-down procedure."""
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    passed = np.zeros(len(p_values), dtype=bool)
    still_passing = True
    for rank, index in enumerate(order):
        still_passing &= p_values[index] <= alpha / (len(p_values) - rank)
        passed[index] = still_passing
    return passed


def run_partition(partition_seed, model_seeds, betas, include_null=False):
    _, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    cache = np.load(FEATURE_CACHE, allow_pickle=True)
    X = np.hstack([
        cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"], cache["NIC_TR"],
        cache["ATL_TR"],
    ]).astype(np.float32)
    comp_cache = np.load(COMPOSITION_CACHE, allow_pickle=True)
    key = "K10" if "K10" in comp_cache.files else "k10"
    composition_all = comp_cache[key]
    composition = composition_all[:len(y)]
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    shuffled = C.shuffled_null(composition_all, meta_all)[:len(y)]
    global_composition = np.clip(composition_all[:, :len(classes)].mean(axis=0), 1e-7, None)

    predictions = {"baseline": np.empty(len(y), dtype=object)}
    for beta in betas:
        predictions[f"beta_{beta:.2f}"] = np.empty(len(y), dtype=object)
    if include_null:
        if len(betas) != 1:
            raise ValueError("null is evaluated only at one selected beta")
        predictions["shuffled_null"] = np.empty(len(y), dtype=object)

    t0 = time.time()
    folds = StratifiedKFold(5, shuffle=True, random_state=partition_seed).split(X, y)
    for fold, (train_rows, valid_rows) in enumerate(folds, start=1):
        probabilities = M.fit_extra_trees(
            X[train_rows], pd.Series(y[train_rows]), classes, X[valid_rows],
            seeds=model_seeds,
        )
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[train_rows]), classes), ALPHA
        )
        allow = Q.compatibility_mask(
            meta_train.iloc[train_rows], y[train_rows], meta_train.iloc[valid_rows], classes
        )
        predictions["baseline"][valid_rows] = class_array[
            np.where(allow, probabilities, -1.0).argmax(axis=1)
        ]
        for beta in betas:
            adjusted = adjust_with_local_prior(
                probabilities, composition[valid_rows], global_composition, beta
            )
            predictions[f"beta_{beta:.2f}"][valid_rows] = class_array[
                np.where(allow, adjusted, -1.0).argmax(axis=1)
            ]
        if include_null:
            adjusted = adjust_with_local_prior(
                probabilities, shuffled[valid_rows], global_composition, betas[0]
            )
            predictions["shuffled_null"][valid_rows] = class_array[
                np.where(allow, adjusted, -1.0).argmax(axis=1)
            ]
        print(f"fold {fold}/5 complete ({time.time()-t0:.0f}s)", flush=True)
    return meta_train, y, predictions


def summarise(mode, partition_seed, meta, y, predictions):
    glia = meta["Region"].isna().to_numpy()
    baseline = predictions["baseline"]
    baseline_ok = baseline == y
    rows = []
    for name, prediction in predictions.items():
        correct = prediction == y
        p_value, table = M.paired_mcnemar(correct, baseline_ok)
        rows.append({
            "mode": mode, "partition_seed": partition_seed, "variant": name,
            "accuracy": correct.mean(), "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "gain": correct.mean() - baseline_ok.mean(),
            "changed_cells": int((prediction != baseline).sum()),
            "baseline_only_correct": table[1][0],
            "candidate_only_correct": table[0][1], "mcnemar_p": p_value,
        })
    return pd.DataFrame(rows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        partition_seed, model_seeds, betas = 7, tuple(range(5)), BETAS
        meta, y, predictions = run_partition(partition_seed, model_seeds, betas)
        results = summarise(mode, partition_seed, meta, y, predictions)
        candidates = results[results.variant.str.startswith("beta_")].copy()
        candidates["holm_pass"] = holm_pass(candidates.mcnemar_p.to_numpy())
        winner = candidates.sort_values(["gain", "mcnemar_p"], ascending=[False, True]).iloc[0]
        selected_beta = float(winner.variant.split("_")[1])

        # The null reuses an independent baseline fit only for code simplicity; its
        # prediction comparison is deterministic at these fixed estimator seeds.
        meta, y, with_null = run_partition(
            partition_seed, model_seeds, (selected_beta,), include_null=True
        )
        null_row = summarise(mode, partition_seed, meta, y, with_null)
        null_row = null_row[null_row.variant == "shuffled_null"]
        results = pd.concat([results, null_row], ignore_index=True)
        null_gain = float(null_row.gain.iloc[0])
        proceed = bool(winner.holm_pass and winner.gain > 0 and winner.gain > null_gain)
        SCREEN_JSON.write_text(json.dumps({
            "selected_beta": selected_beta,
            "winner_gain": float(winner.gain),
            "winner_p": float(winner.mcnemar_p),
            "winner_holm_pass": bool(winner.holm_pass),
            "null_gain": null_gain,
            "proceed_to_confirmation": proceed,
        }, indent=2))
        print(f"\nselected beta={selected_beta:.2f}; Holm={bool(winner.holm_pass)}; "
              f"null gain={null_gain:+.5f}; proceed={proceed}")
    elif mode == "confirm":
        decision = json.loads(SCREEN_JSON.read_text())
        if not decision["proceed_to_confirmation"]:
            raise SystemExit("screen did not authorise confirmation")
        betas = (float(decision["selected_beta"]),)
        partition_seed, model_seeds = 23, tuple(range(20))
        meta, y, predictions = run_partition(partition_seed, model_seeds, betas)
        results = summarise(mode, partition_seed, meta, y, predictions)
    else:
        raise SystemExit("mode must be 'screen' or 'confirm'")

    stem = f"local_prior_{mode}_partition{partition_seed}"
    results.to_csv(OUT / f"{stem}.csv", index=False)
    print("\n" + results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
