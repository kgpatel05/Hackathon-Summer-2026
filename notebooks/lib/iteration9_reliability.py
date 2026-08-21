"""Iteration 9d - reference-informed robust learning under annotation noise.

The target is itself a consensus of automated annotation methods, with only 57-80%
agreement among constituent callers.  The final ExtraTrees model nevertheless gives every
training label identical weight.  Two independent external transfer models (SNI and the
non-challenge parent atlas) provide a label-free way to identify cells that are atypical
for their assigned class in the released 200-gene space.

For each labelled cell, reliability is the geometric mean probability that the two
external models assign to its challenge label.  Scores are percentile-ranked *within
class* so difficult/rare classes are not globally downweighted.  The resulting weight is
0.5 + percentile (approximately 0.5..1.5 with class mean 1).  Classes below ten cells are
left at weight 1 because their rank is unstable.

The matched null permutes these exact weights within class.  It preserves every class's
total weight and weight distribution, isolating whether the external evidence identifies
better training examples.  Screen uses fold partition 7/model seeds 0..4; confirmation
uses partition 23/model seeds 10..19.  No validation or test label enters a weight.

Usage:
    python3 notebooks/lib/iteration9_reliability.py screen
    python3 notebooks/lib/iteration9_reliability.py confirm
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


OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
ALPHA = 0.45


def reliability_weights(y, classes, external, atlas):
    class_index = {label: k for k, label in enumerate(classes)}
    true_column = np.asarray([class_index[label] for label in y])
    rows = np.arange(len(y))
    score = 0.5 * (
        np.log(np.clip(external[rows, true_column], 1e-8, 1.0))
        + np.log(np.clip(atlas[rows, true_column], 1e-8, 1.0))
    )
    weights = np.ones(len(y), dtype=np.float32)
    for label in classes:
        label_rows = np.flatnonzero(y == label)
        if len(label_rows) < 10:
            continue
        ranks = pd.Series(score[label_rows]).rank(method="average", pct=True).to_numpy()
        raw = 0.5 + ranks
        weights[label_rows] = raw / raw.mean()
    return weights


def fit_weighted(X_train, y_train, weights, classes, X_eval, seeds):
    probabilities = np.zeros((len(X_eval), len(classes)), dtype=np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **M.ET_KWARGS)
        model.fit(X_train, y_train, sample_weight=weights)
        probabilities += M.align_proba(model, X_eval, classes)
    return probabilities / len(seeds)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        partition_seed, model_seeds = 7, tuple(range(5))
        baseline_file = OUT / "cell_id_screen_partition7_predictions.npz"
    elif mode == "confirm":
        partition_seed, model_seeds = 23, tuple(range(10, 20))
        baseline_file = OUT / "cell_id_confirm_partition23_predictions.npz"
    else:
        raise SystemExit("mode must be 'screen' or 'confirm'")

    _, meta, _, _ = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    cache = np.load(CACHE, allow_pickle=True)
    X = np.hstack([
        cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"], cache["NIC_TR"],
        cache["ATL_TR"],
    ]).astype(np.float32)
    weights = reliability_weights(y, classes, cache["EXT_TR"], cache["ATL_TR"])

    rng = np.random.default_rng(28411)
    null_weights = weights.copy()
    for label in classes:
        label_rows = np.flatnonzero(y == label)
        null_weights[label_rows] = weights[rng.permutation(label_rows)]

    if not baseline_file.exists():
        raise FileNotFoundError(
            f"{baseline_file} is required so the identical frozen baseline is reused"
        )
    baseline = np.load(baseline_file, allow_pickle=True)["baseline"].astype(str)
    predictions = {"baseline": baseline}
    splits = list(StratifiedKFold(
        5, shuffle=True, random_state=partition_seed
    ).split(X, y))
    t0 = time.time()
    for name, sample_weights in {
        "reference_reliability": weights,
        "within_class_permuted_null": null_weights,
    }.items():
        prediction = np.empty(len(y), dtype=object)
        for train_rows, valid_rows in splits:
            probabilities = fit_weighted(
                X[train_rows], pd.Series(y[train_rows]), sample_weights[train_rows],
                classes, X[valid_rows], model_seeds,
            )
            probabilities = M.correct_prior(
                probabilities, M.prior_vector(pd.Series(y[train_rows]), classes), ALPHA
            )
            allow = Q.compatibility_mask(
                meta.iloc[train_rows], y[train_rows], meta.iloc[valid_rows], classes
            )
            prediction[valid_rows] = class_array[
                np.where(allow, probabilities, -1.0).argmax(axis=1)
            ]
        predictions[name] = prediction
        print(f"{name}: complete ({time.time()-t0:.0f}s)", flush=True)

    glia = meta["Region"].isna().to_numpy()
    baseline_ok = baseline == y
    rows = []
    for name, prediction in predictions.items():
        correct = prediction == y
        p_value, table = M.paired_mcnemar(correct, baseline_ok)
        rows.append({
            "mode": mode,
            "partition_seed": partition_seed,
            "variant": name,
            "accuracy": correct.mean(),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "gain": correct.mean() - baseline_ok.mean(),
            "changed_cells": int((prediction != baseline).sum()),
            "baseline_only_correct": table[1][0],
            "candidate_only_correct": table[0][1],
            "mcnemar_p": p_value,
        })
    results = pd.DataFrame(rows)
    stem = f"reliability_{mode}_partition{partition_seed}"
    results.to_csv(OUT / f"{stem}.csv", index=False)
    np.savez_compressed(OUT / f"{stem}_predictions.npz", truth=y, **predictions)
    print("\nweight summary:")
    print(pd.Series(weights).describe().to_string())
    print("\n" + results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
