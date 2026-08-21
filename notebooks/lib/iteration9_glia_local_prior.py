"""Iteration 9f - glia-only atlas-composition decision prior.

The all-cell local-prior screen (partition 7) exposed a mechanistic interaction: beta=.25
improved glia by +0.51 point but reduced neurons by -0.75.  This was expected biologically
from the full-atlas homophily audit: glial compartments are spatially structured, whereas
neuronal subtypes are interdigitated.  `Region` missing is a deterministic, released
metadata definition of the 21-class non-neuronal branch, so the rule needs no predicted
gate and reads no label.

Because beta=.25 and the branch restriction were selected after seeing partition 7, that
partition is discovery only.  Confirmation uses partition 23 and replication uses 101,
both with 20-seed ExtraTrees.  Adopt only if gain > 0 and exact paired McNemar p < .05 on
BOTH untouched partitions.  Every neuron prediction must remain exactly equal to baseline.

Usage:
    python3 notebooks/lib/iteration9_glia_local_prior.py confirm
    python3 notebooks/lib/iteration9_glia_local_prior.py replicate
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_local_prior as L
import iteration9_quota as Q


OUT = Path("outputs/iteration9")
BETA = 0.25
ALPHA = 0.45


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "confirm"
    if mode == "confirm":
        partition_seed = 23
    elif mode == "replicate":
        partition_seed = 101
    else:
        raise SystemExit("mode must be 'confirm' or 'replicate'")

    _, meta, _, _ = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    feature_cache = np.load(L.FEATURE_CACHE, allow_pickle=True)
    X = np.hstack([
        feature_cache["BASE_TR"], feature_cache["EXT_TR"], feature_cache["SPA_TR"],
        feature_cache["NIC_TR"], feature_cache["ATL_TR"],
    ]).astype(np.float32)
    composition = np.load(L.COMPOSITION_CACHE, allow_pickle=True)["k10"][:len(y)]
    composition_all = np.load(L.COMPOSITION_CACHE, allow_pickle=True)["k10"]
    global_composition = np.clip(
        composition_all[:, :len(classes)].mean(axis=0), 1e-7, None
    )
    glia = meta["Region"].isna().to_numpy()
    baseline = np.empty(len(y), dtype=object)
    candidate = np.empty(len(y), dtype=object)

    t0 = time.time()
    folds = StratifiedKFold(5, shuffle=True, random_state=partition_seed).split(X, y)
    for fold, (train_rows, valid_rows) in enumerate(folds, start=1):
        probabilities = M.fit_extra_trees(
            X[train_rows], pd.Series(y[train_rows]), classes, X[valid_rows],
            seeds=tuple(range(20)),
        )
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[train_rows]), classes), ALPHA
        )
        allow = Q.compatibility_mask(
            meta.iloc[train_rows], y[train_rows], meta.iloc[valid_rows], classes
        )
        base_fold = class_array[np.where(allow, probabilities, -1.0).argmax(axis=1)]
        adjusted = L.adjust_with_local_prior(
            probabilities, composition[valid_rows], global_composition, BETA
        )
        adjusted_fold = class_array[np.where(allow, adjusted, -1.0).argmax(axis=1)]
        use_prior = glia[valid_rows]
        candidate_fold = base_fold.copy()
        candidate_fold[use_prior] = adjusted_fold[use_prior]
        baseline[valid_rows] = base_fold
        candidate[valid_rows] = candidate_fold
        print(f"fold {fold}/5 complete ({time.time()-t0:.0f}s)", flush=True)

    if not np.array_equal(candidate[~glia], baseline[~glia]):
        raise AssertionError("glia-only rule changed a neuronal prediction")
    base_ok, candidate_ok = baseline == y, candidate == y
    p_value, table = M.paired_mcnemar(candidate_ok, base_ok)
    result = pd.DataFrame([{
        "mode": mode, "partition_seed": partition_seed, "beta": BETA,
        "baseline_accuracy": base_ok.mean(),
        "candidate_accuracy": candidate_ok.mean(),
        "gain": candidate_ok.mean() - base_ok.mean(),
        "baseline_glia": base_ok[glia].mean(),
        "candidate_glia": candidate_ok[glia].mean(),
        "neuron_accuracy_unchanged": base_ok[~glia].mean(),
        "changed_glia": int((candidate[glia] != baseline[glia]).sum()),
        "baseline_only_correct": table[1][0],
        "candidate_only_correct": table[0][1],
        "mcnemar_p": p_value,
        "passes": bool(candidate_ok.mean() > base_ok.mean() and p_value < 0.05),
    }])
    stem = f"glia_local_prior_{mode}_partition{partition_seed}"
    result.to_csv(OUT / f"{stem}.csv", index=False)
    np.savez_compressed(OUT / f"{stem}_predictions.npz", truth=y,
                        baseline=baseline, candidate=candidate)
    print("\n" + result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
