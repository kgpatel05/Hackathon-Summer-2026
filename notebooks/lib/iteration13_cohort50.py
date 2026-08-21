"""Iteration 13 - cohort decoding under a challenge-matched 50/50 protocol.

The Iteration-9 quota experiment used five-fold validation: 4,000 labelled cells were
used to infer the composition of only 1,000 validation cells.  Production has the
opposite statistical question: infer a 5,000-cell cohort from an equal-sized labelled
cohort.  This script corrects that ratio without assuming that train/test were label
stratified.

Three fixed, non-stratified shuffled two-fold partitions produce one OOF prediction per
cell and partition.  Splits with a missing training class are deterministically advanced
to the next seed; this mirrors the released training set, which contains all 60 labels,
without forcing the two halves to have equal class counts.  Each fold fits the adopted
694-feature ExtraTrees posterior once.  Four fixed decoders are then compared:

* 25% soft quota, all cells;
* 25% soft quota, glia only (``Region`` is missing, an observed deterministic gate);
* 95% IID count interval, all cells;
* 95% IID count interval, glia only.

Advance only if a decoder gains >0 in every partition, averages >0.30 point, and its
median exact McNemar p passes Holm correction across the four decoder hypotheses.  No
test label or recovered-label artifact is read; no submission is written.
"""
from __future__ import annotations

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
from iteration10_spatial_fields import current_stack


OUT = Path("outputs/iteration13")
OUT.mkdir(parents=True, exist_ok=True)
PARTITIONS = (1741, 1777, 1811)
ET_SEEDS = tuple(range(5))
ALPHA = 0.45


def complete_twofold(y: np.ndarray, seed: int):
    """Return a non-stratified 50/50 split with all labels present in each train half."""
    required = set(y)
    candidate = seed
    while True:
        folds = list(KFold(2, shuffle=True, random_state=candidate).split(y))
        if all(set(y[train]) == required for train, _ in folds):
            return candidate, folds
        candidate += 1


def decode_variants(
    probabilities: np.ndarray,
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: list[str],
) -> dict[str, np.ndarray]:
    allow = Q.compatibility_mask(meta_train, y_train, meta_eval, classes)
    baseline = np.where(allow, probabilities, -1.0).argmax(axis=1)
    quota = Q.quota_decode(
        probabilities, meta_train, y_train, meta_eval, classes, strength=0.25
    )
    interval = Q.interval_decode(
        probabilities, meta_train, y_train, meta_eval, classes, z_score=1.96
    )
    glia = meta_eval["Region"].isna().to_numpy()
    quota_glia = baseline.copy()
    quota_glia[glia] = quota[glia]
    interval_glia = baseline.copy()
    interval_glia[glia] = interval[glia]
    return {
        "baseline": baseline,
        "quota_025_all": quota,
        "quota_025_glia": quota_glia,
        "interval_196_all": interval,
        "interval_196_glia": interval_glia,
    }


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts.columns))
    glia = meta["Region"].isna().to_numpy()
    rows = []
    t0 = time.time()

    for requested_seed in PARTITIONS:
        used_seed, folds = complete_twofold(y, requested_seed)
        predictions = {
            name: np.empty(len(y), dtype=np.int64)
            for name in (
                "baseline", "quota_025_all", "quota_025_glia",
                "interval_196_all", "interval_196_glia",
            )
        }
        print(
            f"partition requested={requested_seed} used={used_seed} ratio=2500/2500",
            flush=True,
        )
        for fold, (train, valid) in enumerate(folds, 1):
            probabilities = M.fit_extra_trees(
                x[train], pd.Series(y[train]), classes, x[valid], seeds=ET_SEEDS
            )
            probabilities = M.correct_prior(
                probabilities, M.prior_vector(pd.Series(y[train]), classes), ALPHA
            )
            decoded = decode_variants(
                probabilities, meta.iloc[train], y[train], meta.iloc[valid], classes
            )
            for name, values in decoded.items():
                predictions[name][valid] = values
            print(f"  fold {fold}/2 complete ({time.time()-t0:.1f}s)", flush=True)

        base_correct = class_array[predictions["baseline"]] == y
        for name, prediction in predictions.items():
            correct = class_array[prediction] == y
            if name == "baseline":
                p_value, wins, losses = 1.0, 0, 0
            else:
                p_value, _ = M.paired_mcnemar(correct, base_correct)
                wins = int((correct & ~base_correct).sum())
                losses = int((base_correct & ~correct).sum())
            row = {
                "requested_seed": requested_seed,
                "used_seed": used_seed,
                "variant": name,
                "accuracy": correct.mean(),
                "gain_pt": 100 * (correct.mean() - base_correct.mean()),
                "glia": correct[glia].mean(),
                "neurons": correct[~glia].mean(),
                "wins": wins,
                "losses": losses,
                "p": p_value,
            }
            rows.append(row)
            print(
                f"  {name:20s} acc={row['accuracy']:.4f} "
                f"gain={row['gain_pt']:+.2f}pt {wins}w/{losses}l p={p_value:.5g}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cohort50_screen.csv", index=False)
    candidates = frame[frame.variant != "baseline"].groupby("variant").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_gain_pt=("gain_pt", "mean"),
        min_gain_pt=("gain_pt", "min"),
        median_p=("p", "median"),
    ).reset_index()
    candidates = candidates.sort_values("median_p").reset_index(drop=True)
    candidates["holm_threshold"] = [0.05 / (4 - i) for i in range(4)]
    candidates["passes"] = (
        (candidates.min_gain_pt > 0)
        & (candidates.mean_gain_pt > 0.30)
        & (candidates.median_p < candidates.holm_threshold)
    )
    candidates.to_csv(OUT / "cohort50_summary.csv", index=False)
    print("\n" + candidates.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(
        "VERDICT: "
        + ("ADVANCE" if candidates.passes.any() else "REJECT"),
        flush=True,
    )


if __name__ == "__main__":
    main()
