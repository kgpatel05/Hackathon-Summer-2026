"""Iteration 13 - fresh confirmation of modern-stack linear diversity.

The original Iteration-2 model blended logistic regression into a much smaller feature
stack.  The adopted model has since grown to 694 columns, including four external-atlas
posterior/context blocks, without re-testing a strongly regularized linear member.

An exploratory screen on partition 997 compared C={0.01, 0.1, 1.0}.  The single frozen
survivor is multinomial logistic C=0.01 at a 10% probability weight:

    ExtraTrees 0.8048 -> 0.8088, +0.40 point, 44 wins / 24 losses, p=0.0205.

This file performs one fresh test only: five-fold partition seed 1879, 20 ExtraTrees
seeds, fold-scoped scaling, prior correction alpha 0.45, metadata compatibility mask,
and the fixed 90/10 blend.  Confirm only for gain >0.30 point and paired exact McNemar
p<0.05.  No test label is read and no submission is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_quota as Q
from iteration10_spatial_fields import current_stack


OUT = Path("outputs/iteration13")
OUT.mkdir(parents=True, exist_ok=True)
FOLD_SEED = 1879
LOGISTIC_C = 0.01
BLEND_WEIGHT = 0.10
ALPHA = 0.45
ET_SEEDS = tuple(range(20))


def masked(
    probabilities: np.ndarray,
    fit_meta: pd.DataFrame,
    fit_y: np.ndarray,
    eval_meta: pd.DataFrame,
    classes: list[str],
) -> np.ndarray:
    allow = Q.compatibility_mask(fit_meta, fit_y, eval_meta, classes)
    output = np.where(allow, probabilities, 0.0)
    output /= np.maximum(output.sum(1, keepdims=True), 1e-12)
    return output.astype(np.float32)


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts.columns))
    et_oof = np.zeros((len(y), len(classes)), np.float32)
    linear_oof = np.zeros_like(et_oof)
    folds = StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED)
    t0 = time.time()
    print(
        f"partition={FOLD_SEED} C={LOGISTIC_C} blend={BLEND_WEIGHT} "
        f"ET_seeds={len(ET_SEEDS)} features={x.shape[1]}",
        flush=True,
    )

    for fold, (train, valid) in enumerate(folds.split(x, y), 1):
        prior = M.prior_vector(pd.Series(y[train]), classes)
        et = M.fit_extra_trees(
            x[train], pd.Series(y[train]), classes, x[valid], seeds=ET_SEEDS
        )
        et = M.correct_prior(et, prior, ALPHA)
        et_oof[valid] = masked(
            et, meta.iloc[train], y[train], meta.iloc[valid], classes
        )

        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=LOGISTIC_C, max_iter=3000).fit(
            scaler.transform(x[train]), y[train]
        )
        linear = M.align_proba(model, scaler.transform(x[valid]), classes)
        linear = M.correct_prior(linear, prior, ALPHA)
        linear_oof[valid] = masked(
            linear, meta.iloc[train], y[train], meta.iloc[valid], classes
        )
        print(f"fold {fold}/5 complete ({time.time()-t0:.1f}s)", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et_oof + BLEND_WEIGHT * linear_oof
    glia = meta["Region"].isna().to_numpy()
    base_correct = class_array[et_oof.argmax(axis=1)] == y
    rows = []
    for name, probabilities in {
        "ExtraTrees incumbent": et_oof,
        "logistic C=0.01": linear_oof,
        "0.90 ET + 0.10 logistic": blend,
    }.items():
        correct = class_array[probabilities.argmax(axis=1)] == y
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_correct)
            wins = int((correct & ~base_correct).sum())
            losses = int((base_correct & ~correct).sum())
        rows.append({
            "config": name,
            "accuracy": correct.mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "glia": correct[glia].mean(),
            "neurons": correct[~glia].mean(),
            "wins": wins,
            "losses": losses,
            "p": p_value,
            "fold_seed": FOLD_SEED,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "linear_confirm_oof.csv", index=False)
    np.savez_compressed(
        OUT / "linear_confirm_oof.npz",
        et=et_oof,
        logistic=linear_oof,
        blend=blend,
        truth=y,
        classes=class_array,
        fold_seed=FOLD_SEED,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    candidate = rows[-1]
    confirmed = candidate["gain_pt"] > 0.30 and candidate["p"] < 0.05
    print("VERDICT: " + ("CONFIRMED" if confirmed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
