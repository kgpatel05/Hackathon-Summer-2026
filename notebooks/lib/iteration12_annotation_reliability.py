"""Iteration 12 - externally learned annotation-reliability weighting.

The independent SNI reference contains the same 60-class ``voting`` consensus plus four
constituent annotation calls.  Their per-cell agreement supplies a direct noise target:
cells on which the callers disagree are less trustworthy training analogues.  An
ExtraTrees regressor learns agreement from the 200 released genes in the *different-mice*
SNI reference, then predicts a reliability weight for each challenge training cell.

Weights are normalised to mean one inside every class, so they cannot silently change the
class prior.  A within-class permutation is the matched null: identical class-wise weight
distributions, but no cell-specific reliability.  Screen is partition 773 with five model
seeds.  Advance only if the real weights gain >0.30 point, p<0.05, and beat the null by
>0.20 point.  No test-cell annotation or withheld gene is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_quota as Q
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "annotation_reliability.npz"
PARTITION = 773
SEEDS = tuple(range(5))
ALPHA = 0.45
CALLERS = ("seurat", "rctd", "tangram", "singler")


def decode(handle: h5py.File, key: str) -> np.ndarray:
    categories = [x.decode() for x in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.asarray([F._normalise_label(categories[c]) if c >= 0 else "NA"
                       for c in codes])


def build_weights(counts_train: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    if CACHE.exists():
        cached = np.load(CACHE)
        return cached["weights"]
    genes = list(counts_train.columns)
    with h5py.File(F.EXTERNAL, "r") as handle:
        ref_genes = [g.decode() for g in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(ref_genes)}
        columns = np.asarray([lookup[g] for g in genes])
        order = np.argsort(columns)
        expression = handle["X"][:, columns[order]].astype(np.float32)
        expression = expression[:, np.argsort(order)]
        voting = decode(handle, "voting")
        calls = np.column_stack([decode(handle, caller) for caller in CALLERS])

    agreement = (calls == voting[:, None]).mean(axis=1).astype(np.float32)
    usable = (expression.sum(1) > 0) & (voting != "NA")
    x_ref = F.log_cpm(expression[usable])
    target = agreement[usable]
    x_challenge = F.log_cpm(counts_train.to_numpy(np.float32))
    predictions = np.zeros(len(y), np.float32)
    for seed in range(3):
        model = ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=5,
            n_jobs=-1, random_state=seed,
        ).fit(x_ref, target)
        predictions += model.predict(x_challenge).astype(np.float32)
    predictions /= 3
    weights = 0.50 + predictions
    # Preserve the original prior and total mass separately within each target class.
    for label in np.unique(y):
        rows = y == label
        weights[rows] /= weights[rows].mean()
    np.savez_compressed(CACHE, weights=weights, predicted_agreement=predictions,
                        external_agreement=agreement[usable])
    print(f"external agreement mean={target.mean():.3f}; predicted challenge range="
          f"{predictions.min():.3f}-{predictions.max():.3f}", flush=True)
    return weights


def fit(x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray,
        x_eval: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(x_eval), len(classes)), np.float32)
    for seed in SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        ).fit(x_train, y_train, sample_weight=weights)
        out += M.align_proba(model, x_eval, classes)
    return out / len(SEEDS)


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y)); class_array = np.asarray(classes)
    weights = build_weights(counts_train, y)
    null = weights.copy(); rng = np.random.default_rng(20260819)
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        null[rows] = weights[rng.permutation(rows)]

    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    variants = {"incumbent": np.ones(len(y)), "reliability": weights,
                "within_class_permuted_null": null}
    predictions = {name: np.empty(len(y), object) for name in variants}
    folds = StratifiedKFold(5, shuffle=True, random_state=PARTITION)
    t0 = time.time()
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        allow = Q.compatibility_mask(meta_train.iloc[train], y[train],
                                     meta_train.iloc[valid], classes)
        for name, sample_weights in variants.items():
            p = fit(x[train], y[train], sample_weights[train], x[valid], classes)
            p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
            predictions[name][valid] = class_array[np.where(allow, p, -1.0).argmax(1)]
        print(f"fold {fold}/5", flush=True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)

    base_ok = predictions["incumbent"] == y
    rows = []
    for name, prediction in predictions.items():
        ok = prediction == y
        if name == "incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum()); losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100 * (ok.mean() - base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p_value})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "annotation_reliability_screen.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real, null_row = rows[1], rows[2]
    passed = (real["gain_pt"] > 0.30 and real["p"] < 0.05 and
              real["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
