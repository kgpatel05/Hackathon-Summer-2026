"""Iteration 14 - confidence-ranked transductive self-training.

Starting from the adopted 694-feature ExtraTrees stack, add the 250 (5% of the
test cohort) most confident, not-yet-selected test cells with their pseudo-labels
after each fit.  The number of rounds is selected without test truth: a fixed
80/20 split of the released training cells is followed until validation accuracy
first declines.  Recovered test labels are never imported by this script.

The production candidate is written separately.  ``prediction/prediction.csv``
is never changed unless ``--promote`` is passed explicitly; this prevents a fast
experimental run from silently replacing the established submission.

Usage:
    python3 notebooks/lib/iteration14_self_training.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration14")
OUT.mkdir(parents=True, exist_ok=True)
BASE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
COMP_CACHE = Path("outputs/iteration9/atlas_composition_cache.npz")
NICHE_CACHE = Path("outputs/iteration8/atlas_niche.npz")
ATLAS_ET_CACHE = Path("outputs/iteration9/atlas_et_block.npz")
PRODUCTION = Path("prediction/prediction.csv")

ALPHA = 0.45
ADD_PER_ROUND = 250
MAX_ROUNDS = 10
VALIDATION_SEED = 20260820
SCREEN_SEEDS = (0, 1, 2)
PRODUCTION_SEEDS = tuple(range(20))
MASK_COLS = ("Region", "Excitatory_vs_Inhibitory", "Segment")


def load_stack() -> tuple[np.ndarray, np.ndarray]:
    """Load the exact adopted 694 columns for train and test from source caches."""
    base = np.load(BASE_CACHE, allow_pickle=True)
    comp = np.load(COMP_CACHE, allow_pickle=True)["k10"]
    niche = np.load(NICHE_CACHE, allow_pickle=True)["k50"]
    atlas_et = np.load(ATLAS_ET_CACHE, allow_pickle=True)
    train = np.hstack([
        base["BASE_TR"], base["EXT_TR"], base["SPA_TR"], base["NIC_TR"],
        comp[:5000], niche[:5000], base["ATL_TR"],
        atlas_et["ATL_ET_TR"], atlas_et["COARSE_TR"],
    ]).astype(np.float32)
    test = np.hstack([
        base["BASE_TE"], base["EXT_TE"], base["SPA_TE"], base["NIC_TE"],
        comp[5000:], niche[5000:], base["ATL_TE"],
        atlas_et["ATL_ET_TE"], atlas_et["COARSE_TE"],
    ]).astype(np.float32)
    if train.shape != (5000, 694) or test.shape != (5000, 694):
        raise ValueError(f"unexpected stack shapes: {train.shape}, {test.shape}")
    return train, test


def compatibility_mask(meta_fit: pd.DataFrame, y_fit: np.ndarray,
                       meta_eval: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Training-label-only hard metadata compatibility constraint."""
    allow = np.ones((len(meta_eval), len(classes)), dtype=bool)
    for col in MASK_COLS:
        fit_values = meta_fit[col].astype(str).to_numpy()
        known = set(fit_values)
        seen = [set(fit_values[y_fit == cls]) for cls in classes]
        for i, value in enumerate(meta_eval[col].astype(str).to_numpy()):
            if value in known:
                allow[i] &= np.asarray([value in values for values in seen])
    allow[~allow.any(axis=1)] = True
    return allow


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray,
                classes: np.ndarray, seeds: tuple[int, ...],
                genuine_y: np.ndarray) -> np.ndarray:
    """Fit the production ExtraTrees family and retain the genuine-label prior."""
    out = np.zeros((len(x_eval), len(classes)), dtype=np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **M.ET_KWARGS)
        model.fit(x_fit, y_fit)
        out += M.align_proba(model, x_eval, classes.tolist())
    out /= len(seeds)
    return M.correct_prior(out, M.prior_vector(genuine_y, classes.tolist()), ALPHA)


def masked_predictions(probs: np.ndarray, allow: np.ndarray,
                       classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return constrained labels and comparable post-mask confidence."""
    constrained = np.where(allow, probs, 0.0)
    constrained /= np.maximum(constrained.sum(axis=1, keepdims=True), 1e-12)
    best = constrained.argmax(axis=1)
    return classes[best], constrained[np.arange(len(best)), best]


def run_validation(x_train: np.ndarray, y: np.ndarray, x_test: np.ndarray,
                   meta_train: pd.DataFrame, meta_test: pd.DataFrame,
                   classes: np.ndarray) -> tuple[int, pd.DataFrame]:
    """Choose the round count using released training labels only."""
    fit_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=VALIDATION_SEED,
        stratify=y,
    )
    x_fit, y_fit = x_train[fit_idx], y[fit_idx]
    selected = np.zeros(len(x_test), dtype=bool)
    pseudo_rows: list[int] = []
    pseudo_y: list[str] = []
    rows: list[dict] = []
    previous = None

    for round_id in range(MAX_ROUNDS + 1):
        if pseudo_rows:
            x_aug = np.vstack([x_fit, x_test[pseudo_rows]])
            y_aug = np.concatenate([y_fit, np.asarray(pseudo_y)])
        else:
            x_aug, y_aug = x_fit, y_fit
        probs = fit_predict(
            x_aug, y_aug, np.vstack([x_train[val_idx], x_test]),
            classes, SCREEN_SEEDS, genuine_y=y_fit,
        )
        val_probs, test_probs = probs[:len(val_idx)], probs[len(val_idx):]
        val_allow = compatibility_mask(
            meta_train.iloc[fit_idx], y_fit, meta_train.iloc[val_idx], classes
        )
        test_allow = compatibility_mask(
            meta_train.iloc[fit_idx], y_fit, meta_test, classes
        )
        val_pred, _ = masked_predictions(val_probs, val_allow, classes)
        test_pred, confidence = masked_predictions(test_probs, test_allow, classes)
        accuracy = float(np.mean(val_pred == y[val_idx]))
        declined = previous is not None and accuracy < previous
        rows.append({
            "round": round_id,
            "pseudo_cells": len(pseudo_rows),
            "validation_accuracy": accuracy,
            "delta_vs_previous": np.nan if previous is None else accuracy - previous,
            "declined": declined,
        })
        print(f"validation round={round_id} pseudo={len(pseudo_rows):4d} "
              f"accuracy={accuracy:.4f}", flush=True)
        if declined or round_id == MAX_ROUNDS:
            break
        previous = accuracy
        candidates = np.flatnonzero(~selected)
        chosen = candidates[np.argsort(confidence[candidates])[-ADD_PER_ROUND:]]
        selected[chosen] = True
        pseudo_rows.extend(chosen.tolist())
        pseudo_y.extend(test_pred[chosen].tolist())

    audit = pd.DataFrame(rows)
    eligible = audit.loc[~audit["declined"]]
    best_round = int(eligible.loc[eligible.validation_accuracy.idxmax(), "round"])
    return best_round, audit


def run_production(rounds: int, x_train: np.ndarray, y: np.ndarray,
                   x_test: np.ndarray, meta_train: pd.DataFrame,
                   meta_test: pd.DataFrame, classes: np.ndarray) -> Path:
    """Run the validation-frozen number of rounds on all released training cells."""
    selected = np.zeros(len(x_test), dtype=bool)
    pseudo_rows: list[int] = []
    pseudo_y: list[str] = []
    test_allow = compatibility_mask(meta_train, y, meta_test, classes)
    target_col = pd.read_csv(PRODUCTION, nrows=0).columns[1]
    final_path = OUT / "prediction_self_training_honest.csv"

    for round_id in range(rounds + 1):
        if pseudo_rows:
            x_aug = np.vstack([x_train, x_test[pseudo_rows]])
            y_aug = np.concatenate([y, np.asarray(pseudo_y)])
        else:
            x_aug, y_aug = x_train, y
        t0 = time.time()
        probs = fit_predict(
            x_aug, y_aug, x_test, classes, PRODUCTION_SEEDS, genuine_y=y
        )
        pred, confidence = masked_predictions(probs, test_allow, classes)
        path = OUT / f"prediction_self_training_round{round_id}.csv"
        pd.DataFrame({
            "Cell_ID": meta_test.index.astype(str), target_col: pred,
        }).to_csv(path, index=False)
        print(f"production round={round_id} pseudo={len(pseudo_rows):4d} "
              f"fit_seconds={time.time()-t0:.1f}", flush=True)
        if round_id == rounds:
            shutil.copy(path, final_path)
            break
        candidates = np.flatnonzero(~selected)
        chosen = candidates[np.argsort(confidence[candidates])[-ADD_PER_ROUND:]]
        selected[chosen] = True
        pseudo_rows.extend(chosen.tolist())
        pseudo_y.extend(pred[chosen].tolist())
    return final_path


def main() -> None:
    x_train, x_test = load_stack()
    _, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))

    best_round, audit = run_validation(
        x_train, y, x_test, meta_train, meta_test, classes
    )
    audit.to_csv(OUT / "self_training_validation.csv", index=False)
    base = float(audit.loc[audit["round"] == 0, "validation_accuracy"].iloc[0])
    chosen = float(audit.loc[audit["round"] == best_round, "validation_accuracy"].iloc[0])
    print(f"selected round={best_round}: {base:.4f} -> {chosen:.4f}", flush=True)

    candidate = run_production(
        best_round, x_train, y, x_test, meta_train, meta_test, classes
    )
    if "--promote" in sys.argv and best_round > 0 and chosen > base:
        backup = OUT / "prediction_before_self_training.csv"
        shutil.copy(PRODUCTION, backup)
        shutil.copy(candidate, PRODUCTION)
        print(f"promoted {candidate} -> {PRODUCTION}; backup={backup}")
    else:
        print(f"retained {PRODUCTION}; pass --promote to opt into a validation-approved candidate")
    print(f"candidate={candidate}")


if __name__ == "__main__":
    main()
