"""Shared, leakage-resistant infrastructure for the Iteration-16 novel suite.

The training runner in :mod:`iteration16_novel_suite` deliberately cannot import the
recovered test truth.  Test scoring lives in the separate ``iteration16_score.py``
thermometer.  Every candidate uses the same frozen 80/20 screen, incumbent probability
stack, metadata mask, prior correction, and output contract.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as I15


OUT = Path("outputs/iteration16")
OUT.mkdir(parents=True, exist_ok=True)
PRODUCTION = Path("prediction/prediction.csv")
SCREEN_SEED = 20260821
ALPHA = 0.45
SCREEN_ET_SEEDS = (0, 1, 2, 3, 4)
TEST_ET_SEEDS = tuple(range(20))


def seed_everything(seed: int = SCREEN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def device_name() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_data() -> dict:
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    x_train, x_test = I15.load_incumbent()
    fit, valid = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=SCREEN_SEED, stratify=y
    )
    return {
        "counts_train": counts_train.to_numpy(np.float32),
        "counts_test": counts_test.to_numpy(np.float32),
        "genes": np.asarray(counts_train.columns.astype(str)),
        "meta_train": meta_train,
        "meta_test": meta_test,
        "x_train": x_train,
        "x_test": x_test,
        "y": y,
        "classes": classes,
        "fit": fit,
        "valid": valid,
    }


def mask_probabilities(
    probabilities: np.ndarray,
    meta_fit: pd.DataFrame,
    y_fit: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    allow = I15.compatibility_mask(
        meta_fit, np.asarray(y_fit, dtype=str), meta_eval, classes.tolist()
    )
    out = np.where(allow, probabilities, 0.0)
    out /= np.maximum(out.sum(1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(probabilities, dtype=np.float64), 1e-12)
    out /= out.sum(1, keepdims=True)
    return out.astype(np.float32)


def blend(base: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    return normalize_probabilities((1.0 - weight) * base + weight * candidate)


def fit_incumbent(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    meta_fit: pd.DataFrame,
    x_eval: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: np.ndarray,
    seeds: tuple[int, ...],
) -> np.ndarray:
    probabilities = M.fit_extra_trees(
        x_fit, pd.Series(y_fit), classes.tolist(), x_eval, seeds=seeds
    )
    probabilities = M.correct_prior(
        probabilities, M.prior_vector(pd.Series(y_fit), classes.tolist()), ALPHA
    )
    return mask_probabilities(
        probabilities, meta_fit, y_fit, meta_eval, classes
    )


def reduce_features(
    x_fit: np.ndarray, x_eval: np.ndarray, n_components: int = 64
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler().fit(x_fit)
    fit_scaled = np.clip(scaler.transform(x_fit), -8.0, 8.0)
    eval_scaled = np.clip(scaler.transform(x_eval), -8.0, 8.0)
    n_components = min(n_components, x_fit.shape[0] - 1, x_fit.shape[1])
    pca = PCA(n_components=n_components, random_state=SCREEN_SEED).fit(fit_scaled)
    fit_reduced = pca.transform(fit_scaled).astype(np.float32)
    eval_reduced = pca.transform(eval_scaled).astype(np.float32)
    scale = np.maximum(fit_reduced.std(0, keepdims=True), 1e-4)
    return fit_reduced / scale, eval_reduced / scale, scaler, pca


def metric_row(
    name: str,
    probabilities: np.ndarray,
    truth: np.ndarray,
    classes: np.ndarray,
    glia: np.ndarray,
    base_correct: np.ndarray | None = None,
) -> dict:
    pred = classes[probabilities.argmax(1)]
    correct = pred == truth
    row = {
        "candidate": name,
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "cohen_kappa": cohen_kappa_score(truth, pred),
        "glia_accuracy": accuracy_score(truth[glia], pred[glia]),
        "neuron_accuracy": accuracy_score(truth[~glia], pred[~glia]),
        "changed_vs_incumbent": 0,
        "wins_vs_incumbent": 0,
        "losses_vs_incumbent": 0,
        "mcnemar_p": 1.0,
    }
    if base_correct is not None:
        p_value, _ = M.paired_mcnemar(correct, base_correct)
        row.update({
            "wins_vs_incumbent": int((correct & ~base_correct).sum()),
            "losses_vs_incumbent": int((base_correct & ~correct).sum()),
            "mcnemar_p": p_value,
        })
    return row


def write_candidate(
    name: str,
    probabilities: np.ndarray,
    meta_test: pd.DataFrame,
    classes: np.ndarray,
) -> Path:
    target = pd.read_csv(PRODUCTION, nrows=0).columns[1]
    pred = classes[probabilities.argmax(1)]
    path = OUT / "predictions" / f"prediction_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"Cell_ID": meta_test.index.astype(str), target: pred})
    if len(frame) != 5000 or frame.Cell_ID.duplicated().any():
        raise ValueError(f"invalid candidate output {name}")
    frame.to_csv(path, index=False)
    return path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
