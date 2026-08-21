"""Iteration 12 - hurdle-multinomial empirical Bayes for sparse MERFISH counts.

The median challenge cell contains only ~21 transcripts.  A tree split does not encode
the sampling process behind those counts.  This experiment factors the likelihood into

    P(detected genes | class) * P(transcript allocation | class, total count),

using a beta-Bernoulli detection model and a Dirichlet-multinomial allocation model.
All class parameters are estimated inside each CV fold.  The candidate is evaluated both
alone and as one frozen 10% probability expert beside the adopted 694-feature ExtraTrees.
The ExtraTrees OOF probabilities are reused from the iteration-10 CatBoost screen, whose
partition (seed 307) and predictions were saved before this hypothesis was proposed.

Screen rule fixed before running: advance only if the 90/10 blend gains >0.30 percentage
points and exact paired McNemar p<0.05.  No test label, external label, or withheld gene is
read.  The script never writes a submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/iteration10/catboost_screen_oof.npz")
PARTITION = 307
N_SPLITS = 5
DIRICHLET = 0.5
BETA_POS = 1.0
BETA_NEG = 4.0
DETECTION_WEIGHT = 0.35
BLEND_WEIGHT = 0.10


def hurdle_log_posterior(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Return normalised class posteriors under a hurdle-multinomial model."""
    k, g = len(classes), x_train.shape[1]
    theta = np.empty((k, g), np.float64)
    detect = np.empty((k, g), np.float64)
    prior = np.empty(k, np.float64)

    for j, label in enumerate(classes):
        rows = x_train[y_train == label]
        gene_counts = rows.sum(axis=0, dtype=np.float64) + DIRICHLET
        theta[j] = gene_counts / gene_counts.sum()
        positive = (rows > 0).sum(axis=0, dtype=np.float64) + BETA_POS
        detect[j] = positive / (len(rows) + BETA_POS + BETA_NEG)
        prior[j] = (len(rows) + 0.5) / (len(x_train) + 0.5 * k)

    binary = (x_eval > 0).astype(np.float64)
    # Multinomial combinatorial constants are identical across classes and cancel.
    allocation = x_eval @ np.log(np.clip(theta, 1e-12, 1.0)).T
    detection = (
        binary @ np.log(np.clip(detect, 1e-8, 1.0)).T
        + (1.0 - binary) @ np.log(np.clip(1.0 - detect, 1e-8, 1.0)).T
    )
    logits = allocation + DETECTION_WEIGHT * detection + np.log(prior)[None, :]
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits).astype(np.float32)


def main() -> None:
    counts_train, meta_train, _, _ = F.load_challenge()
    x = counts_train.to_numpy(np.float64)
    y = meta_train[F.TARGET].astype(str).to_numpy()

    cached = np.load(CACHE, allow_pickle=True)
    classes = cached["classes"].astype(str)
    if not np.array_equal(cached["y"].astype(str), y):
        raise ValueError("cached ExtraTrees targets do not match current training order")
    et = cached["et"].astype(np.float32)

    bayes = np.zeros_like(et)
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=PARTITION)
    for fold, (train, valid) in enumerate(folds.split(x, y), 1):
        bayes[valid] = hurdle_log_posterior(x[train], y[train], x[valid], classes)
        print(f"fold {fold}/{N_SPLITS} complete", flush=True)

    # Candidate probabilities can be extremely sharp at 21 counts.  A geometric pool
    # combines evidence in log space while the fixed 10% exponent limits its influence.
    log_blend = (
        (1.0 - BLEND_WEIGHT) * np.log(np.clip(et, 1e-8, 1.0))
        + BLEND_WEIGHT * np.log(np.clip(bayes, 1e-8, 1.0))
    )
    log_blend -= logsumexp(log_blend, axis=1, keepdims=True)
    blend = np.exp(log_blend)

    labels = np.asarray(classes)
    base_ok = labels[et.argmax(1)] == y
    rows = []
    for name, probabilities in {
        "ExtraTrees incumbent": et,
        "hurdle-multinomial": bayes,
        "0.90 ET x 0.10 Bayes": blend,
    }.items():
        ok = labels[probabilities.argmax(1)] == y
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        gain = ok.mean() - base_ok.mean()
        rows.append({"config": name, "accuracy": ok.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:25s} acc={ok.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "count_bayes_screen.csv", index=False)
    np.savez_compressed(OUT / "count_bayes_screen_oof.npz", et=et, bayes=bayes,
                        y=y, classes=classes)
    candidate = rows[-1]
    passed = candidate["gain_pt"] > 0.30 and candidate["p"] < 0.05
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
