"""Iteration 17c - prune only high-information, label-contradictory cells.

Iteration 17b showed that class-balanced molecular inconsistency is a real data-quality
signal (+0.26 point on its fresh screen), but the effect missed the advancement gate.
Inspection supplied an important distinction: a sparse cell can look inconsistent only
because it contains too few observations, whereas a well-observed cell that strongly
resembles other classes is credible annotation-noise evidence.

This refinement removes a fold-training cell only when all conditions hold:

1. its self-excluded mean class-proximity rank is below 0.75;
2. its detected-gene count is at least the median of its own target class;
3. its class has at least 20 fold-training examples;
4. at most the lowest-consistency 10% of a class can be removed.

The matched null removes the same number of high-information rows from each class at
random.  The rule was frozen after one-seed exploration on historical partition 307.
Formal screen: untouched partition 541, five ExtraTrees seeds; require >0.30 point,
McNemar p<0.05, and >0.20 point over null.  Confirmation: untouched partition 557,
twenty seeds; require >0.20 point and p<0.05.  No test truth or prediction is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as I15
import iteration17_data_curation as C17


OUT = Path("outputs/iteration17")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN_RESULT = OUT / "high_information_pruning_screen.csv"
ALPHA = 0.45
CONSISTENCY_THRESHOLD = 0.75
MIN_CLASS_SIZE = 20
MAX_CLASS_FRACTION = 0.10
SCREEN_PARTITION = 541
CONFIRM_PARTITION = 557
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def pruning_weights(score: np.ndarray, detected: np.ndarray, y: np.ndarray,
                    classes: np.ndarray, random: bool, seed: int) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for label in classes:
        rows = np.flatnonzero(y == label)
        if len(rows) < MIN_CLASS_SIZE:
            continue
        median_detected = np.median(detected[rows])
        high_information = rows[detected[rows] >= median_detected]
        contradictory = high_information[score[high_information] < CONSISTENCY_THRESHOLD]
        maximum = max(1, int(np.floor(MAX_CLASS_FRACTION * len(rows))))
        contradictory = contradictory[np.argsort(score[contradictory])[:maximum]]
        number = len(contradictory)
        if number == 0:
            continue
        removed = (rng.choice(high_information, number, replace=False)
                   if random else contradictory)
        weights[removed] = 0.0
    return weights


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
                x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
                seeds: tuple[int, ...], weights: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(x_eval), len(classes)), dtype=np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **M.ET_KWARGS)
        model.fit(x_fit, y_fit, sample_weight=weights)
        probabilities += M.align_proba(model, x_eval, classes.tolist())
    probabilities /= len(seeds)
    probabilities = M.correct_prior(
        probabilities, M.prior_vector(pd.Series(y_fit), classes.tolist()), ALPHA
    )
    allow = I15.compatibility_mask(meta_fit, y_fit, meta_eval, classes.tolist())
    probabilities = np.where(allow, probabilities, 0.0)
    probabilities /= np.maximum(probabilities.sum(1, keepdims=True), 1e-12)
    return probabilities.astype(np.float32)


def evaluate(mode: str) -> pd.DataFrame:
    counts_train, meta_train, _, _ = F.load_challenge()
    counts = counts_train.to_numpy(np.float32)
    detected = (counts > 0).sum(1)
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    x_base, _ = I15.load_incumbent()
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    names = ["incumbent_694", "prune high-information contradictions",
             "QC/class-matched random prune (null)"]
    probabilities = {name: np.zeros((len(y), len(classes)), np.float32) for name in names}
    removals = {name: [] for name in names}
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    started = time.time()
    print(f"mode={mode} partition={partition} seeds={len(seeds)}", flush=True)

    for fold, (fit, valid) in enumerate(folds.split(x_base, y), 1):
        fold_started = time.time()
        score = C17.consistency_score(
            counts[fit], y[fit], meta_train.iloc[fit], classes,
            seed=partition * 10 + fold,
        )
        weights = {
            "incumbent_694": np.ones(len(fit), np.float32),
            "prune high-information contradictions": pruning_weights(
                score, detected[fit], y[fit], classes, False, partition + fold
            ),
            "QC/class-matched random prune (null)": pruning_weights(
                score, detected[fit], y[fit], classes, True, partition + fold
            ),
        }
        for name in names:
            probabilities[name][valid] = fit_predict(
                x_base[fit], y[fit], meta_train.iloc[fit], x_base[valid],
                meta_train.iloc[valid], classes, seeds, weights[name],
            )
            removals[name].append(int((weights[name] == 0).sum()))
        print(f"  fold {fold}/5 complete in {time.time()-fold_started:.1f}s", flush=True)

    glia = meta_train["Region"].isna().to_numpy()
    class_array = np.asarray(classes)
    base_correct = class_array[probabilities["incumbent_694"].argmax(1)] == y
    rows = []
    for name in names:
        prediction = class_array[probabilities[name].argmax(1)]
        correct = prediction == y
        if name == "incumbent_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_correct)
            wins = int((correct & ~base_correct).sum())
            losses = int((base_correct & ~correct).sum())
        rows.append({
            "mode": mode,
            "partition": partition,
            "config": name,
            "accuracy": correct.mean(),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "cohen_kappa": cohen_kappa_score(y, prediction),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "wins": wins,
            "losses": losses,
            "mcnemar_p": p_value,
            "removed_per_fold": "/".join(map(str, removals[name])),
        })
    result = pd.DataFrame(rows)
    by_name = result.set_index("config")
    real = by_name.loc["prune high-information contradictions"]
    null = by_name.loc["QC/class-matched random prune (null)"]
    if mode == "screen":
        passed = bool(real.gain_pt > 0.30 and real.mcnemar_p < 0.05 and
                      real.gain_pt - null.gain_pt > 0.20)
        result["advance"] = result.config.eq("prune high-information contradictions") & passed
        path = SCREEN_RESULT
    else:
        passed = bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
        result["adopt"] = result.config.eq("prune high-information contradictions") & passed
        path = OUT / "high_information_pruning_confirm.csv"
    result.to_csv(path, index=False)
    np.savez_compressed(
        OUT / f"high_information_pruning_{mode}_oof.npz",
        incumbent=probabilities["incumbent_694"],
        pruned=probabilities["prune high-information contradictions"],
        random_null=probabilities["QC/class-matched random prune (null)"],
        truth=y, classes=classes, partition=partition,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    print(f"VERDICT: {'PASS' if passed else 'REJECT'}; total {time.time()-started:.1f}s", flush=True)
    return result


def screen_passed() -> bool:
    if not SCREEN_RESULT.exists():
        return False
    result = pd.read_csv(SCREEN_RESULT)
    return bool(result.get("advance", pd.Series(dtype=bool)).fillna(False).astype(bool).any())


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm", "run"}:
        raise SystemExit("mode must be screen, confirm, or run")
    if mode in {"screen", "run"}:
        evaluate("screen")
    if mode == "confirm" and not screen_passed():
        raise SystemExit("confirmation is locked: the frozen screen did not pass")
    if mode == "confirm" or (mode == "run" and screen_passed()):
        evaluate("confirm")
    elif mode == "run":
        print("confirmation not run because the pre-registered screen gate failed", flush=True)


if __name__ == "__main__":
    main()
