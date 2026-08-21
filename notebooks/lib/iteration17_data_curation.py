"""Iteration 17b - fold-scoped removal of label-inconsistent exemplars.

The data audit found no train/test QC shift and no highly redundant gene pairs, so low
depth or correlation-based deletion is not justified.  This experiment instead asks a
narrower question: do a few training cells have labels inconsistent with their sparse
molecular neighbourhood?

For each outer-fold training cell, three self-excluded class-proximity ranks are built
from the dropout-aware Tversky, TF-IDF detection, and square-root-count geometries in
``iteration17_data_geometry``.  Their mean is a label-consistency score.  The bottom
2.5% is removed *within each class* (classes with fewer than 20 fold-training examples
are untouched), preserving class balance.  An identically sized within-class random
removal is the matched null.  ExtraTrees features and hyperparameters remain unchanged.

The 2.5% rule was frozen after a one-seed exploration on historical partition 307;
5%/10% removal and soft weighting were worse.  The formal screen is fresh partition 487
with five seeds.  Advance only for >0.30 point, McNemar p<0.05, and >0.20 point over the
random-removal null.  Confirmation is untouched partition 509 with twenty seeds and
requires >0.20 point and p<0.05.  No recovered test label is read and no prediction file
is written.
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
import iteration17_data_geometry as G


OUT = Path("outputs/iteration17")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN_RESULT = OUT / "data_curation_screen.csv"
ALPHA = 0.45
TRIM_FRACTION = 0.025
MIN_CLASS_SIZE = 20
SCREEN_PARTITION = 487
CONFIRM_PARTITION = 509
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def consistency_score(counts: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                      classes: np.ndarray, seed: int) -> np.ndarray:
    real_fit, _, _, _ = G.proximity_features(
        counts, y, meta, counts[:1], classes, seed=seed
    )
    class_index = {label: index for index, label in enumerate(classes)}
    codes = np.asarray([class_index[label] for label in y])
    ranks = []
    for geometry in range(3):
        block = real_fit[:, geometry * len(classes):(geometry + 1) * len(classes)]
        own = block[np.arange(len(y)), codes]
        ranks.append((block <= own[:, None]).mean(1))
    return np.mean(ranks, axis=0)


def curation_weights(score: np.ndarray, y: np.ndarray, classes: np.ndarray,
                     random: bool, seed: int) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for label in classes:
        rows = np.flatnonzero(y == label)
        if len(rows) < MIN_CLASS_SIZE:
            continue
        number = max(1, int(np.floor(TRIM_FRACTION * len(rows))))
        if random:
            removed = rng.choice(rows, number, replace=False)
        else:
            removed = rows[np.argsort(score[rows])[:number]]
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
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    x_base, _ = I15.load_incumbent()
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    names = ["incumbent_694", "trim label-inconsistent 2.5%", "random class-matched trim (null)"]
    probabilities = {name: np.zeros((len(y), len(classes)), np.float32) for name in names}
    removal_counts = {name: [] for name in names}
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    started = time.time()
    print(f"mode={mode} partition={partition} seeds={len(seeds)}", flush=True)

    for fold, (fit, valid) in enumerate(folds.split(x_base, y), 1):
        fold_started = time.time()
        score = consistency_score(
            counts[fit], y[fit], meta_train.iloc[fit], classes,
            seed=partition * 10 + fold,
        )
        weights = {
            "incumbent_694": np.ones(len(fit), np.float32),
            "trim label-inconsistent 2.5%": curation_weights(
                score, y[fit], classes, random=False, seed=partition + fold
            ),
            "random class-matched trim (null)": curation_weights(
                score, y[fit], classes, random=True, seed=partition + fold
            ),
        }
        for name in names:
            probabilities[name][valid] = fit_predict(
                x_base[fit], y[fit], meta_train.iloc[fit], x_base[valid],
                meta_train.iloc[valid], classes, seeds, weights[name],
            )
            removal_counts[name].append(int((weights[name] == 0).sum()))
        print(f"  fold {fold}/5 complete in {time.time()-fold_started:.1f}s", flush=True)

    class_array = np.asarray(classes)
    glia = meta_train["Region"].isna().to_numpy()
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
            "removed_per_fold": "/".join(map(str, removal_counts[name])),
        })
    result = pd.DataFrame(rows)
    by_name = result.set_index("config")
    real = by_name.loc["trim label-inconsistent 2.5%"]
    null = by_name.loc["random class-matched trim (null)"]
    if mode == "screen":
        passed = bool(real.gain_pt > 0.30 and real.mcnemar_p < 0.05 and
                      real.gain_pt - null.gain_pt > 0.20)
        result["advance"] = result.config.eq("trim label-inconsistent 2.5%") & passed
        path = SCREEN_RESULT
    else:
        passed = bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
        result["adopt"] = result.config.eq("trim label-inconsistent 2.5%") & passed
        path = OUT / "data_curation_confirm.csv"
    result.to_csv(path, index=False)
    np.savez_compressed(
        OUT / f"data_curation_{mode}_oof.npz",
        incumbent=probabilities["incumbent_694"],
        curated=probabilities["trim label-inconsistent 2.5%"],
        random_null=probabilities["random class-matched trim (null)"],
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
