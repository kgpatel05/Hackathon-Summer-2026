"""Iteration 17 - data geometry for an extremely sparse MERFISH panel.

This iteration changes the representation of the *data*, not the model family.  A
median challenge cell detects only 12 of 200 genes, so ordinary one-feature tree splits
may miss a useful high-order object: which small set of genes co-occurred.  We turn that
object into fold-scoped, per-class proximity features under three geometries:

* dropout-aware Tversky similarity, which penalises an observed query gene missing from
  a reference more strongly than a reference gene absent from a sparse query;
* TF-IDF cosine on gene detection, which gives rare detections more weight;
* square-root-count cosine, which retains molecule multiplicity without letting a few
  high-count genes dominate.

For every cell and class, a feature is the mean similarity to that class's three closest
fold-training cells.  Training-row features exclude the row itself.  Validation labels
are never read.  A matched null permutes fine labels within the broad glia/neuron
compartment before constructing an identically sized 180-column block.  This controls
both the wider ExtraTrees feature matrix and generic nearest-neighbour smoothness.

Protocol fixed before the formal screen
---------------------------------------
Exploration/audit may use the already-existing partition-307 OOF cache.  The formal
screen is partition 433 with five ExtraTrees seeds.  Advance only for >0.30 percentage
point over the exact 694-feature incumbent, exact paired McNemar p<0.05, and >0.20 point
over the matched null.  A survivor is re-run on untouched partition 461 with twenty
seeds and adopted only for >0.20 point and p<0.05.  Recovered test truth is never read;
this file never changes ``prediction/prediction.csv``.

Usage
-----
``python3 notebooks/lib/iteration17_data_geometry.py audit``
``python3 notebooks/lib/iteration17_data_geometry.py screen``
``python3 notebooks/lib/iteration17_data_geometry.py confirm``
``python3 notebooks/lib/iteration17_data_geometry.py run``
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as I15


OUT = Path("outputs/iteration17")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN_RESULT = OUT / "data_geometry_screen.csv"
EXPLORATION_CACHE = Path("outputs/iteration10/catboost_screen_oof.npz")
ALPHA = 0.45
SCREEN_PARTITION = 433
CONFIRM_PARTITION = 461
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
TOP_PER_CLASS = 3
TVERSKY_QUERY_MISS = 1.0
TVERSKY_DROPOUT_MISS = 0.15


def _keys(matrix: np.ndarray, binary: bool = False) -> list[bytes]:
    if binary:
        return [np.packbits(row > 0).tobytes() for row in matrix]
    contiguous = np.ascontiguousarray(matrix)
    return [row.tobytes() for row in contiguous]


def _duplicate_summary(matrix: np.ndarray, labels: np.ndarray | None,
                       binary: bool) -> dict:
    keys = _keys(matrix, binary=binary)
    counts = Counter(keys)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for row, key in enumerate(keys):
        groups[key].append(row)
    repeated = [key for key, number in counts.items() if number > 1]
    conflicting = 0
    if labels is not None:
        conflicting = sum(len(set(labels[groups[key]])) > 1 for key in repeated)
    return {
        "unique_profiles": len(counts),
        "repeated_groups": len(repeated),
        "rows_in_repeated_groups": int(sum(counts[key] for key in repeated)),
        "conflicting_label_groups": int(conflicting),
        "largest_group": int(max(counts.values())),
    }


def _qc_frame(counts: np.ndarray) -> pd.DataFrame:
    depth = counts.sum(1).astype(float)
    detected = (counts > 0).sum(1).astype(float)
    positive_total = np.maximum(depth[:, None], 1.0)
    proportions = counts / positive_total
    entropy = -(np.where(proportions > 0, proportions * np.log(proportions + 1e-12), 0)).sum(1)
    return pd.DataFrame({
        "depth": depth,
        "detected_genes": detected,
        "zero_fraction": 1.0 - detected / counts.shape[1],
        "top_gene_share": counts.max(1) / np.maximum(depth, 1.0),
        "count_entropy": entropy,
    })


def _decile_error_table(qc: pd.DataFrame, correct: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in qc.columns:
        bins = pd.qcut(qc[metric], 10, duplicates="drop")
        for number, interval in enumerate(bins.cat.categories):
            selected = (bins == interval).to_numpy()
            rows.append({
                "metric": metric,
                "bin": number,
                "low": float(interval.left),
                "high": float(interval.right),
                "cells": int(selected.sum()),
                "error_rate": float(1.0 - correct[selected].mean()),
            })
    return pd.DataFrame(rows)


def audit() -> None:
    """Write train-only/data-only diagnostics; recovered test labels stay sealed."""
    counts_train, meta_train, counts_test, _ = F.load_challenge()
    train = counts_train.to_numpy(np.float32)
    test = counts_test.to_numpy(np.float32)
    y = meta_train[F.TARGET].astype(str).to_numpy()
    qc_train, qc_test = _qc_frame(train), _qc_frame(test)

    summary: dict[str, object] = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "classes": int(len(np.unique(y))),
        "train_zero_fraction": float((train == 0).mean()),
        "test_zero_fraction": float((test == 0).mean()),
        "train_count_duplicates": _duplicate_summary(train, y, binary=False),
        "train_detection_support_duplicates": _duplicate_summary(train, y, binary=True),
        "test_count_duplicates": _duplicate_summary(test, None, binary=False),
    }

    train_count_keys = set(_keys(train, binary=False))
    train_support_keys = set(_keys(train, binary=True))
    summary["test_cells_with_exact_train_count_profile"] = int(
        sum(key in train_count_keys for key in _keys(test, binary=False))
    )
    summary["test_cells_with_exact_train_detection_support"] = int(
        sum(key in train_support_keys for key in _keys(test, binary=True))
    )

    distribution_rows = []
    for column in qc_train.columns:
        statistic, p_value = ks_2samp(qc_train[column], qc_test[column])
        distribution_rows.append({
            "metric": column,
            "train_mean": qc_train[column].mean(),
            "test_mean": qc_test[column].mean(),
            "train_median": qc_train[column].median(),
            "test_median": qc_test[column].median(),
            "ks_statistic": statistic,
            "ks_p": p_value,
        })
    pd.DataFrame(distribution_rows).to_csv(OUT / "train_test_qc_shift.csv", index=False)

    # Gene redundancy: sparsity does not imply that the 200 measured genes are copies.
    expression = np.log1p(train)
    correlation = np.corrcoef(expression, rowvar=False)
    upper = np.triu_indices_from(correlation, 1)
    values = correlation[upper]
    order = np.argsort(-np.abs(values))
    pair_rows = []
    for position in order[:50]:
        pair_rows.append({
            "gene_a": str(counts_train.columns[upper[0][position]]),
            "gene_b": str(counts_train.columns[upper[1][position]]),
            "pearson_log1p": float(values[position]),
        })
    pd.DataFrame(pair_rows).to_csv(OUT / "top_gene_correlations.csv", index=False)
    summary["absolute_gene_correlation_quantiles"] = {
        str(q): float(np.quantile(np.abs(values), q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)
    }
    summary["gene_pairs_abs_correlation_over_0_8"] = int((np.abs(values) > 0.8).sum())

    # Per-gene train/test detection shift, useful for deciding whether row/gene filtering
    # would repair covariate shift or merely discard matched data.
    detect_train, detect_test = train > 0, test > 0
    p_train, p_test = detect_train.mean(0), detect_test.mean(0)
    pooled = (detect_train.sum(0) + detect_test.sum(0)) / (len(train) + len(test))
    standard_error = np.sqrt(np.maximum(pooled * (1.0 - pooled) *
                                        (1.0 / len(train) + 1.0 / len(test)), 1e-12))
    shift = pd.DataFrame({
        "gene": counts_train.columns.astype(str),
        "train_detection_rate": p_train,
        "test_detection_rate": p_test,
        "difference": p_test - p_train,
        "z_score": (p_test - p_train) / standard_error,
    }).sort_values("z_score", key=np.abs, ascending=False)
    shift.to_csv(OUT / "gene_detection_shift.csv", index=False)

    # Reuse a historical OOF partition only as an exploratory error microscope.
    if EXPLORATION_CACHE.exists():
        cached = np.load(EXPLORATION_CACHE, allow_pickle=True)
        if np.array_equal(cached["y"].astype(str), y):
            classes = cached["classes"].astype(str)
            probabilities = cached["et"].astype(np.float32)
            correct = classes[probabilities.argmax(1)] == y
            table = _decile_error_table(qc_train, correct)
            table.to_csv(OUT / "oof_error_by_qc_decile.csv", index=False)
            summary["exploratory_oof_accuracy_partition_307"] = float(correct.mean())
            summary["qc_spearman_with_correctness"] = {
                column: {
                    "rho": float(spearmanr(qc_train[column], correct).statistic),
                    "p": float(spearmanr(qc_train[column], correct).pvalue),
                } for column in qc_train.columns
            }

    (OUT / "data_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote data audit to {OUT}", flush=True)


def _normalised_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.maximum(
        np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :],
        1e-8,
    )
    return (left @ right.T) / denominator


def similarity_matrix(query: np.ndarray, reference: np.ndarray, kind: str) -> np.ndarray:
    query_binary = (query > 0).astype(np.float32)
    reference_binary = (reference > 0).astype(np.float32)
    if kind == "tversky":
        intersection = query_binary @ reference_binary.T
        query_only = query_binary.sum(1)[:, None] - intersection
        reference_only = reference_binary.sum(1)[None, :] - intersection
        denominator = (
            intersection
            + TVERSKY_QUERY_MISS * query_only
            + TVERSKY_DROPOUT_MISS * reference_only
        )
        return (intersection / np.maximum(denominator, 1e-8)).astype(np.float32)
    if kind == "tfidf":
        idf = np.log((1.0 + len(reference)) /
                     (1.0 + reference_binary.sum(0))) + 1.0
        return _normalised_dot(query_binary * idf, reference_binary * idf).astype(np.float32)
    if kind == "sqrt_count":
        return _normalised_dot(np.sqrt(query), np.sqrt(reference)).astype(np.float32)
    raise ValueError(f"unknown similarity kind: {kind}")


def class_proximity(similarity: np.ndarray, reference_codes: np.ndarray,
                    n_classes: int) -> np.ndarray:
    output = np.zeros((len(similarity), n_classes), dtype=np.float32)
    for code in range(n_classes):
        block = similarity[:, reference_codes == code]
        if block.shape[1] == 0:
            continue
        keep = min(TOP_PER_CLASS, block.shape[1])
        values = np.partition(block, -keep, axis=1)[:, -keep:].mean(1)
        # A fold-training class can have a single released example.  Its own row then
        # has no legal same-class neighbour after diagonal removal; zero is the honest
        # no-evidence value and prevents an infinity entering ExtraTrees.
        output[:, code] = np.where(np.isfinite(values), values, 0.0)
    return output


def broad_label_permutation(y: np.ndarray, meta: pd.DataFrame,
                            rng: np.random.Generator) -> np.ndarray:
    """Permute fine labels within glia/neuron, preserving the dominant easy split."""
    result = y.copy()
    glia = meta["Region"].isna().to_numpy()
    for group in (glia, ~glia):
        rows = np.flatnonzero(group)
        result[rows] = result[rows][rng.permutation(len(rows))]
    return result


def proximity_features(counts_fit: np.ndarray, y_fit: np.ndarray,
                       meta_fit: pd.DataFrame, counts_eval: np.ndarray,
                       classes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray]:
    """Return real/null features for fit and eval, with self-neighbours removed."""
    class_index = {label: index for index, label in enumerate(classes)}
    real_codes = np.asarray([class_index[label] for label in y_fit], dtype=np.int16)
    permuted = broad_label_permutation(y_fit, meta_fit, np.random.default_rng(seed))
    null_codes = np.asarray([class_index[label] for label in permuted], dtype=np.int16)
    real_fit_blocks, real_eval_blocks = [], []
    null_fit_blocks, null_eval_blocks = [], []

    for kind in ("tversky", "tfidf", "sqrt_count"):
        fit_similarity = similarity_matrix(counts_fit, counts_fit, kind)
        np.fill_diagonal(fit_similarity, -np.inf)
        eval_similarity = similarity_matrix(counts_eval, counts_fit, kind)
        real_fit_blocks.append(class_proximity(fit_similarity, real_codes, len(classes)))
        real_eval_blocks.append(class_proximity(eval_similarity, real_codes, len(classes)))
        null_fit_blocks.append(class_proximity(fit_similarity, null_codes, len(classes)))
        null_eval_blocks.append(class_proximity(eval_similarity, null_codes, len(classes)))

    return tuple(np.hstack(blocks).astype(np.float32) for blocks in (
        real_fit_blocks, real_eval_blocks, null_fit_blocks, null_eval_blocks
    ))


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
                x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
                seeds: tuple[int, ...]) -> np.ndarray:
    probabilities = np.zeros((len(x_eval), len(classes)), dtype=np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **M.ET_KWARGS).fit(x_fit, y_fit)
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
    class_array = np.asarray(classes)
    x_base, _ = I15.load_incumbent()
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    names = ["incumbent_694", "+ sparse class proximity", "+ permuted-label proximity (null)"]
    probabilities = {name: np.zeros((len(y), len(classes)), np.float32) for name in names}
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    started = time.time()
    print(f"mode={mode} partition={partition} seeds={len(seeds)}", flush=True)

    for fold, (fit, valid) in enumerate(folds.split(x_base, y), 1):
        fold_started = time.time()
        real_fit, real_valid, null_fit, null_valid = proximity_features(
            counts[fit], y[fit], meta_train.iloc[fit], counts[valid], classes,
            seed=partition * 10 + fold,
        )
        feature_sets = {
            "incumbent_694": (x_base[fit], x_base[valid]),
            "+ sparse class proximity": (
                np.hstack([x_base[fit], real_fit]),
                np.hstack([x_base[valid], real_valid]),
            ),
            "+ permuted-label proximity (null)": (
                np.hstack([x_base[fit], null_fit]),
                np.hstack([x_base[valid], null_valid]),
            ),
        }
        for name, (x_fit, x_valid) in feature_sets.items():
            probabilities[name][valid] = fit_predict(
                x_fit, y[fit], meta_train.iloc[fit], x_valid,
                meta_train.iloc[valid], classes, seeds,
            )
        print(f"  fold {fold}/5 complete in {time.time()-fold_started:.1f}s", flush=True)

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
        })
    result = pd.DataFrame(rows)
    real = result.set_index("config").loc["+ sparse class proximity"]
    null = result.set_index("config").loc["+ permuted-label proximity (null)"]
    if mode == "screen":
        passed = bool(real.gain_pt > 0.30 and real.mcnemar_p < 0.05 and
                      real.gain_pt - null.gain_pt > 0.20)
        result["advance"] = result.config.eq("+ sparse class proximity") & passed
        path = SCREEN_RESULT
    else:
        passed = bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
        result["adopt"] = result.config.eq("+ sparse class proximity") & passed
        path = OUT / "data_geometry_confirm.csv"
    result.to_csv(path, index=False)
    np.savez_compressed(
        OUT / f"data_geometry_{mode}_oof.npz",
        incumbent=probabilities["incumbent_694"],
        sparse_proximity=probabilities["+ sparse class proximity"],
        null_proximity=probabilities["+ permuted-label proximity (null)"],
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"audit", "screen", "confirm", "run"}:
        raise SystemExit("mode must be audit, screen, confirm, or run")
    if mode in {"audit", "run"}:
        audit()
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
