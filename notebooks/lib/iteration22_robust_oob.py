"""Iteration 22 - out-of-bag co-teaching for robust train-cell reweighting.

Iteration 17 ranked suspicious cells from molecular nearest-neighbour geometry and
either deleted them or assigned a hand-written soft weight.  This experiment asks a
different question: does a cell's released label survive two independently perturbed,
strictly self-excluded classifiers?  For every outer-fold training set, two bootstrap
ExtraTrees models produce out-of-bag (OOB) probabilities for their own training rows:

* the adopted 694-column representation (expression, metadata, and reference views);
* square-root-transformed released counts only.

The OOB predictions never use a row in the trees that score that row, and the outer
validation fold is absent from both scorers.  Own-label likelihood is converted to a
within-class rank before constructing sample weights, preventing rare classes from
being mistaken for label noise merely because they are rare.  Several predeclared
weight maps are compared on partition 18 only; that choice is frozen before the formal
screen (partition 41) and confirmation (partitions 59/83).

The selected robust ExtraTrees is evaluated both standalone and as one intentionally
diverse expert added to the adopted 40-member log pool.  Pool exponents are themselves
fit on four fifths of cells and scored on the disjoint fifth.  Recovered test labels are
never imported.  A test candidate can be frozen only after both gates pass.

This is a scikit-learn tree workload, for which PyTorch MPS has no implementation; all
tree fits therefore use CPU parallelism.  No tensor workload is hidden in this module.

Usage
-----
``python3 notebooks/lib/iteration22_robust_oob.py tune``
``python3 notebooks/lib/iteration22_robust_oob.py screen``
``python3 notebooks/lib/iteration22_robust_oob.py confirm``
``python3 notebooks/lib/iteration22_robust_oob.py freeze``
``python3 notebooks/lib/iteration22_robust_oob.py run``
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_models as M
import iteration18_base as B
import iteration18_experts2 as E2
import iteration18_logpool2 as LP
import iteration18_subsets as SS


OUT = Path("outputs/iteration22/robust")
OUT.mkdir(parents=True, exist_ok=True)
CHOICE = OUT / "frozen_choice.json"
SCREEN_RESULT = OUT / "screen.csv"
CONFIRM_RESULT = OUT / "confirmation.csv"
TUNE_PARTITION = 18
SCREEN_PARTITION = 41
CONFIRM_PARTITIONS = (59, 83)
POOL_FOLDS = 5
POOL_SEED = 20260821
MODEL_SEEDS = (0, 1)
TEST_SEEDS = tuple(range(20))
N_TREES_SCORE = 800
N_TREES_FINAL = 300
EPS = 1e-9
L2 = 1e-3

# Fixed before looking at any Iteration-22 result.  Rank is the percentile of a
# cell's two-view OOB own-label likelihood within its observed class (0 = least
# credible).  ``conflict`` means both views confidently choose the same other label.
CONFIGS = (
    "gentle_core",       # continuously favour class-prototypical rows
    "soft_small_loss",   # taper only the least credible within-class quartile
    "consensus_noise",   # downweight only high-confidence two-view contradictions
    "hybrid_robust",     # small-loss taper plus the consensus contradiction test
    "hard_boundary",     # matched opposite: moderately upweight hard non-outliers
)


def adopted_names() -> list[str]:
    manifest = json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())
    return list(manifest["experts"])


def align_oob(model: ExtraTreesClassifier, classes: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.oob_decision_function_, dtype=np.float32)
    result = np.full((len(raw), len(classes)), EPS, dtype=np.float32)
    index = {label: column for column, label in enumerate(classes)}
    for column, label in enumerate(model.classes_):
        result[:, index[str(label)]] = raw[:, column]
    result /= np.maximum(result.sum(1, keepdims=True), EPS)
    return result


def oob_view(x: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int,
             leaf: int) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=N_TREES_SCORE,
        max_features="sqrt",
        min_samples_leaf=leaf,
        bootstrap=True,
        max_samples=0.80,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
    ).fit(x, y)
    return align_oob(model, classes)


def class_rank(values: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Stable 0..1 ranks separately within each class."""
    result = np.zeros(len(y), dtype=np.float32)
    for label in classes:
        rows = np.flatnonzero(y == label)
        if len(rows) == 1:
            result[rows] = 1.0
            continue
        order = np.argsort(values[rows], kind="stable")
        rank = np.empty(len(rows), dtype=np.float32)
        rank[order] = np.arange(len(rows), dtype=np.float32) / (len(rows) - 1)
        result[rows] = rank
    return result


def diagnostic_scores(x: np.ndarray, counts: np.ndarray, y: np.ndarray,
                      classes: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Two self-excluded views; all inputs belong to the outer training fold."""
    full = oob_view(x, y, classes, seed=2 * seed, leaf=2)
    # Counts-only is intentionally lower-capacity and orthogonal to the atlas-derived
    # posterior blocks in the 694-column incumbent.
    count_view = np.sqrt(counts).astype(np.float32, copy=False)
    count = oob_view(count_view, y, classes, seed=2 * seed + 1, leaf=4)
    class_index = {label: column for column, label in enumerate(classes)}
    yi = np.asarray([class_index[label] for label in y])
    own = np.sqrt(np.maximum(full[np.arange(len(y)), yi], EPS)
                  * np.maximum(count[np.arange(len(y)), yi], EPS))
    rank = class_rank(own, y, classes)
    full_top = full.argmax(1)
    count_top = count.argmax(1)
    same_wrong = (full_top == count_top) & (full_top != yi)
    confidence = np.minimum(full.max(1), count.max(1))
    conflict = same_wrong & (confidence >= 0.42)
    return {"rank": rank, "own": own.astype(np.float32),
            "conflict": conflict, "confidence": confidence.astype(np.float32)}


def sample_weights(config: str, diagnostics: dict[str, np.ndarray]) -> np.ndarray:
    rank = diagnostics["rank"]
    conflict = diagnostics["conflict"]
    if config == "unweighted_control":
        weights = np.ones(len(rank), dtype=np.float32)
    elif config == "gentle_core":
        weights = 0.60 + 0.80 * rank
    elif config == "soft_small_loss":
        weights = 0.20 + 0.80 * np.clip(rank / 0.25, 0.0, 1.0)
    elif config == "consensus_noise":
        weights = np.ones(len(rank), dtype=np.float32)
        weights[conflict] = 0.15
    elif config == "hybrid_robust":
        weights = 0.25 + 0.75 * np.clip(rank / 0.30, 0.0, 1.0)
        weights[conflict] *= 0.25
    elif config == "hard_boundary":
        # Emphasise learnable hard cells but not the bottom 10%, which are most likely
        # noise or information-starved rather than useful boundary support.
        weights = np.ones(len(rank), dtype=np.float32)
        middle = (rank >= 0.10) & (rank <= 0.45)
        weights[middle] = 1.60
        weights[rank < 0.10] = 0.50
    else:
        raise ValueError(f"unknown config: {config}")
    # Keep the average effective sample size constant across folds/configurations.
    return (weights / np.mean(weights)).astype(np.float32)


def fit_weighted(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray,
                 classes: np.ndarray, weights: np.ndarray,
                 seeds: tuple[int, ...] = MODEL_SEEDS) -> np.ndarray:
    result = np.zeros((len(x_eval), len(classes)), dtype=np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=N_TREES_FINAL, max_features=0.10, min_samples_leaf=3,
            n_jobs=-1, random_state=seed,
        ).fit(x_fit, y_fit, sample_weight=weights)
        result += M.align_proba(model, x_eval, classes.tolist())
    result /= len(seeds)
    result /= np.maximum(result.sum(1, keepdims=True), EPS)
    return result


def build_partition(seed: int, configs: tuple[str, ...]) -> Path:
    cache = OUT / f"robust_aug4_oof_seed{seed}.npz"
    retained: dict[str, np.ndarray] = {}
    if cache.exists():
        old = np.load(cache, allow_pickle=True)
        if all(config in old.files for config in configs):
            print(f"partition {seed}: loaded {cache}", flush=True)
            return cache
        retained = {name: old[name] for name in old.files
                    if name not in ("y", "classes")}
    data = B.load_all()
    x_score = data["x_train"]
    # Reweight the strongest challenge-trained augmented-tree family rather than the
    # plain 694-column ET, whose exponent is already zero in the adopted pool.
    x, _ = E2.augmented4(data, seed)
    counts = data["counts_train"].to_numpy(np.float32)
    y, classes = data["y"], data["classes"]
    missing_configs = tuple(config for config in configs if config not in retained)
    outputs = {**retained, **{
        config: np.zeros((len(y), len(classes)), np.float32)
        for config in missing_configs
    }}
    summary: list[dict] = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    started = time.time()
    for fold, (fit, valid) in enumerate(splitter.split(x, y), 1):
        fold_started = time.time()
        diag = diagnostic_scores(x_score[fit], counts[fit], y[fit], classes,
                                 seed=seed * 10 + fold)
        for config in missing_configs:
            weights = sample_weights(config, diag)
            outputs[config][valid] = fit_weighted(
                x[fit], y[fit], x[valid], classes, weights,
            )
            summary.append({
                "partition": seed, "fold": fold, "config": config,
                "weight_min": float(weights.min()),
                "weight_mean": float(weights.mean()),
                "weight_max": float(weights.max()),
                "effective_n": float(weights.sum() ** 2 / np.sum(weights ** 2)),
                "conflicts": int(diag["conflict"].sum()),
            })
        print(f"partition {seed} fold {fold}/5 in {time.time()-fold_started:.1f}s",
              flush=True)
    np.savez_compressed(cache, y=y, classes=classes, **outputs)
    pd.DataFrame(summary).to_csv(OUT / f"weight_diagnostics_seed{seed}.csv", index=False)
    print(f"wrote {cache} in {time.time()-started:.1f}s", flush=True)
    return cache


def load_candidate(seed: int, config: str) -> np.ndarray:
    path = build_partition(seed, (config,))
    return np.load(path, allow_pickle=True)[config].astype(np.float32)


def pool_prediction(seed: int, candidate: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """Strict cell-disjoint fitting of log-pool parameters for one OOF partition."""
    logdict, allow, y, classes = SS.part(seed)
    names = adopted_names()
    missing = [name for name in names if name not in logdict]
    if missing:
        raise ValueError(f"partition {seed} lacks adopted experts {missing}")
    logs = [logdict[name] for name in names]
    if candidate is not None:
        logs.append(np.log(np.maximum(candidate, EPS)))
        names = names + ["robust_oob"]
    logs_array = np.stack(logs)
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    log_prior = np.log(prior)
    glia = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    prediction = np.empty(len(y), dtype=object)
    robust_weights = []
    splitter = StratifiedKFold(POOL_FOLDS, shuffle=True, random_state=POOL_SEED)
    for fold, (fit, valid) in enumerate(splitter.split(logs_array[0], y), 1):
        fold_record = {"fold": fold}
        for branch, mask in (("glia", glia), ("neuron", ~glia)):
            fit_rows = fit[mask[fit]]
            valid_rows = valid[mask[valid]]
            weights, alpha = LP.fit(logs_array, y, classes, log_prior, allow,
                                    rows=fit_rows, l2=L2)
            scores = LP.apply(logs_array[:, valid_rows], weights, alpha,
                              log_prior, allow[valid_rows])
            prediction[valid_rows] = classes[scores.argmax(1)]
            fold_record[branch] = float(weights[-1]) if candidate is not None else 0.0
        robust_weights.append(fold_record)
    return prediction, {"names": names, "robust_weights": robust_weights,
                        "y": y, "classes": classes}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correct = prediction == y
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "cohen_kappa": float(cohen_kappa_score(y, prediction)),
    }


def compare_pool(seed: int, config: str) -> dict:
    candidate = load_candidate(seed, config)
    base_pred, base_meta = pool_prediction(seed, None)
    candidate_pred, candidate_meta = pool_prediction(seed, candidate)
    y = base_meta["y"]
    base_correct, cand_correct = base_pred == y, candidate_pred == y
    p_value, _ = M.paired_mcnemar(cand_correct, base_correct)
    result = {
        "partition": seed, "config": config,
        "base_accuracy": float(base_correct.mean()),
        "candidate_accuracy": float(cand_correct.mean()),
        "candidate_balanced_accuracy": float(
            balanced_accuracy_score(y, candidate_pred)),
        "candidate_kappa": float(cohen_kappa_score(y, candidate_pred)),
        "gain_pt": float(100 * (cand_correct.mean() - base_correct.mean())),
        "wins": int((cand_correct & ~base_correct).sum()),
        "losses": int((base_correct & ~cand_correct).sum()),
        "mcnemar_p": float(p_value),
        "mean_robust_exponent": float(np.mean([
            record[branch]
            for record in candidate_meta["robust_weights"]
            for branch in ("glia", "neuron")
        ])),
    }
    np.savez_compressed(
        OUT / f"pool_{config}_seed{seed}.npz",
        y=y, classes=base_meta["classes"], base_prediction=base_pred,
        candidate_prediction=candidate_pred, base_correct=base_correct,
        candidate_correct=cand_correct,
    )
    (OUT / f"pool_{config}_weights_seed{seed}.json").write_text(
        json.dumps(candidate_meta["robust_weights"], indent=2) + "\n"
    )
    return result


def standalone_rows(seed: int, configs: tuple[str, ...]) -> list[dict]:
    data = B.load_all()
    y, classes = data["y"], data["classes"]
    _, allow, _, _ = SS.part(seed)
    et = np.exp(SS.part(seed)[0]["et"])
    base = B.decode(B.prior_correct(et, y, classes), allow, classes)
    base_correct = base == y
    rows = []
    for config in configs:
        raw = load_candidate(seed, config)
        pred = B.decode(B.prior_correct(raw, y, classes), allow, classes)
        correct = pred == y
        p_value, _ = M.paired_mcnemar(correct, base_correct)
        score = metrics(y, pred)
        rows.append({
            "partition": seed, "config": config,
            **{f"standalone_{key}": value for key, value in score.items()},
            "standalone_gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "standalone_wins": int((correct & ~base_correct).sum()),
            "standalone_losses": int((base_correct & ~correct).sum()),
            "standalone_mcnemar_p": p_value,
        })
    return rows


def tune() -> pd.DataFrame:
    build_partition(TUNE_PARTITION, CONFIGS)
    standalone = pd.DataFrame(standalone_rows(TUNE_PARTITION, CONFIGS))
    rows = []
    for config in CONFIGS:
        started = time.time()
        row = compare_pool(TUNE_PARTITION, config)
        row["seconds"] = time.time() - started
        rows.append(row)
        print(f"tune {config:16s}: pool {row['base_accuracy']:.4f} -> "
              f"{row['candidate_accuracy']:.4f} ({row['gain_pt']:+.2f} pt), "
              f"{row['wins']}w/{row['losses']}l", flush=True)
    frame = pd.DataFrame(rows).merge(standalone, on=["partition", "config"])
    # Pool gain is primary; standalone gain and name provide deterministic tie-breaks.
    chosen = frame.sort_values(
        ["gain_pt", "standalone_gain_pt", "config"], ascending=[False, False, True]
    ).iloc[0]
    frame["selected"] = frame.config.eq(chosen.config)
    frame.to_csv(OUT / "tuning.csv", index=False)
    CHOICE.write_text(json.dumps({
        "config": str(chosen.config), "tune_partition": TUNE_PARTITION,
        "selection_metric": "cell-disjoint gain when added to adopted 40-expert pool",
        "screen_partition": SCREEN_PARTITION,
        "confirmation_partitions": list(CONFIRM_PARTITIONS),
        "screen_gate": "gain >= 0.10 point and wins > losses",
        "confirmation_gate": "mean gain >= 0.10 point, no negative partition, aggregate wins > losses",
        "test_truth_read": False,
    }, indent=2) + "\n")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"FROZEN CONFIG: {chosen.config}")
    return frame


def chosen_config() -> str:
    if not CHOICE.exists():
        raise RuntimeError("run tune first; no robust configuration has been frozen")
    return str(json.loads(CHOICE.read_text())["config"])


def evaluate_stage(stage: str) -> pd.DataFrame:
    config = chosen_config()
    seeds = (SCREEN_PARTITION,) if stage == "screen" else CONFIRM_PARTITIONS
    rows = []
    for seed in seeds:
        started = time.time()
        row = compare_pool(seed, config)
        row["stage"] = stage
        row["seconds"] = time.time() - started
        rows.append(row)
        print(f"{stage} partition {seed}: {row['base_accuracy']:.4f} -> "
              f"{row['candidate_accuracy']:.4f} ({row['gain_pt']:+.2f} pt), "
              f"{row['wins']}w/{row['losses']}l p={row['mcnemar_p']:.4g}",
              flush=True)
    frame = pd.DataFrame(rows)
    standalone = pd.DataFrame(
        row for seed in seeds for row in standalone_rows(seed, (config,))
    )
    frame = frame.merge(standalone, on=["partition", "config"], how="left")
    if stage == "screen":
        passed = bool(frame.gain_pt.iloc[0] >= 0.10
                      and frame.wins.iloc[0] > frame.losses.iloc[0])
        path = SCREEN_RESULT
    else:
        passed = bool(frame.gain_pt.mean() >= 0.10
                      and (frame.gain_pt >= 0).all()
                      and frame.wins.sum() > frame.losses.sum())
        path = CONFIRM_RESULT
    frame["mean_gain_pt"] = float(frame.gain_pt.mean())
    frame["passed"] = passed
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"VERDICT: {'PASS' if passed else 'REJECT'}")
    return frame


def passed(path: Path) -> bool:
    if not path.exists():
        return False
    return bool(pd.read_csv(path).passed.fillna(False).astype(bool).all())


def freeze() -> None:
    if not (passed(SCREEN_RESULT) and passed(CONFIRM_RESULT)):
        raise RuntimeError("freeze is locked: screen and confirmation must both pass")
    config = chosen_config()
    data = B.load_all()
    y, classes = data["y"], data["classes"]
    x_score = data["x_train"]
    x, x_test = E2.augmented4(data, TUNE_PARTITION)
    counts = data["counts_train"].to_numpy(np.float32)
    diag = diagnostic_scores(x_score, counts, y, classes, seed=202608210)
    weights = sample_weights(config, diag)
    test_raw = fit_weighted(x, y, x_test, classes, weights,
                            seeds=TEST_SEEDS)

    parts = []
    for seed in (TUNE_PARTITION, SCREEN_PARTITION) + CONFIRM_PARTITIONS:
        logdict, allow, yy, cls = SS.part(seed)
        names = adopted_names()
        logs = np.stack([logdict[name] for name in names]
                        + [np.log(np.maximum(load_candidate(seed, config), EPS))])
        parts.append((logs, allow, yy, cls))
    logs = np.concatenate([part[0] for part in parts], axis=1)
    allow = np.concatenate([part[1] for part in parts])
    yy = np.concatenate([part[2] for part in parts])
    prior = pd.Series(yy).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    log_prior = np.log(prior)
    glia0 = data["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(parts))
    fits = {
        branch: LP.fit(logs, yy, classes, log_prior, allow,
                       rows=np.flatnonzero(mask), l2=L2)
        for branch, mask in (("glia", glia), ("neuron", ~glia))
    }

    test_store = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    names = adopted_names() + ["robust_oob"]
    test_logs = np.stack(
        [np.log(np.maximum(test_store[name], EPS)) for name in names[:-1]]
        + [np.log(np.maximum(test_raw, EPS))]
    )
    test_allow = test_store["allow"]
    test_glia = data["meta_test"]["Region"].isna().to_numpy()
    scores = np.zeros((len(test_allow), len(classes)), np.float64)
    for branch, mask in (("glia", test_glia), ("neuron", ~test_glia)):
        scores[mask] = LP.apply(test_logs[:, mask], *fits[branch], log_prior,
                                test_allow[mask])
    prediction = classes[scores.argmax(1)]
    example = pd.read_csv("prediction/prediction.csv", nrows=0)
    output = pd.DataFrame({
        "Cell_ID": data["meta_test"].index.astype(str),
        example.columns[1]: prediction,
    })
    path = OUT / "prediction_robust_oob.csv"
    text = output.to_csv(index=False).rstrip("\n")
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (OUT / "freeze_manifest.json").write_text(json.dumps({
        "config": config, "file": str(path), "sha256": digest,
        "experts": names,
        "robust_exponents": {branch: float(fit[0][-1])
                             for branch, fit in fits.items()},
        "test_truth_read": False, "production_modified": False,
    }, indent=2) + "\n")
    print(f"wrote {path}\nsha256 {digest}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"tune", "screen", "confirm", "freeze", "run"}:
        raise SystemExit("mode must be tune, screen, confirm, freeze, or run")
    if mode in {"tune", "run"}:
        tune()
    if mode in {"screen", "run"}:
        evaluate_stage("screen")
    if mode == "confirm" and not passed(SCREEN_RESULT):
        raise RuntimeError("confirmation is locked: screen did not pass")
    if mode == "confirm" or (mode == "run" and passed(SCREEN_RESULT)):
        evaluate_stage("confirm")
    elif mode == "run":
        print("confirmation not run because the frozen screen gate failed")
    if mode == "freeze":
        freeze()


if __name__ == "__main__":
    main()
