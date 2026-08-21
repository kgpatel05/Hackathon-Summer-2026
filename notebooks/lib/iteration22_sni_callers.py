"""Iteration 22 - constituent SNI annotators as calibrated pool experts.

The adopted 40-expert pool contains two models trained on SNI's consensus ``voting``
label, but none trained on its four constituent annotations.  Iteration 5 concatenated
all four posterior blocks as ExtraTrees features and lost accuracy; that experiment
predated the likelihood-fitted log pool whose purpose is to absorb expert sharpness and
set unhelpful exponents to zero.

This module fits one external-only logistic expert for each of ``seurat``, ``rctd``,
``tangram`` and ``singler``.  No challenge label enters those fits.  It then compares the
exact adopted 40-member set with the same set plus all four callers under the corrected
cell-disjoint protocol: pool weights are fitted on four fifths of challenge-training
cells and scored on the fifth.  Partitions 18/41 are the frozen screen; 59/83 are sealed
confirmation.  A candidate is frozen for test only if both stages pass.

Usage:
  python3 notebooks/lib/iteration22_sni_callers.py build
  python3 notebooks/lib/iteration22_sni_callers.py screen
  python3 notebooks/lib/iteration22_sni_callers.py confirm
  python3 notebooks/lib/iteration22_sni_callers.py freeze
  python3 notebooks/lib/iteration22_sni_callers.py run
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS


OUT = Path("outputs/iteration22/sni")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "sni_constituent_probabilities.npz"
SCREEN_RESULT = OUT / "screen.csv"
CONFIRM_RESULT = OUT / "confirm.csv"
CALLERS = ("seurat", "rctd", "tangram", "singler")
SCREEN_PARTITIONS = (18, 41)
CONFIRM_PARTITIONS = (59, 83)
OUTER_FOLDS = 5
OUTER_SEED = 20260821
L2 = 1e-3
EPS = 1e-9


def filtered_reference_transfer(gene_order: np.ndarray, classes: np.ndarray,
                                matrices: list[pd.DataFrame], label_column: str,
                                C: float = 0.1) -> tuple[list[np.ndarray], int, int]:
    """Fit a caller after discarding its labels outside the challenge taxonomy.

    The constituent algorithms do not all emit the challenge's exact 60-class
    vocabulary.  Their out-of-taxonomy rows cannot be aligned and are therefore
    excluded, just as missing/NA reference labels are.  This uses no challenge label.
    """
    reference, labels = F.load_reference(gene_order, label_column)
    keep = np.isin(labels, classes)
    model = LogisticRegression(C=C, max_iter=2000, n_jobs=1)
    model.fit(F.zscore(F.log_cpm(reference[keep])), labels[keep])
    index = {label: i for i, label in enumerate(classes)}
    outputs = []
    for counts in matrices:
        raw = model.predict_proba(F.zscore(F.log_cpm(counts.to_numpy())))
        aligned = np.zeros((len(raw), len(classes)), np.float32)
        for j, label in enumerate(model.classes_):
            aligned[:, index[label]] = raw[:, j]
        outputs.append(aligned)
    return outputs, int(keep.sum()), int((~keep).sum())


def adopted_names() -> list[str]:
    manifest = json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())
    return list(manifest["experts"])


def build_callers() -> None:
    if CACHE.exists():
        print(f"loaded {CACHE}")
        return
    data = B.load_all()
    matrices = [data["counts_train"], data["counts_test"]]
    payload: dict[str, np.ndarray] = {
        "classes": data["classes"], "y": data["y"],
    }
    for caller in CALLERS:
        started = time.time()
        (train, test), retained, excluded = filtered_reference_transfer(
            data["genes"], data["classes"], matrices,
            label_column=caller, C=0.1,
        )
        payload[f"{caller}_train"] = train.astype(np.float32)
        payload[f"{caller}_test"] = test.astype(np.float32)
        prediction = data["classes"][train.argmax(1)]
        print(f"{caller:8s}: ref rows={retained:5d}, excluded={excluded:4d}, "
              f"standalone={np.mean(prediction == data['y']):.4f} "
              f"({time.time()-started:.1f}s)", flush=True)
    np.savez_compressed(CACHE, **payload)
    print(f"wrote {CACHE}")


def partition(seed: int, include_callers: bool) -> tuple[np.ndarray, list[str], np.ndarray,
                                                          np.ndarray, np.ndarray]:
    logdict, allow, y, classes = SS.part(seed)
    base = adopted_names()
    missing = [name for name in base if name not in logdict]
    if missing:
        raise ValueError(f"partition {seed} is missing adopted experts: {missing}")
    chosen = base.copy()
    selected = [logdict[name] for name in chosen]
    if include_callers:
        build_callers()
        callers = np.load(CACHE, allow_pickle=True)
        for name in CALLERS:
            selected.append(np.log(np.maximum(callers[f"{name}_train"], EPS)))
            chosen.append(f"sni_{name}")
    return np.stack(selected), chosen, allow, y, classes


def cell_disjoint(seed: int, include_callers: bool) -> dict:
    logs, names, allow, y, classes = partition(seed, include_callers)
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    predictions = np.empty(len(y), dtype=object)
    fold_weights: list[dict] = []
    splitter = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    for fold, (fit, valid) in enumerate(splitter.split(logs[0], y), 1):
        record = {"fold": fold}
        for branch, mask in (("glia", glia), ("neuron", ~glia)):
            fit_rows = fit[mask[fit]]
            valid_rows = valid[mask[valid]]
            weights, alpha = LP.fit(
                logs, y, classes, log_prior, allow, rows=fit_rows, l2=L2
            )
            scores = LP.apply(logs[:, valid_rows], weights, alpha, log_prior,
                              allow[valid_rows])
            predictions[valid_rows] = classes[scores.argmax(1)]
            record[branch] = {
                "prior_exponent": float(alpha),
                "nonzero": {name: float(weight) for name, weight in zip(names, weights)
                            if weight > 1e-4},
            }
        fold_weights.append(record)
    correct = predictions == y
    return {
        "partition": seed,
        "include_callers": include_callers,
        "prediction": predictions,
        "correct": correct,
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y, predictions)),
        "fold_weights": fold_weights,
        "y": y,
        "classes": classes,
    }


def evaluate(stage: str) -> pd.DataFrame:
    partitions = SCREEN_PARTITIONS if stage == "screen" else CONFIRM_PARTITIONS
    rows, paired = [], []
    for seed in partitions:
        started = time.time()
        base = cell_disjoint(seed, False)
        candidate = cell_disjoint(seed, True)
        p_value, _ = M.paired_mcnemar(candidate["correct"], base["correct"])
        wins = int((candidate["correct"] & ~base["correct"]).sum())
        losses = int((base["correct"] & ~candidate["correct"]).sum())
        rows.append({
            "stage": stage, "partition": seed,
            "base_accuracy": base["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "candidate_balanced_accuracy": candidate["balanced_accuracy"],
            "candidate_kappa": candidate["cohen_kappa"],
            "gain_pt": 100 * (candidate["accuracy"] - base["accuracy"]),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
            "seconds": time.time() - started,
        })
        paired.append((base, candidate))
        print(f"{stage} partition {seed}: {base['accuracy']:.4f} -> "
              f"{candidate['accuracy']:.4f} ({rows[-1]['gain_pt']:+.2f} pt), "
              f"{wins}w/{losses}l p={p_value:.4g}", flush=True)

    frame = pd.DataFrame(rows)
    mean_gain = float(frame.gain_pt.mean())
    # The tactical target is nine cells, but a method must deliver at least five net
    # cells per 5,000 on average and never reverse on a frozen partition.
    threshold = 0.10
    passed = bool(mean_gain >= threshold and (frame.gain_pt >= 0).all())
    frame["mean_gain_pt"] = mean_gain
    frame["passed"] = passed
    path = SCREEN_RESULT if stage == "screen" else CONFIRM_RESULT
    frame.to_csv(path, index=False)
    for base, candidate in paired:
        np.savez_compressed(
            OUT / f"{stage}_partition{base['partition']}.npz",
            y=base["y"], classes=base["classes"],
            base_prediction=base["prediction"],
            candidate_prediction=candidate["prediction"],
            base_correct=base["correct"], candidate_correct=candidate["correct"],
        )
        (OUT / f"{stage}_partition{base['partition']}_weights.json").write_text(
            json.dumps(candidate["fold_weights"], indent=2) + "\n"
        )
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"VERDICT: {'PASS' if passed else 'REJECT'}")
    return frame


def passed(path: Path) -> bool:
    return path.exists() and bool(pd.read_csv(path).passed.astype(bool).all())


def frozen_fit() -> tuple[list[str], dict, np.ndarray]:
    parts = [partition(seed, True) for seed in SCREEN_PARTITIONS + CONFIRM_PARTITIONS]
    names = parts[0][1]
    logs = np.concatenate([part[0] for part in parts], axis=1)
    allow = np.concatenate([part[2] for part in parts])
    y = np.concatenate([part[3] for part in parts])
    classes = parts[0][4]
    glia0 = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(parts))
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    fits = {
        "glia": LP.fit(logs, y, classes, log_prior, allow,
                       rows=np.flatnonzero(glia), l2=L2),
        "neuron": LP.fit(logs, y, classes, log_prior, allow,
                         rows=np.flatnonzero(~glia), l2=L2),
    }
    return names, fits, classes


def freeze() -> Path:
    if not (passed(SCREEN_RESULT) and passed(CONFIRM_RESULT)):
        raise SystemExit("freeze locked: both train-only stages must pass")
    build_callers()
    data = B.load_all()
    names, fits, classes = frozen_fit()
    experts = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    callers = np.load(CACHE, allow_pickle=True)
    logs = []
    for name in names:
        if name.startswith("sni_"):
            probs = callers[f"{name[4:]}_test"]
        else:
            probs = experts[name]
        logs.append(np.log(np.maximum(probs, EPS)))
    logs = np.stack(logs)
    allow = experts["allow"]
    prior = pd.Series(data["y"]).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    glia = data["meta_test"]["Region"].isna().to_numpy()
    scores = np.zeros((len(glia), len(classes)))
    scores[glia] = LP.apply(logs[:, glia], *fits["glia"], log_prior, allow[glia])
    scores[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], log_prior, allow[~glia])
    prediction = classes[scores.argmax(1)]
    output = OUT / "prediction_sni_callers.csv"
    column = pd.read_csv("prediction/prediction.csv", nrows=0).columns[1]
    frame = pd.DataFrame({"Cell_ID": data["meta_test"].index.astype(str),
                          column: prediction})
    text = frame.to_csv(index=False).rstrip("\n")
    output.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    production = pd.read_csv("prediction/prediction.csv", dtype={"Cell_ID": str}).set_index(
        "Cell_ID").iloc[:, 0].reindex(frame.Cell_ID.to_numpy()).to_numpy()
    manifest = {
        "file": str(output), "sha256": digest, "experts": names,
        "weights": {branch: {name: float(weight) for name, weight in zip(names, fit[0])}
                    for branch, fit in fits.items()},
        "prior_exponents": {branch: float(fit[1]) for branch, fit in fits.items()},
        "screen_passed": True, "confirm_passed": True,
        "test_truth_read": False, "production_modified": False,
        "changed_vs_production": int((prediction != production).sum()),
    }
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}; changed {manifest['changed_vs_production']} rows; sha256 {digest}")
    return output


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"build", "screen", "confirm", "freeze", "run"}:
        raise SystemExit("mode must be build, screen, confirm, freeze, or run")
    if mode in {"build", "run"}:
        build_callers()
    if mode in {"screen", "run"}:
        evaluate("screen")
    if mode == "confirm" and not passed(SCREEN_RESULT):
        raise SystemExit("confirmation locked: screen failed")
    if mode == "confirm" or (mode == "run" and passed(SCREEN_RESULT)):
        evaluate("confirm")
    if mode == "freeze" or (mode == "run" and passed(SCREEN_RESULT) and passed(CONFIRM_RESULT)):
        freeze()
    elif mode == "run" and not passed(SCREEN_RESULT):
        print("confirmation and freeze skipped because screen failed")
    elif mode == "run" and not passed(CONFIRM_RESULT):
        print("freeze skipped because confirmation failed")


if __name__ == "__main__":
    main()
