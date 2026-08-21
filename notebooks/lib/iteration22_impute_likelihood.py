"""Iteration 22b -- uncertainty-gated likelihood expert over imputed gene programs.

The first conditional-program experiment appended 64 inferred coordinates to the main
tree and failed its screen.  This follow-up keeps the same frozen atlas-only imputer but
uses the representation in its natural geometry: a shrinkage diagonal Gaussian model of
each cell type in predicted-program space.  A cell's mean predicted program uncertainty
attenuates its evidence, so uncertain imputations become nearly uniform rather than
making brittle class calls.  No challenge label is used to fit the expert.

The algorithm was frozen after the failed appended-feature screen and before opening the
59/83 confirmation partitions.  It is evaluated by the corrected cell-disjoint pool
protocol.  It never reads recovered test labels and never modifies ``prediction/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_models as M
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS
import iteration22_impute_programs as IP


OUT = Path("outputs/iteration22/impute")
CACHE = OUT / "program_likelihood_expert.npz"
SCREEN_RESULT = OUT / "likelihood_screen.csv"
CONFIRM_RESULT = OUT / "likelihood_confirm.csv"
SCREEN_PARTITIONS = (18, 41)
CONFIRM_PARTITIONS = (59, 83)
OUTER_FOLDS = 5
OUTER_SEED = 20260822
SHRINKAGE = 100.0
POOL_L2 = 1e-3
EPS = 1e-9


def adopted_names() -> list[str]:
    return list(json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())[
        "experts"])


def build_likelihood() -> None:
    if CACHE.exists():
        print(f"loaded {CACHE}")
        return
    import torch

    IP.build_representation()
    atlas = IP.load_external_full()
    transform = np.load(IP.TRANSFORM_CACHE, allow_pickle=True)
    x = np.clip((atlas["released"] - transform["input_mean"])
                / transform["input_std"], -8.0, 8.0).astype(np.float32)
    device = IP.torch_device()
    net = IP._network(x.shape[1]).to(device)
    net.load_state_dict(torch.load(IP.MODEL_CACHE, map_location=device, weights_only=True))
    _, atlas_mean, _ = IP.encode(net, x, device)

    challenge = np.load(IP.REP_CACHE, allow_pickle=True)
    challenge_mean = challenge["conditional_mean"].astype(np.float32)
    challenge_logvar = challenge["conditional_logvar"].astype(np.float32)
    classes, labels = atlas["data"]["classes"], atlas["labels"]
    global_variance = atlas_mean.var(0) + 1e-3
    centroids, variances, sizes = [], [], []
    for label in classes:
        rows = labels == label
        sizes.append(int(rows.sum()))
        centroids.append(atlas_mean[rows].mean(0))
        variances.append((atlas_mean[rows].var(0) * rows.sum()
                          + SHRINKAGE * global_variance) / (rows.sum() + SHRINKAGE))
    centroids = np.asarray(centroids, np.float32)
    variances = np.asarray(variances, np.float32)

    # Class likelihood of the conditional mean.  Heteroscedastic uncertainty is used as
    # a scalar reliability temperature rather than added to class variance: the latter
    # double-counts uncertainty because the class distributions are already distributions
    # of conditional means.  Median normalization preserves the global sharpness scale.
    raw = -0.5 * (((challenge_mean[:, None, :] - centroids[None, :, :]) ** 2
                   / variances[None, :, :]) + np.log(variances[None, :, :])).sum(2)
    uncertainty = np.exp(challenge_logvar).mean(1)
    reliability = 1.0 / np.sqrt(np.maximum(uncertainty, 1e-6))
    reliability /= np.median(reliability)
    reliability = np.clip(reliability, 0.35, 2.5)
    probabilities = softmax(raw * reliability[:, None], axis=1).astype(np.float32)
    n_train = len(atlas["data"]["y"])
    np.savez_compressed(
        CACHE, train=probabilities[:n_train], test=probabilities[n_train:],
        classes=classes, centroids=centroids, variances=variances,
        class_sizes=np.asarray(sizes), uncertainty=uncertainty,
        reliability=reliability,
    )
    y = atlas["data"]["y"]
    allow = B.compat_mask(atlas["data"]["meta_train"], y,
                          atlas["data"]["meta_train"], classes)
    prediction = classes[np.where(allow, probabilities[:n_train], -1).argmax(1)]
    glia = atlas["data"]["meta_train"]["Region"].isna().to_numpy()
    mechanism = {
        "standalone_accuracy": float(np.mean(prediction == y)),
        "standalone_glia": float(np.mean(prediction[glia] == y[glia])),
        "standalone_neuron": float(np.mean(prediction[~glia] == y[~glia])),
        "mean_max_probability": float(probabilities[:n_train].max(1).mean()),
        "mean_reliability": float(reliability.mean()),
        "external_cells": int(len(labels)), "test_truth_read": False,
        "challenge_labels_used_to_fit_expert": False,
    }
    (OUT / "likelihood_mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    print("program-likelihood standalone: "
          f"{mechanism['standalone_accuracy']:.4f} "
          f"(glia {mechanism['standalone_glia']:.4f}, "
          f"neuron {mechanism['standalone_neuron']:.4f})", flush=True)
    print(f"wrote {CACHE}")


def partition(seed: int, include: bool):
    logdict, allow, y, classes = SS.part(seed)
    names = adopted_names()
    logs = [logdict[name] for name in names]
    if include:
        build_likelihood()
        p = np.load(CACHE, allow_pickle=True)["train"]
        logs.append(np.log(np.maximum(p, EPS)))
        names.append("impute_program_likelihood")
    return np.stack(logs), names, allow, y, classes


def cell_disjoint(seed: int, include: bool) -> dict:
    logs, names, allow, y, classes = partition(seed, include)
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    prediction = np.empty(len(y), dtype=object)
    fit_records = []
    splits = StratifiedKFold(OUTER_FOLDS, shuffle=True,
                             random_state=OUTER_SEED).split(logs[0], y)
    for fold, (fit, valid) in enumerate(splits, 1):
        record = {"fold": fold}
        for branch, branch_mask in (("glia", glia), ("neuron", ~glia)):
            fit_rows = fit[branch_mask[fit]]
            valid_rows = valid[branch_mask[valid]]
            weights, alpha = LP.fit(logs, y, classes, log_prior, allow,
                                    rows=fit_rows, l2=POOL_L2)
            scores = LP.apply(logs[:, valid_rows], weights, alpha, log_prior,
                              allow[valid_rows])
            prediction[valid_rows] = classes[scores.argmax(1)]
            record[branch] = {
                "prior_exponent": float(alpha),
                "candidate_exponent": float(weights[-1]) if include else 0.0,
                "nonzero": {name: float(value) for name, value in zip(names, weights)
                            if value > 1e-4},
            }
        fit_records.append(record)
    correct = prediction == y
    return {
        "prediction": prediction, "correct": correct,
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "kappa": float(cohen_kappa_score(y, prediction)),
        "records": fit_records, "y": y, "classes": classes,
    }


def evaluate(stage: str) -> pd.DataFrame:
    partitions = SCREEN_PARTITIONS if stage == "screen" else CONFIRM_PARTITIONS
    rows = []
    for seed in partitions:
        base = cell_disjoint(seed, False)
        candidate = cell_disjoint(seed, True)
        p_value, _ = M.paired_mcnemar(candidate["correct"], base["correct"])
        wins = int((candidate["correct"] & ~base["correct"]).sum())
        losses = int((base["correct"] & ~candidate["correct"]).sum())
        exponents = [branch["candidate_exponent"] for fold in candidate["records"]
                     for branch in (fold["glia"], fold["neuron"])]
        rows.append({
            "stage": stage, "partition": seed,
            "base_accuracy": base["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "candidate_balanced_accuracy": candidate["balanced_accuracy"],
            "candidate_kappa": candidate["kappa"],
            "gain_pt": 100 * (candidate["accuracy"] - base["accuracy"]),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
            "mean_candidate_exponent": float(np.mean(exponents)),
        })
        (OUT / f"likelihood_{stage}_partition{seed}_weights.json").write_text(
            json.dumps(candidate["records"], indent=2) + "\n")
        np.savez_compressed(
            OUT / f"likelihood_{stage}_partition{seed}_predictions.npz",
            y=base["y"], base_prediction=base["prediction"],
            candidate_prediction=candidate["prediction"],
            base_correct=base["correct"], candidate_correct=candidate["correct"],
        )
        print(f"{stage} partition {seed}: {base['accuracy']:.4f} -> "
              f"{candidate['accuracy']:.4f} ({rows[-1]['gain_pt']:+.2f} pt), "
              f"{wins}w/{losses}l p={p_value:.4g}, "
              f"mean exponent={np.mean(exponents):.4f}", flush=True)
    frame = pd.DataFrame(rows)
    mean_gain = float(frame.gain_pt.mean())
    passed = bool(mean_gain >= 0.10 and (frame.gain_pt >= 0).all())
    frame["mean_gain_pt"] = mean_gain
    frame["passed"] = passed
    path = SCREEN_RESULT if stage == "screen" else CONFIRM_RESULT
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"VERDICT: {'PASS' if passed else 'REJECT'}")
    return frame


def passed(path: Path) -> bool:
    return path.exists() and bool(pd.read_csv(path)["passed"].astype(bool).all())


def frozen_pool():
    names = adopted_names() + ["impute_program_likelihood"]
    likelihood = np.log(np.maximum(np.load(CACHE, allow_pickle=True)["train"], EPS))
    parts = []
    for seed in SCREEN_PARTITIONS + CONFIRM_PARTITIONS:
        logdict, allow, y, classes = SS.part(seed)
        parts.append((np.stack([logdict[name] for name in names[:-1]] + [likelihood]),
                      allow, y, classes))
    logs = np.concatenate([part[0] for part in parts], axis=1)
    allow = np.concatenate([part[1] for part in parts])
    y = np.concatenate([part[2] for part in parts])
    classes = parts[0][3]
    glia0 = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(parts))
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    fits = {
        "glia": LP.fit(logs, y, classes, log_prior, allow,
                       rows=np.flatnonzero(glia), l2=POOL_L2),
        "neuron": LP.fit(logs, y, classes, log_prior, allow,
                         rows=np.flatnonzero(~glia), l2=POOL_L2),
    }
    return names, fits, classes


def freeze() -> Path:
    if not (passed(SCREEN_RESULT) and passed(CONFIRM_RESULT)):
        raise SystemExit("freeze locked: both frozen stages must pass")
    build_likelihood()
    data = B.load_all()
    names, fits, classes = frozen_pool()
    adopted = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    likelihood = np.load(CACHE, allow_pickle=True)["test"]
    logs = np.stack([np.log(np.maximum(adopted[name], EPS)) for name in names[:-1]]
                    + [np.log(np.maximum(likelihood, EPS))])
    allow = adopted["allow"]
    prior = pd.Series(data["y"]).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    glia = data["meta_test"]["Region"].isna().to_numpy()
    scores = np.zeros((len(glia), len(classes)))
    scores[glia] = LP.apply(logs[:, glia], *fits["glia"], log_prior, allow[glia])
    scores[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], log_prior, allow[~glia])
    prediction = classes[scores.argmax(1)]
    column = pd.read_csv("prediction/prediction.csv", nrows=0).columns[1]
    frame = pd.DataFrame({"Cell_ID": data["meta_test"].index.astype(str),
                          column: prediction})
    text = frame.to_csv(index=False).rstrip("\n")
    output = OUT / "prediction_impute_likelihood.csv"
    output.write_text(text)
    production = pd.read_csv("prediction/prediction.csv", dtype={"Cell_ID": str}).set_index(
        "Cell_ID").iloc[:, 0].reindex(frame.Cell_ID.to_numpy()).to_numpy()
    manifest = {
        "file": str(output), "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "weights": {branch: {name: float(weight) for name, weight in zip(names, fit[0])}
                    for branch, fit in fits.items()},
        "prior_exponents": {branch: float(fit[1]) for branch, fit in fits.items()},
        "changed_vs_production": int((prediction != production).sum()),
        "test_truth_read": False, "production_modified": False,
    }
    (OUT / "likelihood_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}; changed {manifest['changed_vs_production']} rows")
    return output


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"build", "screen", "confirm", "freeze", "run"}:
        raise SystemExit("mode must be build, screen, confirm, freeze, or run")
    if mode in {"build", "run"}:
        build_likelihood()
    if mode in {"screen", "run"}:
        evaluate("screen")
    if mode == "confirm" and not passed(SCREEN_RESULT):
        raise SystemExit("confirmation locked: likelihood screen failed")
    if mode == "confirm" or (mode == "run" and passed(SCREEN_RESULT)):
        evaluate("confirm")
    if mode == "freeze" or (mode == "run" and passed(SCREEN_RESULT)
                            and passed(CONFIRM_RESULT)):
        freeze()
    elif mode == "run" and not passed(SCREEN_RESULT):
        print("confirmation and freeze skipped because screen failed")
    elif mode == "run" and not passed(CONFIRM_RESULT):
        print("freeze skipped because confirmation failed")


if __name__ == "__main__":
    main()
