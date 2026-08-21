"""Iteration 22: bag log-pool exponents across independent OOF partitions.

The adopted pool fits one exponent vector to concatenated stochastic OOF realizations.
This candidate instead fits an exponent vector to each realization and averages the
vectors, reducing sensitivity to any one set of base-model errors.  Screen and
confirmation use disjoint partition pairs; recovered test labels are never imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration5_models as M

OUT = Path("outputs/iteration22/weightbag")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = (18, 41)
CONFIRM = (59, 83)
EPS = 1e-9
L2 = 1e-3


def adopted():
    return list(json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())["experts"])


def load(seed):
    return LP.load_partition(seed, adopted())


def fit_pooled(seeds):
    pieces = [load(seed) for seed in seeds]
    logs = np.concatenate([p[0] for p in pieces], axis=1)
    allow = np.concatenate([p[2] for p in pieces])
    y = np.concatenate([p[3] for p in pieces])
    classes = pieces[0][4]
    lp = np.log(pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy())
    glia0 = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(seeds))
    return {
        branch: LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(mask), l2=L2)
        for branch, mask in (("glia", glia), ("neuron", ~glia))
    }


def fit_bagged(seeds):
    by_seed = [fit_pooled((seed,)) for seed in seeds]
    result = {}
    for branch in ("glia", "neuron"):
        weights = np.mean([fit[branch][0] for fit in by_seed], axis=0)
        alpha = float(np.mean([fit[branch][1] for fit in by_seed]))
        result[branch] = weights, alpha
    return result


def predict(seed, fits):
    logs, _, allow, y, classes = load(seed)
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    lp = np.log(prior)
    glia = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    scores = np.zeros((len(y), len(classes)))
    scores[glia] = LP.apply(logs[:, glia], *fits["glia"], lp, allow[glia])
    scores[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], lp, allow[~glia])
    return classes[scores.argmax(1)], y


def evaluate(stage):
    fit_seeds, eval_seeds = (CONFIRM, SCREEN) if stage == "screen" else (SCREEN, CONFIRM)
    baseline = fit_pooled(fit_seeds)
    candidate = fit_bagged(fit_seeds)
    rows = []
    for seed in eval_seeds:
        base, y = predict(seed, baseline)
        pred, _ = predict(seed, candidate)
        wins = int(np.sum((pred == y) & (base != y)))
        losses = int(np.sum((base == y) & (pred != y)))
        rows.append({
            "stage": stage, "partition": seed,
            "base_accuracy": np.mean(base == y), "candidate_accuracy": np.mean(pred == y),
            "gain_pt": 100 * (np.mean(pred == y) - np.mean(base == y)),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "kappa": cohen_kappa_score(y, pred), "wins": wins, "losses": losses,
            "mcnemar_p": M.paired_mcnemar(pred == y, base == y)[0],
        })
    frame = pd.DataFrame(rows)
    passed = bool(frame.gain_pt.mean() >= 0.10 and (frame.gain_pt >= 0).all())
    frame["mean_gain_pt"] = frame.gain_pt.mean()
    frame["passed"] = passed
    frame.to_csv(OUT / f"{stage}.csv", index=False)
    manifest = {
        "stage": stage, "fit_partitions": list(fit_seeds),
        "evaluation_partitions": list(eval_seeds), "passed": passed,
        "baseline_weights": {b: baseline[b][0].tolist() for b in baseline},
        "bagged_weights": {b: candidate[b][0].tolist() for b in candidate},
        "test_truth_read": False,
    }
    (OUT / f"{stage}_weights.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("PASS" if passed else "REJECT")
    return passed


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if stage == "screen":
        evaluate(stage)
    elif stage == "confirm":
        prior = OUT / "screen.csv"
        if not prior.exists() or not pd.read_csv(prior).passed.astype(bool).all():
            raise SystemExit("confirmation locked: screen rejected")
        evaluate(stage)
    else:
        raise SystemExit("stage must be screen or confirm")


if __name__ == "__main__":
    main()
