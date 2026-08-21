"""Iteration 22: a dropout-pattern expert for the sparse 200-gene panel.

Multinomial NB models where transcript mass lands but not the probability that a gene is
observed at all.  This complementary Bernoulli component models the zero/nonzero pattern
and is offered as a separate expert to the calibrated log pool.  Pool weights are always
fitted on cells disjoint from the cells being evaluated; test truth is never imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS
import iteration5_models as M

OUT = Path("outputs/iteration22/hurdle_nb")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = (18, 41)
CONFIRM = (59, 83)
EPS = 1e-9
ALPHA = 0.5


def adopted():
    return list(json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())["experts"])


def align(model, matrix, classes):
    raw = model.predict_proba(matrix)
    out = np.zeros((len(raw), len(classes)), np.float32)
    lookup = {label: i for i, label in enumerate(classes)}
    for j, label in enumerate(model.classes_):
        out[:, lookup[str(label)]] = raw[:, j]
    return out


def build(seed):
    path = OUT / f"bnb_oof_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    data = B.load_all()
    x = (data["counts_train"].to_numpy() > 0).astype(np.uint8)
    y, classes = data["y"], data["classes"]
    out = np.zeros((len(y), len(classes)), np.float32)
    splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit, valid in splitter.split(x, y):
        model = BernoulliNB(alpha=ALPHA).fit(x[fit], y[fit])
        out[valid] = align(model, x[valid], classes)
    np.save(path, out)
    print(f"seed {seed}: Bernoulli standalone {np.mean(classes[out.argmax(1)] == y):.4f}")
    return out


def partition(seed, include):
    logdict, allow, y, classes = SS.part(seed)
    used = adopted()
    logs = [logdict[name] for name in used]
    if include:
        logs.append(np.log(np.maximum(build(seed), EPS)))
        used.append("bernoulli_dropout")
    return np.stack(logs), used, allow, y, classes


def nested(seed, include):
    logs, used, allow, y, classes = partition(seed, include)
    glia = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    lp = np.log(pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy())
    prediction = np.empty(len(y), dtype=object)
    splitter = StratifiedKFold(5, shuffle=True, random_state=20260822)
    weights = []
    for fold, (fit_rows, valid) in enumerate(splitter.split(logs[0], y), 1):
        record = {"fold": fold}
        for branch, mask in (("glia", glia), ("neuron", ~glia)):
            fit = fit_rows[mask[fit_rows]]
            score_rows = valid[mask[valid]]
            w, a = LP.fit(logs, y, classes, lp, allow, rows=fit, l2=1e-3)
            scores = LP.apply(logs[:, score_rows], w, a, lp, allow[score_rows])
            prediction[score_rows] = classes[scores.argmax(1)]
            record[branch] = {"prior": float(a), "bnb_weight": float(w[-1]) if include else None}
        weights.append(record)
    return prediction, y, weights


def evaluate(stage):
    seeds = SCREEN if stage == "screen" else CONFIRM
    rows = []
    for seed in seeds:
        base, y, _ = nested(seed, False)
        pred, _, weights = nested(seed, True)
        wins = int(np.sum((pred == y) & (base != y)))
        losses = int(np.sum((base == y) & (pred != y)))
        rows.append({
            "stage": stage, "partition": seed,
            "base_accuracy": np.mean(base == y), "candidate_accuracy": np.mean(pred == y),
            "gain_pt": 100 * (np.mean(pred == y) - np.mean(base == y)),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "kappa": cohen_kappa_score(y, pred), "wins": wins, "losses": losses,
            "mcnemar_p": M.paired_mcnemar(pred == y, base == y)[0],
            "mean_bnb_weight": np.mean([f[b]["bnb_weight"] for f in weights for b in ("glia", "neuron")]),
        })
    frame = pd.DataFrame(rows)
    passed = bool(frame.gain_pt.mean() >= 0.10 and (frame.gain_pt >= 0).all())
    frame["mean_gain_pt"] = frame.gain_pt.mean()
    frame["passed"] = passed
    frame.to_csv(OUT / f"{stage}.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("PASS" if passed else "REJECT")
    return passed


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if stage == "screen":
        evaluate(stage)
    elif stage == "confirm":
        screen = OUT / "screen.csv"
        if not screen.exists() or not pd.read_csv(screen).passed.astype(bool).all():
            raise SystemExit("confirmation locked: screen rejected")
        evaluate(stage)
    else:
        raise SystemExit("stage must be screen or confirm")


if __name__ == "__main__":
    main()
