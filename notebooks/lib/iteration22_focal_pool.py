"""Iteration 22: accuracy-oriented focal log-opinion pool.

The production pool learns non-negative expert exponents with ordinary log loss.  That
is a proper calibration objective, but the competition is ranked by argmax accuracy.
This experiment replaces it with focal cross-entropy so already-easy cells stop
dominating the global exponents.  Hyperparameters are selected on partitions 18/41
after fitting 59/83, then evaluated once on sealed partitions 59/83 after fitting 18/41.
Recovered test labels are never imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration5_models as M

OUT = Path("outputs/iteration22/focal")
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = (18, 41)
CONFIRM = (59, 83)
GAMMAS = (0.5, 1.0, 2.0)
L2 = 1e-3
EPS = 1e-9


def names() -> list[str]:
    return list(json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())["experts"])


def part(seed: int):
    logs, used, allow, y, classes = LP.load_partition(seed, names())
    if used != names():
        raise ValueError(f"partition {seed} lacks an adopted expert")
    return logs.astype(np.float64), allow, y, classes


def _loss(theta, logs, yi, log_prior, block, gamma, l2):
    w, alpha = theta[:-1], theta[-1]
    z = np.tensordot(w, logs, axes=(0, 0)) - alpha * log_prior[None, :] + block
    logden = logsumexp(z, axis=1)
    rows = np.arange(len(yi))
    logpt = z[rows, yi] - logden
    pt = np.exp(logpt)
    one_minus = np.maximum(1.0 - pt, 1e-12)
    ce = -logpt
    loss = np.mean((one_minus ** gamma) * ce) + l2 * np.sum(theta ** 2)
    p = np.exp(z - logden[:, None])
    onehot = np.zeros_like(p)
    onehot[rows, yi] = 1.0
    scale = one_minus ** gamma + gamma * pt * one_minus ** (gamma - 1.0) * ce
    residual = scale[:, None] * (p - onehot)
    grad_w = np.einsum("mnc,nc->m", logs, residual) / len(yi) + 2 * l2 * w
    grad_alpha = -np.sum(residual * log_prior[None, :]) / len(yi) + 2 * l2 * alpha
    return float(loss), np.append(grad_w, grad_alpha)


def fit(logs, y, classes, log_prior, allow, rows, gamma):
    lookup = {label: i for i, label in enumerate(classes)}
    yi = np.array([lookup[label] for label in y])[rows]
    lg = logs[:, rows]
    block = -50.0 * (~allow[rows])
    initial_w, initial_a = LP.fit(logs, y, classes, log_prior, allow, rows=rows, l2=L2)
    initial = np.append(initial_w, initial_a)
    result = minimize(
        _loss, initial, args=(lg, yi, log_prior, block, gamma, L2),
        method="L-BFGS-B", jac=True,
        bounds=[(0.0, 3.0)] * logs.shape[0] + [(0.0, 1.5)],
        options={"maxiter": 100, "ftol": 1e-9},
    )
    if not result.success:
        print(f"warning: focal fit stopped with {result.message}")
    return result.x[:-1], float(result.x[-1])


def train(fit_seeds, gamma, focal):
    pieces = [part(seed) for seed in fit_seeds]
    logs = np.concatenate([piece[0] for piece in pieces], axis=1)
    allow = np.concatenate([piece[1] for piece in pieces])
    y = np.concatenate([piece[2] for piece in pieces])
    classes = pieces[0][3]
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    glia0 = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(pieces))
    fits = {}
    for branch, rows in (("glia", np.flatnonzero(glia)), ("neuron", np.flatnonzero(~glia))):
        fits[branch] = (fit if focal else LP.fit)(
            logs, y, classes, log_prior, allow, rows=rows,
            **({"gamma": gamma} if focal else {"l2": L2}),
        )
    return fits


def predict(seed, fits):
    logs, allow, y, classes = part(seed)
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    lp = np.log(prior)
    glia = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    scores = np.zeros((len(y), len(classes)))
    scores[glia] = LP.apply(logs[:, glia], *fits["glia"], lp, allow[glia])
    scores[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], lp, allow[~glia])
    prediction = classes[scores.argmax(1)]
    return prediction, y


def compare(fit_seeds, eval_seeds, gamma):
    baseline_fit = train(fit_seeds, gamma, False)
    focal_fit = train(fit_seeds, gamma, True)
    rows = []
    for seed in eval_seeds:
        base, y = predict(seed, baseline_fit)
        candidate, _ = predict(seed, focal_fit)
        wins = int(np.sum((candidate == y) & (base != y)))
        losses = int(np.sum((base == y) & (candidate != y)))
        rows.append({
            "partition": seed, "gamma": gamma,
            "base_accuracy": np.mean(base == y),
            "candidate_accuracy": np.mean(candidate == y),
            "gain_pt": 100 * (np.mean(candidate == y) - np.mean(base == y)),
            "balanced_accuracy": balanced_accuracy_score(y, candidate),
            "kappa": cohen_kappa_score(y, candidate),
            "wins": wins, "losses": losses,
            "mcnemar_p": M.paired_mcnemar(candidate == y, base == y)[0],
        })
    return rows


def screen():
    rows = []
    for gamma in GAMMAS:
        result = compare(CONFIRM, SCREEN, gamma)
        rows.extend(result)
        gain = np.mean([row["gain_pt"] for row in result])
        print(f"gamma={gamma:.1f}: " + ", ".join(
            f"p{row['partition']} {row['gain_pt']:+.2f}pt" for row in result
        ) + f"; mean {gain:+.2f}pt", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("gamma").gain_pt.agg(["mean", "min"]).reset_index()
    winner = summary.sort_values(["mean", "min"], ascending=False).iloc[0]
    selected = float(winner.gamma)
    passed = bool(winner["mean"] >= 0.10 and winner["min"] >= 0.0)
    frame["selected_gamma"] = selected
    frame["passed"] = passed
    frame.to_csv(OUT / "screen.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"selected gamma={selected:g}; {'PASS' if passed else 'REJECT'}")
    return frame


def confirm():
    path = OUT / "screen.csv"
    if not path.exists():
        raise SystemExit("run screen first")
    screened = pd.read_csv(path)
    if not screened.passed.astype(bool).all():
        raise SystemExit("confirmation locked: screen rejected")
    gamma = float(screened.selected_gamma.iloc[0])
    rows = compare(SCREEN, CONFIRM, gamma)
    frame = pd.DataFrame(rows)
    passed = bool(frame.gain_pt.mean() >= 0.10 and (frame.gain_pt >= 0).all())
    frame["mean_gain_pt"] = frame.gain_pt.mean()
    frame["passed"] = passed
    frame.to_csv(OUT / "confirm.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"{'PASS' if passed else 'REJECT'}")
    return frame


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        screen()
    elif mode == "confirm":
        confirm()
    else:
        raise SystemExit("mode must be screen or confirm")


if __name__ == "__main__":
    main()
