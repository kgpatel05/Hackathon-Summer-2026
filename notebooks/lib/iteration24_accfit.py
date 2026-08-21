"""Fit the pool exponents to a smooth surrogate of ACCURACY, not to likelihood.

Every exponent in this project has been fitted by maximising out-of-fold multinomial
log-likelihood.  That objective is unbounded below, so it pays enormous attention to cells
where an expert is confidently wrong and almost none to whether the argmax is right.  Two
findings say this matters here:

  * the semi-supervised experts score 0.8106-0.8108 standalone and receive exponent
    exactly 0.000, because entropy minimisation makes their rare errors cost ~14 nats;
  * flooring the posteriors to bound that penalty makes the pool *worse*, so the fix has
    to be on the objective rather than on the inputs.

The competition metric is 0/1 accuracy.  This fits

    L(w, a) = mean_i  softmax( z_i / tau )[ y_i ]

which is a temperature-smoothed top-1 rate: bounded in [0, 1], so a confidently wrong
expert costs at most one cell, exactly as it does in the real metric.  Non-convex, so it is
initialised from the likelihood solution and only accepted if it survives the cell-disjoint
protocol.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS

EPS = 1e-9


def _soft_err(theta, logs, yi, log_prior, block, tau, l2):
    w, a = theta[:-1], theta[-1]
    z = (np.tensordot(w, logs, axes=(0, 0)) - a * log_prior[None, :] + block) / tau
    lse = logsumexp(z, axis=1)
    p = np.exp(z - lse[:, None])
    rows = np.arange(len(yi))
    py = p[rows, yi]
    value = float(-np.mean(py) + l2 * np.sum(theta ** 2))
    # d(-mean py)/dz_ic = -py_i (delta_cy - p_ic) / tau
    g = -(p * (-py[:, None]))
    g[rows, yi] += -py * (1.0 - p[rows, yi]) - (-py * (0.0 - p[rows, yi]))
    coef = np.zeros_like(p)
    coef[:, :] = py[:, None] * p
    coef[rows, yi] -= py
    coef /= tau
    n = len(yi)
    grad_w = np.einsum("mnc,nc->m", logs, coef) / n + 2 * l2 * w
    grad_a = float(-np.sum(coef * log_prior[None, :]) / n + 2 * l2 * a)
    return value, np.append(grad_w, grad_a)


def fit_acc(logs, y, classes, log_prior, allow, rows=None, tau=0.5, l2=1e-3,
            init=None):
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    block = -50.0 * (~allow)
    if rows is not None:
        logs, yi, block = logs[:, rows], yi[rows], block[rows]
    M = logs.shape[0]
    if init is None:
        w0, a0 = LP.fit(logs, y[rows] if rows is not None else y, classes, log_prior,
                        allow[rows] if rows is not None else allow, l2=l2)
        init = np.append(w0, a0)
    res = minimize(_soft_err, init, args=(logs, yi, log_prior, block, tau, l2),
                   method="L-BFGS-B", jac=True,
                   bounds=[(0.0, 3.0)] * M + [(0.0, 1.5)])
    return res.x[:-1], res.x[-1]


def run(names, tau, l2=1e-3, partitions=(18, 41, 59, 83), folds=5, seed=2026,
        objective="acc"):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    out = []
    for s in partitions:
        lgd, allow, y, classes = SS.part(s)
        logs = np.stack([lgd[n] for n in names])
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        pred = np.empty(len(y), dtype=object)
        base = np.empty(len(y), dtype=object)
        for fit_idx, val_idx in StratifiedKFold(folds, shuffle=True,
                                                random_state=seed).split(logs[0], y):
            for mask in (glia, ~glia):
                rr = fit_idx[mask[fit_idx]]
                vv = val_idx[mask[val_idx]]
                if len(rr) < 80 or len(vv) == 0:
                    continue
                if objective == "acc":
                    w, a = fit_acc(logs, y, classes, lp, allow, rows=rr, tau=tau, l2=l2)
                else:
                    w, a = LP.fit(logs, y, classes, lp, allow, rows=rr, l2=l2)
                pred[vv] = classes[LP.apply(logs[:, vv], w, a, lp, allow[vv]).argmax(1)]
            base[val_idx] = classes[np.where(
                allow[val_idx], B.prior_correct(np.exp(logs[names.index("et")])[val_idx],
                                                y, classes), -1).argmax(1)]
        out.append(100 * (float(np.mean(pred == y)) - float(np.mean(base == y))))
    return float(np.mean(out)), float(np.min(out)), out


def main():
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common
                   if n not in ("rank", "atlaslam_proto", "xgbaug4", "etaug4_0.08"))
    print(f"{len(names)} experts, cell-disjoint validation\n")
    rows = []
    m, mn, g = run(names, tau=1.0, objective="likelihood")
    rows.append({"objective": "likelihood", "tau": "-", "mean_gain": m, "worst": mn,
                 "gains": " ".join(f"{x:.2f}" for x in g)})
    print(f"  likelihood         mean {m:+.3f} worst {mn:+.3f}", flush=True)
    for tau in (0.25, 0.5, 1.0, 2.0):
        m, mn, g = run(names, tau=tau, objective="acc")
        rows.append({"objective": "soft-accuracy", "tau": tau, "mean_gain": m,
                     "worst": mn, "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  soft-accuracy t={tau:<5g} mean {m:+.3f} worst {mn:+.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).sort_values("mean_gain", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
