"""Log-pool with exponents that vary by class frequency and by cell depth.

The Iteration-18/19 pool uses one exponent per expert per branch.  The withheld-gene
diagnostic shows that the errors we could still win back sit in the mid-agreement band -
a median of 11 of 36 experts are right on them, against 3 for the errors that need the
withheld panel - and that they skew towards sparse cells (error rate 0.303 in the lowest
transcript-depth quintile against 0.098 in the highest).  Fixed exponents cannot express
"trust the 136,612-cell reference model more when this particular cell is sparse, or when
this particular class is rare"; a gate linear in those two quantities can, and keeps the
fit convex:

    z(i, c) = sum_m [ w_m + v_m * l_c + u_m * d_i ] * log p_m(i, c)  -  a * l_c

with l_c the standardised log class prior and d_i the standardised log transcript depth.
Fitted by out-of-fold likelihood; selected by two-way held-out accuracy.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_subsets as SS

EPS = 1e-9


def gates(data, classes, y_prior_source):
    counts = data["counts_train"].to_numpy()
    depth = np.log1p(counts.sum(1))
    d = ((depth - depth.mean()) / (depth.std() + 1e-9)).astype(np.float64)
    prior = pd.Series(y_prior_source).value_counts(normalize=True).reindex(
        classes).fillna(EPS).to_numpy()
    l = np.log(prior)
    l = (l - l.mean()) / (l.std() + 1e-9)
    return d, l, np.log(prior)


def _nll(theta, logs, yi, l_c, d_i, log_prior, block, use_v, use_u, l2, l2_int):
    M = logs.shape[0]
    w = theta[:M]
    k = M
    v = theta[k:k + M] if use_v else np.zeros(M); k += M if use_v else 0
    u = theta[k:k + M] if use_u else np.zeros(M); k += M if use_u else 0
    a = theta[k]
    coef = (w[:, None, None] + v[:, None, None] * l_c[None, None, :]
            + u[:, None, None] * d_i[None, :, None])          # (M, n, C)
    z = (coef * logs).sum(0) - a * log_prior[None, :] + block
    lse = logsumexp(z, axis=1)
    rows = np.arange(len(yi))
    val = float(-np.mean(z[rows, yi] - lse)
                + l2 * np.sum(w ** 2) + l2 * a ** 2
                + l2_int * (np.sum(v ** 2) + np.sum(u ** 2)))
    p = np.exp(z - lse[:, None])
    resid = p.copy(); resid[rows, yi] -= 1.0
    n = len(yi)
    gw = np.einsum("mnc,nc->m", logs, resid) / n + 2 * l2 * w
    grad = [gw]
    if use_v:
        grad.append(np.einsum("mnc,nc,c->m", logs, resid, l_c) / n + 2 * l2_int * v)
    if use_u:
        grad.append(np.einsum("mnc,nc,n->m", logs, resid, d_i) / n + 2 * l2_int * u)
    ga = float(-np.sum(resid * log_prior[None, :]) / n + 2 * l2 * a)
    grad.append(np.array([ga]))
    return val, np.concatenate(grad)


def fit(logs, y, classes, l_c, d_i, log_prior, allow, rows=None,
        use_v=True, use_u=True, l2=1e-3, l2_int=1e-2):
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    block = -50.0 * (~allow)
    if rows is not None:
        logs, yi, block, d_i = logs[:, rows], yi[rows], block[rows], d_i[rows]
    M = logs.shape[0]
    x0 = [np.full(M, 1.0 / M)]
    bounds = [(0.0, 3.0)] * M
    if use_v:
        x0.append(np.zeros(M)); bounds += [(-1.0, 1.0)] * M
    if use_u:
        x0.append(np.zeros(M)); bounds += [(-1.0, 1.0)] * M
    x0.append(np.array([0.4])); bounds += [(0.0, 1.5)]
    res = minimize(_nll, np.concatenate(x0),
                   args=(logs, yi, l_c, d_i, log_prior, block, use_v, use_u, l2, l2_int),
                   method="L-BFGS-B", jac=True, bounds=bounds)
    k = M
    v = res.x[k:k + M] if use_v else np.zeros(M); k += M if use_v else 0
    u = res.x[k:k + M] if use_u else np.zeros(M); k += M if use_u else 0
    return res.x[:M], v, u, res.x[k]


def apply(logs, w, v, u, a, l_c, d_i, log_prior, allow):
    coef = (w[:, None, None] + v[:, None, None] * l_c[None, None, :]
            + u[:, None, None] * d_i[None, :, None])
    return (coef * logs).sum(0) - a * log_prior[None, :] + (-1e9 * (~allow))


def evaluate(names, use_v, use_u, l2_int=1e-2):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    gains = []
    for fitseeds, evalseeds in SS.SPLITS:
        logs = np.concatenate([np.stack([SS.part(s)[0][n] for n in names])
                               for s in fitseeds], axis=1)
        allow = np.concatenate([SS.part(s)[1] for s in fitseeds], axis=0)
        y = np.concatenate([SS.part(s)[2] for s in fitseeds])
        classes = SS.part(fitseeds[0])[3]
        d1, l_c, log_prior = gates(data, classes, y)
        d_i = np.tile(d1, len(fitseeds))
        gl = np.tile(glia, len(fitseeds))
        fits = {}
        for tag, rr in (("glia", np.flatnonzero(gl)), ("neuron", np.flatnonzero(~gl))):
            fits[tag] = fit(logs, y, classes, l_c, d_i, log_prior, allow, rows=rr,
                            use_v=use_v, use_u=use_u, l2_int=l2_int)
        for s in evalseeds:
            lgd, al, yy, cl = SS.part(s)
            lg = np.stack([lgd[n] for n in names])
            d2, l2c, lp2 = gates(data, cl, yy)
            base = float(np.mean(cl[np.where(
                al, B.prior_correct(np.exp(lgd["et"]), yy, cl), -1).argmax(1)] == yy))
            z = np.zeros((len(yy), len(cl)))
            z[glia] = apply(lg[:, glia], *fits["glia"], l2c, d2[glia], lp2, al[glia])
            z[~glia] = apply(lg[:, ~glia], *fits["neuron"], l2c, d2[~glia], lp2, al[~glia])
            gains.append(100 * (float(np.mean(cl[z.argmax(1)] == yy)) - base))
    return float(np.mean(gains)), float(np.min(gains)), gains


def main():
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common if n != "rank")
    rows = []
    for tag, (uv, uu) in {"fixed": (False, False), "+prior": (True, False),
                          "+depth": (False, True), "+both": (True, True)}.items():
        for li in ((1e-2,) if not (uv or uu) else (3e-2, 1e-2, 3e-3)):
            m, mn, g = evaluate(names, uv, uu, li)
            rows.append({"gate": tag, "l2_int": li, "mean_gain": m, "worst": mn,
                         "gains": " ".join(f"{x:.2f}" for x in g)})
            print(f"  {tag:7s} l2={li:<6g} mean {m:.3f} worst {mn:.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).sort_values("mean_gain", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
