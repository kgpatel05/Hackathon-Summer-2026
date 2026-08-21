"""Log-pool with optional branch-specific exponents, plus a frozen-weight protocol.

Glia and neurons are two different problems (metadata determines 80% of neurons and
26% of glia), so the experts deserve different exponents on each branch.  Weights are
fitted by out-of-fold likelihood on ONE partition and then evaluated frozen on fresh
partitions; no weight is ever chosen by looking at an accuracy it is scored on.
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
import iteration5_models as M

EPS = 1e-9


def load_partition(seed, names=None):
    d = np.load(B.OUT / f"experts_oof_seed{seed}.npz", allow_pickle=True)
    classes = d["classes"].astype(str)
    avail = sorted(k for k in d.files if k not in ("allow", "y", "classes"))
    use = [n for n in (names or avail) if n in avail]
    logs = np.stack([np.log(np.maximum(d[n], EPS)) for n in use])   # (M, n, C)
    return logs, use, d["allow"], d["y"].astype(str), classes


def _nll(theta, logs, yi, log_prior, block, l2, n_bias=0, l2_bias=0.0):
    """Negative OOF multinomial log-likelihood and its exact gradient.

    With n_bias > 0 the last `n_bias` entries of theta are a per-class logit offset,
    shrunk separately - a likelihood-fitted Dirichlet-style calibration of the pool.
    """
    core = theta[:len(theta) - n_bias] if n_bias else theta
    bias = theta[len(theta) - n_bias:] if n_bias else None
    w, a = core[:-1], core[-1]
    z = np.tensordot(w, logs, axes=(0, 0)) - a * log_prior[None, :] + block
    if n_bias:
        z = z + bias[None, :]
    lse = logsumexp(z, axis=1)
    rows = np.arange(len(yi))
    value = float(-np.mean(z[rows, yi] - lse) + l2 * np.sum(core ** 2))
    if n_bias:
        value += l2_bias * float(np.sum(bias ** 2))
    p = np.exp(z - lse[:, None])
    onehot = np.zeros_like(p)
    onehot[rows, yi] = 1.0
    resid = p - onehot                                   # (n, C)
    grad_w = np.einsum("mnc,nc->m", logs, resid) / len(yi) + 2 * l2 * w
    grad_a = float(-np.sum(resid * log_prior[None, :]) / len(yi) + 2 * l2 * a)
    grad = np.append(grad_w, grad_a)
    if n_bias:
        grad = np.append(grad, resid.mean(0) + 2 * l2_bias * bias)
    return value, grad


def fit(logs, y, classes, log_prior, allow, rows=None, l2=1e-3, bias_l2=None):
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    block = -50.0 * (~allow)          # soft during fitting; hard at decision time
    if rows is not None:
        logs, yi, block = logs[:, rows], yi[rows], block[rows]
    M_ = logs.shape[0]
    C = logs.shape[2]
    nb = C if bias_l2 is not None else 0
    x0 = np.append(np.full(M_, 1.0 / M_), 0.3)
    bounds = [(0.0, 3.0)] * M_ + [(0.0, 1.5)]
    if nb:
        x0 = np.append(x0, np.zeros(C))
        bounds += [(-3.0, 3.0)] * C
    res = minimize(_nll, x0, args=(logs, yi, log_prior, block, l2, nb, bias_l2 or 0.0),
                   method="L-BFGS-B", jac=True, bounds=bounds)
    if nb:
        return res.x[:M_], res.x[M_], res.x[M_ + 1:]
    return res.x[:-1], res.x[-1]


def apply(logs, w, a, log_prior, allow, bias=None):
    z = np.tensordot(w, logs, axes=(0, 0)) - a * log_prior[None, :] + (-1e9 * (~allow))
    return z if bias is None else z + bias[None, :]


def glia_mask(n, split):
    import iteration5_features as F
    _, meta_train, _, _ = F.load_challenge()
    return meta_train["Region"].isna().to_numpy()


def main(fit_seed, eval_seeds, names=None, branch=True, l2=1e-3):
    data = B.load_all()
    glia0 = data["meta_train"]["Region"].isna().to_numpy()
    fit_seeds = fit_seed if isinstance(fit_seed, (list, tuple)) else [fit_seed]
    parts_in = [load_partition(s, names) for s in fit_seeds]
    used = parts_in[0][1]
    for p in parts_in:
        assert p[1] == used, "expert sets differ across fit partitions"
    logs = np.concatenate([p[0] for p in parts_in], axis=1)
    allow = np.concatenate([p[2] for p in parts_in], axis=0)
    y = np.concatenate([p[3] for p in parts_in])
    classes = parts_in[0][4]
    glia = np.tile(glia0, len(fit_seeds))
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    lp = np.log(prior)
    print(f"experts ({len(used)}): {used}\nfit partitions {fit_seeds}, "
          f"{logs.shape[1]} rows")

    w_all, a_all = fit(logs, y, classes, lp, allow, l2=l2)
    print("global  exponents: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(used, w_all))
          + f"  prior_a={a_all:.3f}")
    parts = {}
    if branch:
        for nm, rows in (("glia", np.flatnonzero(glia)), ("neuron", np.flatnonzero(~glia))):
            parts[nm] = fit(logs, y, classes, lp, allow, rows=rows, l2=l2)
            print(f"{nm:7s} exponents: "
                  + "  ".join(f"{n}={v:.3f}" for n, v in zip(used, parts[nm][0]))
                  + f"  prior_a={parts[nm][1]:.3f}")

    def score(seed):
        lg, us, al, yy, cl = load_partition(seed, used)
        assert us == used
        pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(EPS).to_numpy()
        l = np.log(pr)
        base = cl[np.where(al, B.prior_correct(np.exp(lg[us.index("et")]), yy, cl),
                           -1).argmax(1)]
        z = apply(lg, w_all, a_all, l, al)
        pred_g = cl[z.argmax(1)]
        out = {"partition": seed, "et_a45": float(np.mean(base == yy)),
               "logpool": float(np.mean(pred_g == yy))}
        if branch:
            zb = np.zeros_like(z)
            zb[glia0] = apply(lg[:, glia0], *parts["glia"], l, al[glia0])
            zb[~glia0] = apply(lg[:, ~glia0], *parts["neuron"], l, al[~glia0])
            pb = cl[zb.argmax(1)]
            out["logpool_branch"] = float(np.mean(pb == yy))
            out["p_branch_vs_et"] = M.paired_mcnemar(pb == yy, base == yy)[0]
        out["p_pool_vs_et"] = M.paired_mcnemar(pred_g == yy, base == yy)[0]
        return out

    rows = [score(s) for s in list(fit_seeds) + list(eval_seeds)]
    tab = pd.DataFrame(rows)
    for c in ("logpool", "logpool_branch"):
        if c in tab:
            tab[c + "_gain"] = 100 * (tab[c] - tab.et_a45)
    pd.set_option("display.width", 200)
    print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    np.savez(B.OUT / "logpool_fit.npz", used=np.array(used), w_all=w_all, a_all=a_all,
             **({f"w_{k}": v[0] for k, v in parts.items()} if branch else {}),
             **({f"a_{k}": v[1] for k, v in parts.items()} if branch else {}))
    return tab


if __name__ == "__main__":
    fs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [18]
    ev = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else ()
    nm = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    main(fs, ev, nm)
