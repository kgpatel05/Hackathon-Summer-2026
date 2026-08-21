"""Logarithmic opinion pooling with likelihood-fitted exponents.

p(c | x) proportional to  prod_m p_m(c | x)^{w_m}  *  prior(c)^{-a}

Linear pooling assumes the experts are equally sharp; they are not (mean max-posterior
ranges from 0.577 for the random forest to 0.848 for XGBoost).  Log pooling lets the
fitted exponent absorb each expert's sharpness, so a well-informed but diffuse expert is
not drowned by a confident weak one.  The exponents are fitted by maximising the
out-of-fold multinomial log-likelihood - a convex problem with one parameter per expert -
never by searching on accuracy.
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

EPS = 1e-9


def load_partition(seed):
    d = np.load(B.OUT / f"experts_oof_seed{seed}.npz", allow_pickle=True)
    classes = d["classes"].astype(str)
    names = [k for k in d.files if k not in ("allow", "y", "classes")]
    logs = {n: np.log(np.maximum(d[n], EPS)) for n in sorted(names)}
    return logs, d["allow"], d["y"].astype(str), classes


def pooled_logits(logs, names, w, a, log_prior):
    z = np.zeros_like(next(iter(logs.values())))
    for wi, n in zip(w, names):
        z = z + wi * logs[n]
    return z - a * log_prior[None, :]


def fit_weights(logs, names, y, classes, log_prior, allow, l2=1e-3):
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    block = -1e9 * (~allow)

    def nll(theta):
        w, a = theta[:-1], theta[-1]
        z = pooled_logits(logs, names, w, a, log_prior) + block
        return float(-np.mean(z[np.arange(len(yi)), yi] - logsumexp(z, axis=1))
                     + l2 * np.sum(theta ** 2))

    x0 = np.append(np.full(len(names), 1.0 / len(names)), 0.3)
    res = minimize(nll, x0, method="L-BFGS-B",
                   bounds=[(0.0, 3.0)] * len(names) + [(0.0, 1.5)])
    return res.x[:-1], res.x[-1], res.fun


def evaluate(logs, names, w, a, y, classes, log_prior, allow):
    z = pooled_logits(logs, names, w, a, log_prior) + (-1e9 * (~allow))
    pred = classes[z.argmax(1)]
    return float(np.mean(pred == y)), pred


def main(fit_seed=18, eval_seeds=()):
    logs, allow, y, classes = load_partition(fit_seed)
    names = sorted(logs)
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)

    print("standalone OOF accuracy (mask applied, no prior correction):")
    for n in names:
        acc, _ = evaluate(logs, [n], [1.0], 0.0, y, classes, log_prior, allow)
        acc_a, _ = evaluate(logs, [n], [1.0], 0.45, y, classes, log_prior, allow)
        print(f"  {n:6s} raw {acc:.4f}   alpha=0.45 {acc_a:.4f}")

    w, a, f = fit_weights(logs, names, y, classes, log_prior, allow)
    acc, _ = evaluate(logs, names, w, a, y, classes, log_prior, allow)
    print(f"\nfitted exponents: " +
          "  ".join(f"{n}={v:.3f}" for n, v in zip(names, w)) + f"  prior_a={a:.3f}")
    print(f"log-pool OOF accuracy (fit partition {fit_seed}): {acc:.4f}")

    # equal-weight reference and arithmetic reference
    acc_eq, _ = evaluate(logs, names, np.full(len(names), 1.0 / len(names)), 0.45,
                         y, classes, log_prior, allow)
    arith = np.mean([np.exp(logs[n]) for n in names], axis=0)
    arith = B.prior_correct(arith, y, classes)
    acc_ar = float(np.mean(classes[np.where(allow, arith, -1).argmax(1)] == y))
    print(f"equal-exponent log-pool (a=0.45): {acc_eq:.4f} | arithmetic mean + a=0.45: {acc_ar:.4f}")

    for s in eval_seeds:
        logs2, allow2, y2, classes2 = load_partition(s)
        assert list(classes2) == list(classes)
        prior2 = pd.Series(y2).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp2 = np.log(prior2)
        acc2, _ = evaluate(logs2, names, w, a, y2, classes, lp2, allow2)
        base2, _ = evaluate(logs2, ["et"], [1.0], 0.45, y2, classes, lp2, allow2)
        print(f"  fresh partition {s}: incumbent(et,a=.45) {base2:.4f} -> "
              f"frozen log-pool {acc2:.4f}  ({100*(acc2-base2):+.2f} pt)")
    np.save(B.OUT / "logpool_weights.npy",
            np.array(list(w) + [a], dtype=np.float64))
    Path(B.OUT / "logpool_names.txt").write_text(",".join(names))
    return w, a


if __name__ == "__main__":
    fs = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    ev = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else ()
    main(fs, ev)
