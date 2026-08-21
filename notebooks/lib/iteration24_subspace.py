"""Bag the combination: many pools on random expert subspaces, averaged as posteriors.

Every result so far treats the pool as a single fitted object.  Averaging the *log-scores*
of many sub-pools is equivalent to one shrunk weight vector, because the score is linear in
the exponents - but averaging their normalised *posteriors* is not, and that is a genuinely
different aggregate: each sub-pool sees a different subset of experts, so each resolves the
confusable pairs differently, and the vote among them carries information a single fitted
rule cannot express.

Two variants are compared against the adopted single pool under cell-disjoint validation:
`z` averages the scores (the linear control) and `p` averages the posteriors.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS

EPS = 1e-9


def run(names, mode="p", n_bag=25, frac=0.6, row_frac=1.0, l2=1e-3,
        partitions=(18, 41, 59, 83), folds=5, seed=2026, rng_seed=7):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    out = []
    for s in partitions:
        lgd, allow, y, classes = SS.part(s)
        logs = np.stack([lgd[n] for n in names])
        M = len(names)
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        pred = np.empty(len(y), dtype=object)
        base = np.empty(len(y), dtype=object)
        rng = np.random.default_rng(rng_seed)
        for fit_idx, val_idx in StratifiedKFold(folds, shuffle=True,
                                                random_state=seed).split(logs[0], y):
            for mask in (glia, ~glia):
                rr = fit_idx[mask[fit_idx]]
                vv = val_idx[mask[val_idx]]
                if len(rr) < 80 or len(vv) == 0:
                    continue
                if mode == "single":
                    w, a = LP.fit(logs, y, classes, lp, allow, rows=rr, l2=l2)
                    z = LP.apply(logs[:, vv], w, a, lp, allow[vv])
                else:
                    agg = np.zeros((len(vv), len(classes)))
                    for b in range(n_bag):
                        sub = rng.choice(M, max(int(round(frac * M)), 4), replace=False)
                        rows_b = (rng.choice(rr, int(row_frac * len(rr)), replace=True)
                                  if row_frac != 1.0 else rr)
                        w, a = LP.fit(logs[sub], y, classes, lp, allow, rows=rows_b,
                                      l2=l2)
                        zb = LP.apply(logs[sub][:, vv], w, a, lp, allow[vv])
                        if mode == "z":
                            agg += zb
                        else:
                            agg += np.exp(zb - logsumexp(zb, axis=1, keepdims=True))
                    z = agg
                pred[vv] = classes[z.argmax(1)]
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
    for tag, kw in (("single pool", dict(mode="single")),
                    ("bag-p f=0.5", dict(mode="p", frac=0.5, n_bag=12)),
                    ("bag-p f=0.75", dict(mode="p", frac=0.75, n_bag=12)),
                    ("bag-z f=0.5", dict(mode="z", frac=0.5, n_bag=12))):
        m, mn, g = run(names, **kw)
        rows.append({"aggregate": tag, "mean_gain": m, "worst": mn,
                     "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  {tag:18s} mean {m:+.3f} worst {mn:+.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).sort_values("mean_gain", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
