"""Floor the expert posteriors before log-pooling.

The pool fits its exponents by likelihood, and log-likelihood is unbounded below.  An
expert trained with entropy minimisation is near one-hot (mean max-posterior 0.95), so on
the cells where it is confidently wrong it assigns the truth about 1e-6 and contributes
roughly -14 nats.  The fit's cheapest response is to zero its exponent - which is exactly
what happens: all three semi-supervised experts get w = 0.000 on both branches despite
scoring 0.8106-0.8108 standalone, better than any single model this project has deployed.

Accuracy does not care about that penalty; the objective does.  Flooring every expert at
p <- (1 - e) p + e / C bounds the log-ratio an expert can contribute, so a confidently
wrong expert costs a bounded amount and the fit can use its ranking information.  The floor
is applied identically to every expert, so it introduces no per-expert tuning.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS

EPS = 1e-9


def floored(seed, names, eps):
    lgd, allow, y, classes = SS.part(seed)
    C = len(classes)
    p = np.stack([np.exp(lgd[n]) for n in names])
    if eps > 0:
        p = (1.0 - eps) * p + eps / C
    p /= np.maximum(p.sum(2, keepdims=True), 1e-12)
    return np.log(np.maximum(p, 1e-12)), allow, y, classes


def run(names, eps, l2=1e-3, partitions=(18, 41, 59, 83), folds=5, seed=2026):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    out = []
    for s in partitions:
        logs, allow, y, classes = floored(s, names, eps)
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
                w, a = LP.fit(logs, y, classes, lp, allow, rows=rr, l2=l2)
                pred[vv] = classes[LP.apply(logs[:, vv], w, a, lp, allow[vv]).argmax(1)]
            raw = np.exp(SS.part(s)[0]["et"])[val_idx]
            base[val_idx] = classes[np.where(
                allow[val_idx], B.prior_correct(raw, y, classes), -1).argmax(1)]
        out.append(100 * (float(np.mean(pred == y)) - float(np.mean(base == y))))
    return float(np.mean(out)), float(np.min(out)), out


def main():
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common
                   if n not in ("rank", "atlaslam_proto", "xgbaug4", "etaug4_0.08"))
    print(f"{len(names)} experts\n")
    rows = []
    for eps in (0.0, 0.002, 0.005, 0.02, 0.05, 0.10):
        m, mn, g = run(names, eps)
        rows.append({"floor": eps, "mean_gain": m, "worst": mn,
                     "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  floor {eps:<6g} mean {m:+.3f} worst {mn:+.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).sort_values("mean_gain", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
