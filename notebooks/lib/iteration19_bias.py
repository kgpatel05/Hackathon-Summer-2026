"""Does a likelihood-fitted per-class offset on the pooled logits help?

Iteration 7 rejected vector scaling of a single ExtraTrees (-0.19 pt).  The pool is a
different object: 34 experts with fitted exponents, so residual class-wise bias is what is
left after the exponents have done their work.  The offset is fitted by likelihood, not
accuracy, and shrunk; selection is by two-way held-out accuracy.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS

EPS = 1e-9


def main(grid=(None, 3.0, 1.0, 0.3, 0.1, 0.03)):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common if n != "rank")
    rows = []
    for bl in grid:
        gains = []
        for fitseeds, evalseeds in SS.SPLITS:
            logs = np.concatenate([np.stack([SS.part(s)[0][n] for n in names])
                                   for s in fitseeds], axis=1)
            allow = np.concatenate([SS.part(s)[1] for s in fitseeds], axis=0)
            y = np.concatenate([SS.part(s)[2] for s in fitseeds])
            classes = SS.part(fitseeds[0])[3]
            prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
                EPS).to_numpy()
            lp = np.log(prior)
            gl = np.tile(glia, len(fitseeds))
            fits = {}
            for tag, rr in (("glia", np.flatnonzero(gl)), ("neuron", np.flatnonzero(~gl))):
                fits[tag] = LP.fit(logs, y, classes, lp, allow, rows=rr, bias_l2=bl)
            for s in evalseeds:
                lgd, al, yy, cl = SS.part(s)
                lg = np.stack([lgd[n] for n in names])
                pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(
                    EPS).to_numpy()
                l = np.log(pr)
                base = float(np.mean(cl[np.where(
                    al, B.prior_correct(np.exp(lgd["et"]), yy, cl), -1).argmax(1)] == yy))
                def _apply(sub, tag, mask):
                    f = fits[tag]
                    return LP.apply(sub, f[0], f[1], l, al[mask],
                                    f[2] if len(f) > 2 else None)
                z = np.zeros((len(yy), len(cl)))
                z[glia] = _apply(lg[:, glia], "glia", glia)
                z[~glia] = _apply(lg[:, ~glia], "neuron", ~glia)
                gains.append(100 * (float(np.mean(cl[z.argmax(1)] == yy)) - base))
        rows.append({"bias_l2": "none" if bl is None else bl,
                     "mean_gain": np.mean(gains), "worst": np.min(gains),
                     "gains": " ".join(f"{v:.2f}" for v in gains)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
