"""How much ridge does the exponent fit need?

With 25 experts the pool has 26 free parameters; fitted on two fold partitions they
start to absorb partition noise.  This selects the ridge strength by two-way held-out
accuracy - fit on {18,41} score {59,83} and fit on {59,83} score {18,41} - never on test.
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

EPS = 1e-9
SPLITS = [((18, 41), (59, 83)), ((59, 83), (18, 41))]


def pooled(seeds, names=None):
    parts = [LP.load_partition(s, names) for s in seeds]
    used = parts[0][1]
    return (np.concatenate([p[0] for p in parts], axis=1),
            used,
            np.concatenate([p[2] for p in parts], axis=0),
            np.concatenate([p[3] for p in parts]),
            parts[0][4])


def main(names=None, grid=(1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    rows = []
    for l2 in grid:
        gains, bgains = [], []
        for fitseeds, evalseeds in SPLITS:
            logs, used, allow, y, classes = pooled(fitseeds, names)
            prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
                EPS).to_numpy()
            lp = np.log(prior)
            w, a = LP.fit(logs, y, classes, lp, allow, l2=l2)
            gl = np.tile(glia, len(fitseeds))
            wg, ag = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(gl), l2=l2)
            wn, an = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(~gl), l2=l2)
            for s in evalseeds:
                lg, us, al, yy, cl = LP.load_partition(s, used)
                pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(
                    EPS).to_numpy()
                l = np.log(pr)
                base = cl[np.where(al, B.prior_correct(np.exp(lg[us.index("et")]), yy, cl),
                                   -1).argmax(1)]
                acc0 = float(np.mean(base == yy))
                z = LP.apply(lg, w, a, l, al)
                gains.append(100 * (float(np.mean(cl[z.argmax(1)] == yy)) - acc0))
                zb = np.zeros_like(z)
                zb[glia] = LP.apply(lg[:, glia], wg, ag, l, al[glia])
                zb[~glia] = LP.apply(lg[:, ~glia], wn, an, l, al[~glia])
                bgains.append(100 * (float(np.mean(cl[zb.argmax(1)] == yy)) - acc0))
        rows.append({"l2": l2, "global_mean_gain": np.mean(gains),
                     "global_min": np.min(gains),
                     "branch_mean_gain": np.mean(bgains), "branch_min": np.min(bgains)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = tab.loc[tab.global_mean_gain.idxmax()]
    print(f"\nbest global l2 = {best.l2:g} (mean held-out gain {best.global_mean_gain:.2f} pt)")
    bb = tab.loc[tab.branch_mean_gain.idxmax()]
    print(f"best branch l2 = {bb.l2:g} (mean held-out gain {bb.branch_mean_gain:.2f} pt)")


if __name__ == "__main__":
    main()
