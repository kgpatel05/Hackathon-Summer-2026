"""Is the likelihood-optimal prior exponent also the accuracy-optimal one?

The pool exponents are fitted by likelihood; accuracy is a different objective.  This
sweeps only the single prior exponent, with every expert weight held at its fitted
value, and reports accuracy on partitions that were not used to fit anything.
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


def main(fit=(18, 41), evalp=(59, 83)):
    parts = [LP.load_partition(s) for s in fit]
    used = parts[0][1]
    logs = np.concatenate([p[0] for p in parts], axis=1)
    allow = np.concatenate([p[2] for p in parts], axis=0)
    y = np.concatenate([p[3] for p in parts])
    classes = parts[0][4]
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    w, a_ml = LP.fit(logs, y, classes, np.log(prior), allow)
    print(f"likelihood-optimal prior exponent from {fit}: {a_ml:.3f}")

    grid = np.arange(0.0, 1.45, 0.1)
    rows = []
    for tag, seeds in (("fit", fit), ("held-out", evalp)):
        for s in seeds:
            lg, us, al, yy, cl = LP.load_partition(s, used)
            pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(
                EPS).to_numpy()
            r = {"partition": s, "set": tag}
            for a in grid:
                z = LP.apply(lg, w, a, np.log(pr), al)
                r[f"{a:.1f}"] = float(np.mean(cl[z.argmax(1)] == yy))
            rows.append(r)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for tag in ("fit", "held-out"):
        m = tab[tab.set == tag][[f"{a:.1f}" for a in grid]].mean()
        print(f"\nmean accuracy on {tag} partitions:")
        print(m.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"  best a = {m.idxmax()}  ({m.max():.4f}); at a={a_ml:.2f} "
              f"-> {m[f'{round(a_ml,1):.1f}']:.4f}")


if __name__ == "__main__":
    main()
