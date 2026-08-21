"""Leave-one-expert-out on the two-way held-out protocol.

Reports how the mean held-out gain moves when each pool member is removed, so that
members that only add fitting noise can be dropped.  Selection uses training-cell fold
partitions only.
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
import iteration18_l2sweep as SW

EPS = 1e-9


def mean_gain(names, branch=True, l2=1e-3):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    gains = []
    for fitseeds, evalseeds in SW.SPLITS:
        logs, used, allow, y, classes = SW.pooled(fitseeds, names)
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        gl = np.tile(glia, len(fitseeds))
        if branch:
            wg, ag = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(gl), l2=l2)
            wn, an = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(~gl), l2=l2)
        else:
            w, a = LP.fit(logs, y, classes, lp, allow, l2=l2)
        for s in evalseeds:
            lg, us, al, yy, cl = LP.load_partition(s, used)
            pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(
                EPS).to_numpy()
            l = np.log(pr)
            ref = np.load(B.OUT / f"experts_oof_seed{s}.npz", allow_pickle=True)
            base = float(np.mean(cl[np.where(
                al, B.prior_correct(ref["et"], yy, cl), -1).argmax(1)] == yy))
            if branch:
                z = np.zeros((len(yy), len(cl)))
                z[glia] = LP.apply(lg[:, glia], wg, ag, l, al[glia])
                z[~glia] = LP.apply(lg[:, ~glia], wn, an, l, al[~glia])
            else:
                z = LP.apply(lg, w, a, l, al)
            gains.append(100 * (float(np.mean(cl[z.argmax(1)] == yy)) - base))
    return float(np.mean(gains)), float(np.min(gains))


def main(branch=True):
    full = sorted(LP.load_partition(18)[1])
    base_mean, base_min = mean_gain(full, branch)
    print(f"full pool ({len(full)} experts): mean held-out gain {base_mean:.3f} pt "
          f"(worst {base_min:.3f})\n")
    rows = []
    for n in full:
        m, mn = mean_gain([x for x in full if x != n], branch)
        rows.append({"dropped": n, "mean_gain": m, "delta": m - base_mean, "worst": mn})
    tab = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    keep = list(full)
    while True:
        cand = [(mean_gain([x for x in keep if x != n], branch)[0], n) for n in keep]
        best, name = max(cand)
        cur = mean_gain(keep, branch)[0]
        if best <= cur + 1e-9 or len(keep) <= 6:
            break
        keep.remove(name)
        print(f"  drop {name:12s} -> {best:.3f} pt ({len(keep)} left)")
    m, mn = mean_gain(keep, branch)
    print(f"\ngreedy-pruned pool ({len(keep)}): {sorted(keep)}")
    print(f"  mean held-out gain {m:.3f} pt (worst {mn:.3f})")
    Path(B.OUT / "pruned_experts.txt").write_text(",".join(sorted(keep)))


if __name__ == "__main__":
    main(branch="global" not in sys.argv)
