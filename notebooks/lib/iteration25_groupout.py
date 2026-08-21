"""Does the model generalise to a NEW cohort, or only to new cells from the same tissue?

Cash prizes are decided on a validation dataset we cannot see, so the quantity that
matters is no longer accuracy on this test set - it is accuracy on cells from mice,
sections and imaging runs the model has not met.  Every validation protocol in this
project so far has held out random *cells*, which share mice and sections with the cells
used to fit.  This holds out whole groups instead:

    mouse    leave-one-mouse-out      (10 groups)  - a new animal
    dataset  leave-one-run-out        ( 6 groups)  - a new imaging batch
    section  leave-one-section-out    (108 groups) - new tissue from the same animals

Caveat, stated rather than hidden: the per-expert out-of-fold probabilities were produced
with random 5-fold cell splits, so an expert's prediction for a held-out mouse was made by
a model that saw other cells from that mouse.  This therefore measures how well the POOL
EXPONENTS transfer across groups, and is optimistic about the experts themselves.  If the
exponents prove group-sensitive, the experts need rebuilding group-disjointly too.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS

EPS = 1e-9
ADOPTED = None


def _adopted():
    global ADOPTED
    if ADOPTED is None:
        import iteration18_submit as S
        ADOPTED = sorted(S.ADOPTED)
    return ADOPTED


def run(names, grouping, partitions=(18, 41, 59, 83), l2=1e-3):
    data = B.load_all()
    meta = data["meta_train"]
    glia = meta["Region"].isna().to_numpy()
    if grouping == "cell":
        groups = None
    elif grouping == "mouse":
        groups = meta["Mouse_ID"].astype(str).to_numpy()
    elif grouping == "dataset":
        groups = meta["Datasets"].astype(str).to_numpy()
    else:
        groups = meta["Section_ID"].astype(str).to_numpy()
    out = []
    for s in partitions:
        lgd, allow, y, classes = SS.part(s)
        logs = np.stack([lgd[n] for n in names])
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        if groups is None:
            splits = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(
                logs[0], y))
        else:
            n_g = min(len(np.unique(groups)), 10)
            splits = list(GroupKFold(n_splits=n_g).split(logs[0], y, groups))
        pred = np.empty(len(y), dtype=object)
        base = np.empty(len(y), dtype=object)
        for fit_idx, val_idx in splits:
            for mask in (glia, ~glia):
                rr = fit_idx[mask[fit_idx]]
                vv = val_idx[mask[val_idx]]
                if len(rr) < 80 or len(vv) == 0:
                    continue
                w, a = LP.fit(logs, y, classes, lp, allow, rows=rr, l2=l2)
                pred[vv] = classes[LP.apply(logs[:, vv], w, a, lp, allow[vv]).argmax(1)]
            base[val_idx] = classes[np.where(
                allow[val_idx], B.prior_correct(np.exp(logs[names.index("et")])[val_idx],
                                                y, classes), -1).argmax(1)]
        done = pred != None                       # noqa: E711
        out.append((100 * float(np.mean(pred[done] == y[done])),
                    100 * float(np.mean(base[done] == y[done]))))
    acc = float(np.mean([a for a, _ in out]))
    bas = float(np.mean([b for _, b in out]))
    return acc, bas, acc - bas


def main():
    names = _adopted()
    print(f"{len(names)} adopted experts\n")
    rows = []
    for g in ("cell", "section", "mouse", "dataset"):
        acc, bas, gain = run(names, g)
        rows.append({"held out": g, "pool acc": acc, "ExtraTrees acc": bas,
                     "pool gain (pt)": gain})
        print(f"  hold out {g:8s} pool {acc:.2f}%  baseline {bas:.2f}%  "
              f"gain {gain:+.2f} pt", flush=True)
    t = pd.DataFrame(rows)
    print("\n" + t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    d = t.set_index("held out")["pool acc"]
    print(f"\ngeneralisation gap, random cells -> new mouse:   "
          f"{d['cell'] - d['mouse']:+.2f} pt")
    print(f"generalisation gap, random cells -> new imaging run: "
          f"{d['cell'] - d['dataset']:+.2f} pt")


if __name__ == "__main__":
    main()
