"""A three-way branch for the pool exponents instead of two.

Splitting the exponents by glia/neuron was worth +0.17 point over one global set, because
the metadata determines 80% of neurons and 26% of glia.  The same argument says the
oligodendrocyte lineage deserves its own set: it holds the largest confusions in the
problem and its cells are the ones for which the reference models are most informative.

The branch is assigned from a FIXED strong expert's coarse-cluster marginal, not from the
pool, so the assignment does not depend on the exponents being fitted.  Validated with the
cell-disjoint protocol.
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
OL_GROUPS = {"1", "5", "9"}          # {OL1, OPC2}, {OL2}, {OPC, OPG1}
ROUTER = "etaug4_0.25_3"


def families(seed, classes, glia):
    """0 = neurons, 1 = glia routed to the oligodendrocyte lineage, 2 = other glia."""
    h = np.load(B.OUT / "hierarchy_maps.npz", allow_pickle=True)
    g = pd.Series(h["r1"].astype(str), index=h["classes"].astype(str)).reindex(
        classes).to_numpy()
    lgd, allow, y, cl = SS.part(seed)
    p = np.exp(lgd[ROUTER])
    groups = np.array(sorted(set(g)))
    col = np.array([list(groups).index(x) for x in g])
    P = np.zeros((len(y), len(groups)))
    for j in range(len(cl)):
        P[:, col[j]] += p[:, j]
    top = groups[P.argmax(1)]
    fam = np.where(~glia, 0, np.where(np.isin(top, list(OL_GROUPS)), 1, 2))
    return fam


def run(names, n_branch=3, l2=1e-3, partitions=(18, 41, 59, 83), folds=5, seed=2026):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    out = []
    for s in partitions:
        lgd, allow, y, classes = SS.part(s)
        logs = np.stack([lgd[n] for n in names])
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        fam = (families(s, classes, glia) if n_branch == 3
               else np.where(~glia, 0, 1))
        pred = np.empty(len(y), dtype=object)
        base = np.empty(len(y), dtype=object)
        for fit_idx, val_idx in StratifiedKFold(folds, shuffle=True,
                                                random_state=seed).split(logs[0], y):
            for f in np.unique(fam):
                rr = fit_idx[fam[fit_idx] == f]
                vv = val_idx[fam[val_idx] == f]
                if len(rr) < 80 or len(vv) == 0:
                    vv2 = val_idx[fam[val_idx] == f]
                    if len(vv2):
                        rr = fit_idx
                    else:
                        continue
                w, a = LP.fit(logs, y, classes, lp, allow, rows=rr, l2=l2)
                pred[vv] = classes[LP.apply(logs[:, vv], w, a, lp, allow[vv]).argmax(1)]
            base[val_idx] = classes[np.where(
                allow[val_idx], B.prior_correct(np.exp(logs[names.index("et")])[val_idx],
                                                y, classes), -1).argmax(1)]
        out.append(100 * (float(np.mean(pred == y)) - float(np.mean(base == y))))
    return float(np.mean(out)), float(np.min(out)), out


def main():
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common if n not in ("rank", "atlaslam_proto"))
    print(f"{len(names)} experts")
    rows = []
    for nb in (2, 3):
        m, mn, g = run(names, n_branch=nb)
        rows.append({"branches": nb, "mean_gain": m, "worst": mn,
                     "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  {nb}-way branch: mean {m:+.3f} worst {mn:+.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).to_string(index=False,
                                              float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
