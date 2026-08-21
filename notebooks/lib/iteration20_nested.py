"""Cell-disjoint validation of the pool parameters.

The Iteration-18/19 protocol fitted the pool exponents on fold partitions {18,41} and
scored them on {59,83}.  Those are different *fold assignments of the same 5,000 cells*,
so the labels of every scored cell were used to fit the exponents.  For 37 parameters the
optimism was small - the fixed pool predicted +1.51 and delivered +1.62 to +1.92 on the
真 held-out test cells - but it is not a safe protocol for the 109-parameter gated pool,
which gained +0.32 point under it and lost 0.06 on test.

This module splits by CELL: pool parameters are fitted on four fifths of the training
cells and scored on the fifth that contributed nothing to the fit.
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
import iteration20_gated as G

EPS = 1e-9


def run(names, gate, l2=1e-3, l2_int=1e-2, partitions=(18, 41, 59, 83), folds=5,
        seed=2026, branch=True):
    """gate: 'fixed' | 'prior' | 'depth' | 'both'"""
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    counts = data["counts_train"].to_numpy()
    use_v, use_u = {"fixed": (False, False), "prior": (True, False),
                    "depth": (False, True), "both": (True, True)}[gate]
    out = []
    for s in partitions:
        lgd, allow, y, classes = SS.part(s)
        logs = np.stack([lgd[n] for n in names])
        d_all = G.gates(data, classes, y)[0]
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        log_prior = np.log(prior)
        l_c = (log_prior - log_prior.mean()) / (log_prior.std() + 1e-9)
        pred = np.empty(len(y), dtype=object)
        base = np.empty(len(y), dtype=object)
        for fit_idx, val_idx in StratifiedKFold(folds, shuffle=True,
                                                random_state=seed).split(logs[0], y):
            branches = ((("glia", glia), ("neuron", ~glia)) if branch
                        else (("all", np.ones(len(y), bool)),))
            for tag, mask in branches:
                rr = np.array([i for i in fit_idx if mask[i]])
                vv = np.array([i for i in val_idx if mask[i]])
                if len(rr) < 50 or len(vv) == 0:
                    continue
                if gate == "fixed":
                    w, a = LP.fit(logs, y, classes, log_prior, allow, rows=rr, l2=l2)
                    z = LP.apply(logs[:, vv], w, a, log_prior, allow[vv])
                else:
                    w, v, u, a = G.fit(logs, y, classes, l_c, d_all, log_prior, allow,
                                       rows=rr, use_v=use_v, use_u=use_u, l2=l2,
                                       l2_int=l2_int)
                    z = G.apply(logs[:, vv], w, v, u, a, l_c, d_all[vv], log_prior,
                                allow[vv])
                pred[vv] = classes[z.argmax(1)]
            base[val_idx] = classes[np.where(
                allow[val_idx], B.prior_correct(np.exp(logs[names.index("et")])[val_idx],
                                                y, classes), -1).argmax(1)]
        acc = float(np.mean(pred == y))
        b = float(np.mean(base == y))
        out.append(100 * (acc - b))
    return float(np.mean(out)), float(np.min(out)), out


def main():
    common = set(SS.part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(SS.part(s_)[0])
    names = sorted(n for n in common if n != "rank")
    print(f"{len(names)} experts, cell-disjoint 5-fold validation of the pool parameters\n")
    rows = []
    base = [n for n in names if n != "atlaslam_proto"]
    NEW = ("nnaug", "nnaug2", "linaug", "atlasknn", "atlascons", "atlascons2",
           "atlascons_md")
    prev = [n for n in base if n not in NEW and n not in ("xgbaug4", "etaug4_0.08")]
    variants = {
        "A_prev_plus_all_new": prev + [n for n in NEW if n in base],
        "B_prev_plus_cons": prev + [n for n in ("atlascons", "atlascons2",
                                                "atlascons_md") if n in base],
        "C_prev_only": prev,
    }
    for tag, nm in variants.items():
        m, mn, g = run(nm, "fixed", l2=1e-3, branch=True)
        rows.append({"gate": f"{tag} (k={len(nm)})", "l2_int": 0, "mean_gain": m,
                     "worst": mn, "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  {tag:10s} k={len(nm)}  mean {m:+.3f} worst {mn:+.3f}", flush=True)
    for tag, kw in ():
        m, mn, g = run(names, "fixed", **kw)
        rows.append({"gate": tag, "l2_int": 0, "mean_gain": m, "worst": mn,
                     "gains": " ".join(f"{x:.2f}" for x in g)})
        print(f"  {tag:16s} mean {m:+.3f} worst {mn:+.3f}", flush=True)
    print("\n" + pd.DataFrame(rows).sort_values("mean_gain", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
