"""Compare a small number of pre-named pool compositions on the two-way held-out protocol.

A full greedy search over 25 members would select on the same fold partitions it is
scored on; a handful of named, a-priori compositions keeps the selection cheap and the
comparison honest.
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
_CACHE = {}


def part(seed):
    if seed not in _CACHE:
        d = np.load(B.OUT / f"experts_oof_seed{seed}.npz", allow_pickle=True)
        names = sorted(k for k in d.files if k not in ("allow", "y", "classes"))
        _CACHE[seed] = ({n: np.log(np.maximum(d[n], EPS)) for n in names},
                        d["allow"], d["y"].astype(str), d["classes"].astype(str))
    return _CACHE[seed]


def evaluate(names, branch=True, l2=1e-3):
    """branch: False = global exponents, True = per-branch, "both" = average the scores."""
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    gains = []
    for fitseeds, evalseeds in SPLITS:
        logs = np.concatenate([np.stack([part(s)[0][n] for n in names])
                               for s in fitseeds], axis=1)
        allow = np.concatenate([part(s)[1] for s in fitseeds], axis=0)
        y = np.concatenate([part(s)[2] for s in fitseeds])
        classes = part(fitseeds[0])[3]
        prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
            EPS).to_numpy()
        lp = np.log(prior)
        gl = np.tile(glia, len(fitseeds))
        if branch in (True, "both"):
            wg, ag = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(gl), l2=l2)
            wn, an = LP.fit(logs, y, classes, lp, allow, rows=np.flatnonzero(~gl), l2=l2)
        if branch in (False, "both"):
            w, a = LP.fit(logs, y, classes, lp, allow, l2=l2)
        for s in evalseeds:
            lgd, al, yy, cl = part(s)
            lg = np.stack([lgd[n] for n in names])
            pr = pd.Series(yy).value_counts(normalize=True).reindex(cl).fillna(
                EPS).to_numpy()
            l = np.log(pr)
            base = float(np.mean(cl[np.where(
                al, B.prior_correct(np.exp(lgd["et"]), yy, cl), -1).argmax(1)] == yy))
            zb = zg = None
            if branch in (True, "both"):
                zb = np.zeros((len(yy), len(cl)))
                zb[glia] = LP.apply(lg[:, glia], wg, ag, l, al[glia])
                zb[~glia] = LP.apply(lg[:, ~glia], wn, an, l, al[~glia])
            if branch in (False, "both"):
                zg = LP.apply(lg, w, a, l, al)
            z = zb if branch is True else zg if branch is False else 0.5 * (zb + zg)
            gains.append(100 * (float(np.mean(cl[z.argmax(1)] == yy)) - base))
    return float(np.mean(gains)), float(np.min(gains)), gains


def main():
    common = set(part(18)[0])
    for s_ in (41, 59, 83):
        common &= set(part(s_)[0])
    full = sorted(common)
    print(f"{len(full)} experts: {full}\n")
    core = ["etaug4_0.08", "etaug3", "xgb", "xgbaug", "logit", "meta", "meta2",
            "atlaslam_lin", "atlaslam_nn", "atlasftlam", "atlaslin", "atlasnn_md",
            "atlasnn5", "nb", "gliann", "etnog"]
    lean = ["etaug4_0.08", "xgb", "logit", "meta2", "atlaslam_lin", "atlaslam_nn",
            "atlasftlam", "atlasnn_md", "nb"]
    sets = {"C_minus_rank": [n for n in full if n != "rank"],
            "H_core16": [n for n in core if n in full],
            "I_lean9": [n for n in lean if n in full]}
    rows = []
    for tag, names in sets.items():
        for branch in (False, True, "both"):
            m, mn, g = evaluate(names, branch)
            rows.append({"set": tag, "k": len(names), "branch": branch,
                         "mean_gain": m, "worst": mn,
                         "gains": " ".join(f"{v:.2f}" for v in g)})
    tab = pd.DataFrame(rows).sort_values("mean_gain", ascending=False)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
