"""What is the 200-gene ceiling for glia when full tissue context is available?

Within-atlas cross-validation on 136,612 non-challenge cells, 200 released genes only,
adding one context channel at a time.  This bounds how much honest headroom is left.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration5_features as F

CACHE = B.OUT / "atlas_context.npz"


def build_context(atlas, k=12):
    """Neighbour class histogram and mean expression inside each atlas section."""
    labels = atlas["labels"].astype(str)
    classes = np.array(sorted(set(labels)))
    ci = {c: i for i, c in enumerate(classes)}
    code = np.array([ci[l] for l in labels])
    xy = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec = atlas["obs_Section_ID"].astype(str)
    expr = F.log_cpm(atlas["counts"])
    comp = np.zeros((len(labels), len(classes)), np.float32)
    nexp = np.zeros_like(expr)
    ndist = np.zeros((len(labels), 1), np.float32)
    for s in np.unique(sec):
        rows = np.flatnonzero(sec == s)
        tree = cKDTree(xy[rows])
        kk = min(k + 1, len(rows))
        d, j = tree.query(xy[rows], k=kk)
        d, j = d[:, 1:], j[:, 1:]           # drop self
        nb = rows[j]
        for t in range(nb.shape[1]):
            comp[rows, code[nb[:, t]]] += 1.0
        comp[rows] /= max(nb.shape[1], 1)
        nexp[rows] = expr[nb].mean(1)
        ndist[rows, 0] = d.mean(1)
    return comp, nexp, ndist, classes


def main():
    atlas = A.load()
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        comp, nexp, ndist, aclasses = d["comp"], d["nexp"], d["ndist"], d["classes"]
    else:
        t0 = time.time()
        comp, nexp, ndist, aclasses = build_context(atlas)
        np.savez_compressed(CACHE, comp=comp, nexp=nexp, ndist=ndist, classes=aclasses)
        print(f"built atlas context ({time.time()-t0:.0f}s)")

    labels = atlas["labels"].astype(str)
    expr = F.log_cpm(atlas["counts"])
    qc = np.hstack([np.log1p(atlas["obs_volume"])[:, None],
                    np.log1p(atlas["counts"].sum(1))[:, None],
                    (atlas["counts"] > 0).sum(1)[:, None].astype(float)])
    axial = OneHotEncoder(sparse_output=False).fit_transform(
        atlas["obs_Axial_level"].astype(str)[:, None])
    xy = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec = atlas["obs_Section_ID"].astype(str)
    pos = np.zeros((len(sec), 4), np.float32)
    for s_ in np.unique(sec):
        r = np.flatnonzero(sec == s_)
        c = xy[r] - xy[r].mean(0)
        scale = np.abs(c).max(0) + 1e-9
        pos[r, :2] = c / scale
        pos[r, 2] = np.linalg.norm(c / scale, axis=1)
        pos[r, 3] = np.arctan2(c[:, 1], c[:, 0])
    lam = np.hstack([axial, pos])

    glia = atlas["obs_Region"].astype(str) == "nan"
    print(f"atlas glia {glia.sum()}  neurons {(~glia).sum()}")
    # PCA-free: use the 30 leading components of neighbour expression for parity
    from sklearn.decomposition import PCA
    nexp30 = PCA(n_components=30, random_state=0).fit_transform(nexp)

    variants = {
        "expr": [expr],
        "expr+qc": [expr, qc],
        "expr+qc+axial+pos": [expr, qc, lam],
        "expr+qc+axial+pos+nbrcomp": [expr, qc, lam, comp, ndist],
        "expr+qc+axial+pos+nbrcomp+nbrexpr": [expr, qc, lam, comp, ndist, nexp30],
    }
    rows = []
    sub = np.flatnonzero(glia)
    rng = np.random.default_rng(0)
    sub = rng.permutation(sub)[:60000]         # cap for runtime
    yg = labels[sub]
    keep = pd.Series(yg).groupby(yg).transform("size").to_numpy() >= 25
    sub, yg = sub[keep], yg[keep]
    print(f"glia CV subset {len(sub)} cells, {len(set(yg))} classes")
    for name, blocks in variants.items():
        X = np.hstack([b[sub] for b in blocks]).astype(np.float32)
        t0 = time.time()
        oof = np.empty(len(yg), dtype=object)
        for fit, val in StratifiedKFold(4, shuffle=True, random_state=0).split(X, yg):
            m = ExtraTreesClassifier(n_estimators=400, max_features="sqrt",
                                     min_samples_leaf=2, n_jobs=-1,
                                     random_state=0).fit(X[fit], yg[fit])
            oof[val] = m.predict(X[val])
        acc = float(np.mean(oof == yg))
        rows.append({"features": name, "dim": X.shape[1], "glia_acc": acc,
                     "sec": time.time() - t0})
        print(f"  {name:32s} dim={X.shape[1]:4d} glia CV acc {acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(B.OUT / "atlas_glia_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
