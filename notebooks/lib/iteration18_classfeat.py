"""Per-(cell, class) evidence channels that a flat 60-way classifier cannot represent.

A tabular model must use class-agnostic columns.  Quantities such as "the multinomial
log-likelihood of this cell's counts under class c's atlas profile" or "how close is
this cell to its 15th nearest atlas cell OF CLASS c" are indexed by (cell, class) and
are only usable in a ranking formulation.  All atlas quantities come from the 136,612
non-challenge public atlas cells restricted to the 200 released genes.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration5_features as F

CACHE = B.OUT / "class_features.npz"
KNN_K = (1, 5, 15, 50)
COMP_K = (5, 10, 25, 60, 150)


def _torch_topk_by_class(q, r, labels, classes, ks, device):
    import torch
    out = np.zeros((len(q), len(classes), len(ks)), np.float32)
    qt = torch.tensor(q, device=device)
    qn = (qt * qt).sum(1, keepdim=True)
    for ci, cls in enumerate(classes):
        rows = np.flatnonzero(labels == cls)
        if len(rows) == 0:
            out[:, ci, :] = 1e3
            continue
        rt = torch.tensor(r[rows], device=device)
        rn = (rt * rt).sum(1)[None, :]
        d2 = torch.clamp(qn + rn - 2.0 * (qt @ rt.T), min=0.0)
        kk = [min(k, len(rows)) for k in ks]
        vals, _ = torch.topk(d2, max(kk), dim=1, largest=False)
        vals = torch.sqrt(vals)
        for j, k in enumerate(kk):
            out[:, ci, j] = vals[:, :k].mean(1).cpu().numpy()
        del rt, rn, d2, vals
    return out


def build():
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    a_counts = atlas["counts"]

    ch_counts = np.vstack([data["counts_train"].to_numpy(),
                           data["counts_test"].to_numpy()]).astype(np.float32)
    n = len(ch_counts)

    # ---- 1. multinomial / Dirichlet-multinomial log-likelihood under atlas profiles
    t0 = time.time()
    prof = np.zeros((len(classes), a_counts.shape[1]), np.float64)
    for i, c in enumerate(classes):
        m = al == c
        prof[i] = a_counts[m].sum(0) + 0.5 if m.any() else 0.5
    prof /= prof.sum(1, keepdims=True)
    loglik = ch_counts @ np.log(prof).T                      # multinomial kernel
    depth = ch_counts.sum(1, keepdims=True)
    loglik_n = loglik / np.maximum(depth, 1.0)               # per-transcript
    print(f"[loglik] {loglik.shape} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. centroid cosine + correlation in log-CPM space (SingleR-like)
    t0 = time.time()
    a_expr = F.log_cpm(a_counts)
    ch_expr = F.log_cpm(ch_counts)
    cent = np.zeros((len(classes), a_expr.shape[1]), np.float32)
    for i, c in enumerate(classes):
        m = al == c
        cent[i] = a_expr[m].mean(0) if m.any() else 0.0
    def _unit(z):
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    cos = _unit(ch_expr) @ _unit(cent).T
    zc = (cent - cent.mean(1, keepdims=True))
    zx = (ch_expr - ch_expr.mean(1, keepdims=True))
    corr = _unit(zx) @ _unit(zc).T
    print(f"[centroid] ({time.time()-t0:.0f}s)", flush=True)

    # ---- 3. per-class kNN distance in a 50-d atlas PCA space
    t0 = time.time()
    pca = PCA(n_components=50, random_state=0).fit(a_expr[::3])
    a_pc = pca.transform(a_expr).astype(np.float32)
    c_pc = pca.transform(ch_expr).astype(np.float32)
    sd = a_pc.std(0, keepdims=True) + 1e-9
    a_pc, c_pc = a_pc / sd, c_pc / sd
    device = "mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    knn = _torch_topk_by_class(c_pc, a_pc, al, classes, KNN_K, device)
    print(f"[knn] {knn.shape} on {device} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 4. multi-scale spatial composition of atlas neighbours, per class
    t0 = time.time()
    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    xy_c = meta_all[["center_x", "center_y"]].to_numpy()
    sec_c = meta_all["Section_ID"].astype(str).to_numpy()
    xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec_a = atlas["obs_Section_ID"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code_a = np.array([ci.get(x, -1) for x in al])
    comp = np.zeros((n, len(classes), len(COMP_K)), np.float32)
    rad = np.zeros((n, len(COMP_K)), np.float32)
    for s in np.unique(sec_c):
        qi = np.flatnonzero(sec_c == s)
        ri = np.flatnonzero(sec_a == s)
        if len(ri) == 0:
            continue
        tree = cKDTree(xy_a[ri])
        kmax = min(max(COMP_K), len(ri))
        d, j = tree.query(xy_c[qi], k=kmax)
        nb = code_a[ri[j]]
        for t, k in enumerate(COMP_K):
            kk = min(k, kmax)
            h = np.zeros((len(qi), len(classes)), np.float32)
            for col in range(kk):
                valid = nb[:, col] >= 0
                h[np.flatnonzero(valid), nb[valid, col]] += 1.0
            comp[qi, :, t] = h / kk
            rad[qi, t] = d[:, kk - 1]
    print(f"[comp] {comp.shape} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(
        CACHE, classes=classes, loglik=loglik.astype(np.float32),
        loglik_n=loglik_n.astype(np.float32), cos=cos.astype(np.float32),
        corr=corr.astype(np.float32), knn=knn, comp=comp, rad=rad,
        knn_k=np.array(KNN_K), comp_k=np.array(COMP_K))
    print(f"wrote {CACHE}")


def load():
    if not CACHE.exists():
        build()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build()
