"""Parent-atlas neural transfer with matched section identity and tissue context.

Two channels the earlier atlas transfers never had:

* `Section_ID` one-hot.  The 108 tissue sections are shared between the atlas and the
  challenge - a challenge cell and its atlas donors come from the same physical section -
  so section identity carries both the local composition prior and the technical batch.
* the class histogram of the 12 nearest non-challenge atlas cells, computed identically
  for atlas cells (self excluded) and challenge cells.

Trained on the 136,612 non-challenge atlas cells over the 200 released genes only.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration18_refnn as RN
import iteration5_features as F

CACHE = B.OUT / "atlas_nn3_block.npz"
CACHE4 = B.OUT / "atlas_nn4_block.npz"
K = 12


def _multi_neighbour(xy, sec, xy_ref, sec_ref, code_ref, expr_ref, n_class, drop_self,
                     ks=(5, 12, 30, 80)):
    comp = np.zeros((len(sec), n_class * len(ks)), np.float32)
    nexp = np.zeros((len(sec), expr_ref.shape[1]), np.float32)
    dist = np.zeros((len(sec), len(ks)), np.float32)
    for s in np.unique(sec):
        qi = np.flatnonzero(sec == s)
        ri = np.flatnonzero(sec_ref == s)
        if len(ri) == 0:
            continue
        tree = cKDTree(xy_ref[ri])
        kmax = min(max(ks) + (1 if drop_self else 0), len(ri))
        d, j = tree.query(xy[qi], k=kmax)
        if drop_self:
            d, j = d[:, 1:], j[:, 1:]
        nb = code_ref[ri[j]]
        for t, k in enumerate(ks):
            kk = min(k, nb.shape[1])
            h = np.zeros((len(qi), n_class), np.float32)
            for col in range(kk):
                ok = nb[:, col] >= 0
                h[np.flatnonzero(ok), nb[ok, col]] += 1.0
            comp[qi, t * n_class:(t + 1) * n_class] = h / kk
            dist[qi, t] = d[:, kk - 1]
        nexp[qi] = expr_ref[ri[j[:, :min(30, j.shape[1])]]].mean(1)
    return comp, nexp, dist


def _donor_rows(section, sec_ref, group=None, group_ref=None):
    """Donors for a query section, falling back when that section is not in the atlas.

    The validation dataset may come from sections the public atlas does not contain, in
    which case an exact section match yields nothing and the neighbourhood features would
    silently be all zeros.  Fall back to the same mouse, then to every atlas cell.
    """
    ri = np.flatnonzero(sec_ref == section)
    if len(ri):
        return ri
    if group is not None and group_ref is not None:
        ri = np.flatnonzero(group_ref == group)
        if len(ri):
            return ri
    return np.arange(len(sec_ref))


def _neighbour_block(xy, sec, xy_ref, sec_ref, code_ref, n_class, drop_self,
                     group=None, group_ref=None):
    comp = np.zeros((len(sec), n_class), np.float32)
    dist = np.zeros((len(sec), 2), np.float32)
    for s in np.unique(sec):
        qi = np.flatnonzero(sec == s)
        g = group[qi[0]] if group is not None else None
        ri = _donor_rows(s, sec_ref, g, group_ref)
        if len(ri) == 0:
            continue
        tree = cKDTree(xy_ref[ri])
        kk = min(K + (1 if drop_self else 0), len(ri))
        d, j = tree.query(xy[qi], k=kk)
        if drop_self:
            d, j = d[:, 1:], j[:, 1:]
        nb = code_ref[ri[j]]
        h = np.zeros((len(qi), n_class), np.float32)
        for col in range(nb.shape[1]):
            ok = nb[:, col] >= 0
            h[np.flatnonzero(ok), nb[ok, col]] += 1.0
        comp[qi] = h / max(nb.shape[1], 1)
        dist[qi, 0] = d.mean(1)
        dist[qi, 1] = d[:, -1]
    return comp, dist


def build(seeds=tuple(range(10)), hidden=(1024, 512), epochs=55, dropout=0.25,
          multi=False):
    import torch
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code_a = np.array([ci.get(l, -1) for l in al])

    xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec_a = atlas["obs_Section_ID"].astype(str)
    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    xy_c = meta_all[["center_x", "center_y"]].to_numpy()
    sec_c = meta_all["Section_ID"].astype(str).to_numpy()

    t0 = time.time()
    a_expr = F.log_cpm(atlas["counts"])
    if multi:
        from sklearn.decomposition import PCA
        comp_a, nexp_a, dist_a = _multi_neighbour(xy_a, sec_a, xy_a, sec_a, code_a,
                                                  a_expr, len(classes), True)
        comp_c, nexp_c, dist_c = _multi_neighbour(xy_c, sec_c, xy_a, sec_a, code_a,
                                                  a_expr, len(classes), False)
        pca = PCA(n_components=30, random_state=0).fit(nexp_a[::3])
        dist_a = np.hstack([dist_a, pca.transform(nexp_a)])
        dist_c = np.hstack([dist_c, pca.transform(nexp_c)])
    else:
        comp_a, dist_a = _neighbour_block(xy_a, sec_a, xy_a, sec_a, code_a,
                                          len(classes), True)
        comp_c, dist_c = _neighbour_block(xy_c, sec_c, xy_a, sec_a, code_a,
                                          len(classes), False)
    print(f"neighbour blocks built ({time.time()-t0:.0f}s)", flush=True)
    a_qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                            (atlas["counts"] > 0).sum(1),
                            (atlas["counts"] == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_pos = ANN._section_pos(xy_a, sec_a)
    a_cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": ANN._region_to_challenge(atlas["obs_Region"].astype(str)),
        "Excitatory_vs_Inhibitory": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP_position": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str),
        "Section_ID": sec_a})

    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_pos = ANN._section_pos(xy_c, sec_c)
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy() for k in a_cat.columns})

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([a_expr, a_qc, a_pos, comp_a, dist_a, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, c_pos, comp_c, dist_c,
                    enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    Xa, Xc = sc.transform(Xa).astype(np.float32), sc.transform(Xc).astype(np.float32)
    keep = code_a >= 0
    Xa, ya = Xa[keep], code_a[keep]
    print(f"design {Xa.shape} -> challenge {Xc.shape}", flush=True)

    probs = RN._train_predict(Xa, ya, len(classes), Xc, seeds, hidden, epochs, dropout)
    np.savez_compressed(CACHE4 if multi else CACHE, probs=probs, classes=classes)
    y = data["y"]
    print(f"{'atlasnn4' if multi else 'atlasnn3'} standalone on challenge training cells: "
          f"{np.mean(classes[probs[:len(y)].argmax(1)] == y):.4f}")


if __name__ == "__main__":
    build(multi="multi" in sys.argv)
