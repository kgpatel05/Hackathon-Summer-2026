"""Outside-data expert bank: everything SNI_merged_0531.h5ad can give us.

SNI is a different experiment on different animals - outside data, not the source the
challenge was carved from - and it is 11x the size of the released training set. Until now
this project used exactly one thing from it: a C=0.1 logistic transfer on the `voting`
label. The file also carries four further independent annotations (RCTD, Seurat, SingleR,
Tangram) over the same 60-class taxonomy, and spatial coordinates.

Each annotation is a differently-biased view of the same truth, so transfers trained on
them make different mistakes - exactly what a log-linear pool can exploit.

Disclosure: SNI's own labels were produced by transferring from the published atlas. We do
not touch that atlas; we use a separate published dataset and the annotations its authors
released with it.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import no_source_data  # noqa: F401  - blocks any read of the source dataset
import iteration5_features as F
import iteration28_clean as C

OUT = Path("outputs/iteration28")
LABELS = ("voting", "rctd", "seurat", "singler", "tangram")


def _section_pos(xy, sec):
    out = np.zeros((len(sec), 4), np.float32)
    for s in np.unique(sec):
        r = np.flatnonzero(sec == s)
        c = xy[r] - np.median(xy[r], axis=0)
        scale = np.percentile(np.abs(c), 95, axis=0) + 1e-9
        out[r, :2] = c / scale
        out[r, 2] = np.linalg.norm(c / scale, axis=1)
        out[r, 3] = np.arctan2(c[:, 1], c[:, 0])
    return out


def design():
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    genes = list(counts_train.columns)
    classes = sorted(meta_train[F.TARGET].astype(str).unique())
    with h5py.File(F.EXTERNAL, "r") as h:
        ref = [g.decode() for g in h["var/_index"][:]]
        cols = np.array([ref.index(g) for g in genes])
        X = h["X"][:, :][:, cols].astype(np.float32)
        obs = {}
        for c in ["Axial level", "Datasets", "Section ID", "Side", "Condition"]:
            cc = [x.decode() for x in h[f"obs/{c}/categories"][:]]
            kk = h[f"obs/{c}/codes"][:]
            obs[c] = np.array([cc[k] if k >= 0 else "NA" for k in kk])
        for c in ["center_x", "center_y", "volume"]:
            obs[c] = h[f"obs/{c}"][:]
        labs = {}
        for c in LABELS:
            cc = [x.decode() for x in h[f"obs/{c}/categories"][:]]
            kk = h[f"obs/{c}/codes"][:]
            labs[c] = np.array([F._normalise_label(cc[k]) if k >= 0 else "NA" for k in kk])

    ap = {"cervical": "1.0", "thoracic": "2.0", "lumbar": "3.0", "sacral": "4.0"}
    r_qc = np.column_stack([np.log1p(X.sum(1)), (X > 0).sum(1), (X == 0).mean(1),
                            np.log1p(np.clip(obs["volume"], 0, None)),
                            X.sum(1) / np.maximum(obs["volume"], 1)])
    r_pos = _section_pos(np.vstack([obs["center_x"], obs["center_y"]]).T,
                         obs["Section ID"].astype(str))
    r_cat = pd.DataFrame({"AP_position": [ap.get(a, "1.0") for a in obs["Axial level"]]})

    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    counts_all = np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1), np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_pos = _section_pos(meta_all[["center_x", "center_y"]].to_numpy(),
                         meta_all["Section_ID"].astype(str).to_numpy())
    c_cat = pd.DataFrame({"AP_position": meta_all["AP_position"].astype(str).to_numpy()})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([r_cat, c_cat]))
    Xr = np.hstack([F.log_cpm(X), r_qc, r_pos, enc.transform(r_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, c_pos,
                    enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xr)
    return sc.transform(Xr).astype(np.float32), labs, sc.transform(Xc).astype(np.float32), \
        np.array(classes), meta_train


def build():
    Xr, labs, Xc, classes, meta_train = design()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    ci = {c: i for i, c in enumerate(classes)}
    n = len(y)
    store = OUT / "sni_experts.npz"
    d = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    print(f"SNI design {Xr.shape} -> challenge {Xc.shape}", flush=True)
    for lab in LABELS:
        keep = np.array([v in ci for v in labs[lab]])
        yr = np.array([ci[v] for v in labs[lab][keep]])
        for kind in ("nn", "et"):
            name = f"sni_{lab}_{kind}"
            if name in d:
                print(f"  {name}: cached"); continue
            t0 = time.time()
            if kind == "nn":
                yl = classes[yr]
                p = C._mlp(Xr[keep], yl, Xc, classes, 3, hidden=(768, 384),
                           epochs=40, dropout=0.2, wd=1e-2)
            else:
                p = np.zeros((len(Xc), len(classes)), np.float32)
                for s in (0, 1):
                    m = ExtraTreesClassifier(n_estimators=300, max_features=0.1,
                                             min_samples_leaf=2, n_jobs=-1,
                                             random_state=s).fit(Xr[keep], yr)
                    a = np.zeros_like(p); a[:, m.classes_.astype(int)] = m.predict_proba(Xc)
                    p += a
                p /= 2
            p = np.maximum(p, 1e-6); p /= p.sum(1, keepdims=True)
            d[name] = p.astype(np.float32)
            acc = np.mean(classes[p[:n].argmax(1)] == y)
            print(f"  {name:22s} standalone on challenge training cells {acc:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            np.savez_compressed(store, **d)
    np.savez_compressed(store, **d)
    print(f"wrote {store}")


if __name__ == "__main__":
    build()
