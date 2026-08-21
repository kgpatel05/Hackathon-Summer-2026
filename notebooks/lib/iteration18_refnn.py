"""Neural transfer experts from each public reference cohort.

Variants
  atlasnn2    deeper metadata-conditioned parent-atlas network (different inductive bias)
  atlasnn_md  parent-atlas network WITHOUT the label-derived Region/EI columns
  sninn       network trained on the 55,331 SNI cells (different mice, `voting` labels)

Every network sees only the 200 released genes.  No challenge cell is in any training
set: the parent-atlas cache has all 10,000 removed, and SNI is a separate cohort.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration5_features as F


def _train_predict(Xa, ya, n_class, Xq, seeds, hidden, epochs, dropout=0.2, lr=2e-3):
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev); yt = torch.tensor(ya, device=dev)
    Xq_t = torch.tensor(Xq, device=dev)
    out = np.zeros((len(Xq), n_class), np.float32)
    for s in seeds:
        t0 = time.time(); torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        net = nn.Sequential(*layers, nn.Linear(d, n_class)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        n, bs = len(Xt), 4096
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * ((n + bs - 1) // bs))
        lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            net.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward()
                opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            out += np.vstack([torch.softmax(net(Xq_t[i:i + 8192]), 1).cpu().numpy()
                              for i in range(0, len(Xq_t), 8192)])
        print(f"    seed {s} ({time.time()-t0:.0f}s)", flush=True)
    return out / len(seeds)


def atlas_design(with_metadata=True):
    data = B.load_all()
    atlas = A.load()
    a_expr = F.log_cpm(atlas["counts"])
    a_qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                            (atlas["counts"] > 0).sum(1),
                            (atlas["counts"] == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_pos = ANN._section_pos(
        np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T,
        atlas["obs_Section_ID"].astype(str))
    cols = {"Datasets": atlas["obs_Datasets"].astype(str),
            "Gender": atlas["obs_Gender"].astype(str),
            "AP_position": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
            "Mouse_ID": atlas["obs_Mouse_ID"].astype(str)}
    if with_metadata:
        cols["Region"] = ANN._region_to_challenge(atlas["obs_Region"].astype(str))
        cols["Excitatory_vs_Inhibitory"] = atlas["obs_Excitatory_vs_Inhibitory"].astype(str)
    a_cat = pd.DataFrame(cols)

    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_pos = ANN._section_pos(meta_all[["center_x", "center_y"]].to_numpy(),
                             meta_all["Section_ID"].astype(str).to_numpy())
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy() for k in a_cat.columns})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([a_expr, a_qc, a_pos, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, c_pos,
                    enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    return sc.transform(Xa).astype(np.float32), sc.transform(Xc).astype(np.float32), \
        atlas["labels"].astype(str), data


def sni_design():
    import h5py
    from scipy import sparse
    data = B.load_all()
    genes = list(data["counts_train"].columns)
    with h5py.File(F.EXTERNAL, "r") as h:
        ref_genes = [g.decode() for g in h["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(ref_genes)}
        cols = np.array([lookup[g] for g in genes])
        X = h["X"][:, :][:, cols].astype(np.float32)
        cats = [c.decode() for c in h["obs/voting/categories"][:]]
        codes = h["obs/voting/codes"][:]
        obs = {}
        for c in ["Axial level", "Datasets", "Mouse ID", "Section ID", "Side",
                  "Condition", "batch"]:
            cc = [x.decode() for x in h[f"obs/{c}/categories"][:]]
            kk = h[f"obs/{c}/codes"][:]
            obs[c] = np.array([cc[k] if k >= 0 else "NA" for k in kk])
        for c in ["center_x", "center_y", "volume"]:
            obs[c] = h[f"obs/{c}"][:]
    labels = np.array([F._normalise_label(cats[c]) if c >= 0 else "NA" for c in codes])
    qc = np.column_stack([np.log1p(X.sum(1)), (X > 0).sum(1), (X == 0).mean(1),
                          np.log1p(np.clip(obs["volume"], 0, None)),
                          X.sum(1) / np.maximum(obs["volume"], 1)])
    pos = ANN._section_pos(np.vstack([obs["center_x"], obs["center_y"]]).T,
                           obs["Section ID"].astype(str))
    r_cat = pd.DataFrame({"AP_position": ANN._atlas_ap(obs["Axial level"].astype(str))})

    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_pos = ANN._section_pos(meta_all[["center_x", "center_y"]].to_numpy(),
                             meta_all["Section_ID"].astype(str).to_numpy())
    c_cat = pd.DataFrame({"AP_position": meta_all["AP_position"].astype(str).to_numpy()})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([r_cat, c_cat]))
    Xr = np.hstack([F.log_cpm(X), qc, pos, enc.transform(r_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, c_pos,
                    enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xr)
    return sc.transform(Xr).astype(np.float32), sc.transform(Xc).astype(np.float32), \
        labels, data


VARIANTS = {
    "atlasnn2": dict(source="atlas", md=True, hidden=(1024, 512, 256), epochs=60,
                     dropout=0.3, seeds=(10, 11, 12)),
    "atlasnn_md": dict(source="atlas", md=False, hidden=(768, 384), epochs=45,
                       dropout=0.2, seeds=(20, 21, 22)),
    "sninn": dict(source="sni", md=True, hidden=(768, 384), epochs=45, dropout=0.2,
                  seeds=(30, 31, 32)),
}


def build(name):
    cfg = VARIANTS[name]
    out_path = B.OUT / f"refnn_{name}.npz"
    if out_path.exists():
        print(f"{name}: cached"); return
    if cfg["source"] == "atlas":
        Xa, Xc, labels, data = atlas_design(cfg["md"])
    else:
        Xa, Xc, labels, data = sni_design()
    classes = data["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    keep = np.array([l in ci for l in labels])
    Xa, labels = Xa[keep], labels[keep]
    ya = np.array([ci[l] for l in labels])
    print(f"{name}: reference {Xa.shape}, {len(set(labels))} labels", flush=True)
    probs = _train_predict(Xa, ya, len(classes), Xc, cfg["seeds"], cfg["hidden"],
                           cfg["epochs"], cfg["dropout"])
    np.savez_compressed(out_path, probs=probs, classes=classes)
    y = data["y"]
    print(f"{name}: standalone accuracy on challenge training cells "
          f"{np.mean(classes[probs[:len(y)].argmax(1)] == y):.4f}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(VARIANTS)):
        build(nm)
