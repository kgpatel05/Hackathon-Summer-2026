"""A stronger parent-atlas transfer block.

The adopted stack distils the 136,612 non-challenge atlas cells through a C=0.1 logistic
regression (0.5992 standalone) and an ExtraTrees (Iteration 9).  Neither exploits 136k
cells the way a capacity-matched neural model can.  This module fits a metadata-
conditioned MLP on the atlas - 200 released genes only, every challenge cell removed -
and emits fine (60) and coarse (14) posteriors for the 10,000 challenge cells.
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
import iteration5_features as F

CACHE = B.OUT / "atlas_nn_block.npz"
CATS = [("obs_Datasets", "Datasets"), ("obs_Gender", "Gender"),
        ("obs_Region", "Region"), ("obs_Excitatory_vs_Inhibitory",
                                   "Excitatory_vs_Inhibitory"),
        ("obs_Axial_level", "AP_position"), ("obs_Mouse_ID", "Mouse_ID")]


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


def _atlas_ap(v):
    m = {"cervical": "1", "thoracic": "2", "lumbar": "3", "sacral": "4"}
    return np.array([m.get(x, x) for x in v])


def _region_to_challenge(v):
    m = {"dorsal horn": "1.0", "intermediate zone/ventral horn": "4.0",
         "intermediate zone": "3.0", "dorsal horn/intermediate zone": "2.0",
         "nan": "nan"}
    return np.array([m.get(x, "nan") for x in v])


def build(epochs=45, seeds=(0, 1, 2), hidden=(768, 384)):
    import torch, torch.nn as nn
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    h = np.load(B.OUT / "hierarchy_maps.npz", allow_pickle=True)
    r1 = pd.Series(h["r1"].astype(str), index=h["classes"].astype(str))
    groups = np.array(sorted(set(r1)))

    a_expr = F.log_cpm(atlas["counts"])
    a_qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                            (atlas["counts"] > 0).sum(1),
                            (atlas["counts"] == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_pos = _section_pos(np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T,
                         atlas["obs_Section_ID"].astype(str))
    a_cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": _region_to_challenge(atlas["obs_Region"].astype(str)),
        "Excitatory_vs_Inhibitory": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP_position": _atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str)})

    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    c_expr = F.log_cpm(counts_all)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_pos = _section_pos(meta_all[["center_x", "center_y"]].to_numpy(),
                         meta_all["Section_ID"].astype(str).to_numpy())
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy()
                          for k in ["Datasets", "Gender", "Region",
                                    "Excitatory_vs_Inhibitory", "AP_position",
                                    "Mouse_ID"]})
    c_cat["Region"] = c_cat["Region"].replace({"nan": "nan"})

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([a_expr, a_qc, a_pos, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([c_expr, c_qc, c_pos, enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    Xa, Xc = sc.transform(Xa).astype(np.float32), sc.transform(Xc).astype(np.float32)
    print(f"atlas design {Xa.shape}, challenge design {Xc.shape}", flush=True)

    ci = {c: i for i, c in enumerate(classes)}
    keep = np.array([l in ci for l in al])
    Xa, al = Xa[keep], al[keep]
    ya = np.array([ci[l] for l in al])
    gi = {g: i for i, g in enumerate(groups)}
    ga = np.array([gi[r1[l]] for l in al])

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev)
    yt = torch.tensor(ya, device=dev)
    gt = torch.tensor(ga, device=dev)
    Xq = torch.tensor(Xc, device=dev)
    fine = np.zeros((len(Xc), len(classes)), np.float32)
    coarse = np.zeros((len(Xc), len(groups)), np.float32)
    for s in seeds:
        t0 = time.time()
        torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(0.2)]
            d = w
        trunk = nn.Sequential(*layers).to(dev)
        head_f = nn.Linear(d, len(classes)).to(dev)
        head_c = nn.Linear(d, len(groups)).to(dev)
        params = list(trunk.parameters()) + list(head_f.parameters()) + list(head_c.parameters())
        opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        n, bs = len(Xt), 4096
        steps = epochs * ((n + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            trunk.train(); head_f.train(); head_c.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                z = trunk(Xt[idx])
                loss = lossf(head_f(z), yt[idx]) + 0.3 * lossf(head_c(z), gt[idx])
                loss.backward(); opt.step(); sched.step()
        trunk.eval(); head_f.eval(); head_c.eval()
        with torch.no_grad():
            zf, zc = [], []
            for i in range(0, len(Xq), 8192):
                z = trunk(Xq[i:i + 8192])
                zf.append(torch.softmax(head_f(z), 1).cpu().numpy())
                zc.append(torch.softmax(head_c(z), 1).cpu().numpy())
            fine += np.vstack(zf); coarse += np.vstack(zc)
        print(f"  seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    fine /= len(seeds); coarse /= len(seeds)
    np.savez_compressed(CACHE, fine=fine, coarse=coarse, classes=classes, groups=groups)
    y = data["y"]
    acc = np.mean(classes[fine[:len(y)].argmax(1)] == y)
    print(f"atlas-NN standalone accuracy on challenge training cells: {acc:.4f}")
    print(f"wrote {CACHE}")


def load():
    if not CACHE.exists():
        build()
    d = np.load(CACHE, allow_pickle=True)
    return d["fine"], d["coarse"]


if __name__ == "__main__":
    build()
