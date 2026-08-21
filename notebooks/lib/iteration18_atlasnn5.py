"""Multi-task parent-atlas network.

The published annotation is a nested product: `1st round cluster` (14) is a deterministic
coarsening of the 60 types, `2nd round subcluster` (23) narrows any coarse cluster to at
most two types, and `Neurotransmitter` / `Laminae` are further curated views.  Training
all five heads on one trunk regularises the 200-gene representation with structure the
flat 60-way loss never sees.
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
import iteration18_atlasnn3 as A3
import iteration5_features as F

CACHE = B.OUT / "atlas_nn5_block.npz"
AUX = [("obs_1st_round_cluster", 0.3), ("obs_2nd_round_subcluster", 0.3),
       ("obs_Neurotransmitter", 0.15), ("obs_Laminae", 0.15)]


def build(seeds=(0, 1, 2, 3, 4), hidden=(1024, 512), epochs=55, dropout=0.25):
    import torch, torch.nn as nn
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

    comp_a, dist_a = A3._neighbour_block(xy_a, sec_a, xy_a, sec_a, code_a, len(classes), True)
    comp_c, dist_c = A3._neighbour_block(xy_c, sec_c, xy_a, sec_a, code_a, len(classes), False)
    a_qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                            (atlas["counts"] > 0).sum(1),
                            (atlas["counts"] == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": ANN._region_to_challenge(atlas["obs_Region"].astype(str)),
        "Excitatory_vs_Inhibitory": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP_position": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str), "Section_ID": sec_a})
    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy() for k in a_cat.columns})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([F.log_cpm(atlas["counts"]), a_qc, ANN._section_pos(xy_a, sec_a),
                    comp_a, dist_a, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, ANN._section_pos(xy_c, sec_c),
                    comp_c, dist_c, enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    keep = code_a >= 0
    Xa, ya = sc.transform(Xa)[keep].astype(np.float32), code_a[keep]
    Xc = sc.transform(Xc).astype(np.float32)

    aux_targets, aux_w = [], []
    for col, w in AUX:
        v = atlas[col].astype(str)[keep]
        levels = {x: i for i, x in enumerate(sorted(set(v)))}
        aux_targets.append((np.array([levels[x] for x in v]), len(levels)))
        aux_w.append(w)
    print(f"design {Xa.shape}; aux heads {[t[1] for t in aux_targets]}", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev); yt = torch.tensor(ya, device=dev)
    at = [torch.tensor(t, device=dev) for t, _ in aux_targets]
    Xq = torch.tensor(Xc, device=dev)
    out = np.zeros((len(Xc), len(classes)), np.float32)
    for s in seeds:
        t0 = time.time(); torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        trunk = nn.Sequential(*layers).to(dev)
        head = nn.Linear(d, len(classes)).to(dev)
        aux_heads = [nn.Linear(d, k).to(dev) for _, k in aux_targets]
        params = list(trunk.parameters()) + list(head.parameters())
        for hh in aux_heads:
            params += list(hh.parameters())
        opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        n, bs = len(Xt), 4096
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, 2e-3, total_steps=epochs * ((n + bs - 1) // bs))
        lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            trunk.train(); head.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                z = trunk(Xt[idx])
                loss = lossf(head(z), yt[idx])
                for hh, tgt, w in zip(aux_heads, at, aux_w):
                    loss = loss + w * lossf(hh(z), tgt[idx])
                loss.backward(); opt.step(); sched.step()
        trunk.eval(); head.eval()
        with torch.no_grad():
            out += np.vstack([torch.softmax(head(trunk(Xq[i:i + 8192])), 1).cpu().numpy()
                              for i in range(0, len(Xq), 8192)])
        print(f"  seed {s} ({time.time()-t0:.0f}s)", flush=True)
    out /= len(seeds)
    np.savez_compressed(CACHE, probs=out, classes=classes)
    y = data["y"]
    print(f"atlasnn5 standalone on challenge training cells: "
          f"{np.mean(classes[out[:len(y)].argmax(1)] == y):.4f}")


if __name__ == "__main__":
    build()
