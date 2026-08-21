"""Atlas-pretrained, challenge-fine-tuned transfer expert.

Earlier attempts to *pool* the atlas with the challenge cells failed (0.6742 against
0.78 for challenge-only) because 136,612 external rows swamp 5,000 in-domain rows and
because the atlas has no `Segment`.  Sequential transfer avoids both: the network is
pretrained once on the atlas, then fine-tuned on the fold's challenge cells with the
atlas replayed at low weight, and `Segment` enters as an explicit "missing" category so
the network can learn to use it only where it exists.

Fold-scoped: the fine-tuning set for a fold contains only that fold's training cells.
"""
from __future__ import annotations
import copy, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration18_atlasnn3 as A3
import iteration5_features as F

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
HIDDEN = (1024, 512)
PRE_EPOCHS = 55
FT_EPOCHS = 14
CH_WEIGHT = 20.0


def design(use_laminae: bool = False):
    """use_laminae: give atlas rows the Segment implied by their published Laminae
    instead of a "missing" category (see iteration19_laminae)."""
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
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str),
        "Section_ID": sec_a,
        "Segment": np.full(len(sec_a), "missing")})
    if use_laminae:
        import iteration19_laminae as L19
        lam2seg = L19.laminae_to_segment(data["y"], data["meta_train"], al,
                                         atlas["obs_Laminae"].astype(str), verbose=False)
        a_cat["Segment"] = np.array([lam2seg.get(x, "missing")
                                     for x in atlas["obs_Laminae"].astype(str)])

    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy()
                          for k in a_cat.columns if k != "Segment"})
    c_cat["Segment"] = meta_all["Segment"].astype(str).to_numpy()
    c_cat = c_cat[a_cat.columns]

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([F.log_cpm(atlas["counts"]), a_qc, ANN._section_pos(xy_a, sec_a),
                    comp_a, dist_a, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, ANN._section_pos(xy_c, sec_c),
                    comp_c, dist_c, enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    keep = code_a >= 0
    return (sc.transform(Xa)[keep].astype(np.float32), code_a[keep],
            sc.transform(Xc).astype(np.float32), data)


def _net(dim, n_class, dropout=0.25):
    import torch.nn as nn
    layers, d = [], dim
    for w in HIDDEN:
        layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
        d = w
    return nn.Sequential(*layers, nn.Linear(d, n_class))


def _loop(net, X, yv, wv, epochs, lr, bs=4096):
    import torch, torch.nn as nn
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    n = len(X)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=max(epochs * ((n + bs - 1) // bs), 1))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.03, reduction="none")
    for _ in range(epochs):
        perm = torch.randperm(n, device=X.device)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 8:
                continue
            opt.zero_grad()
            loss = (lossf(net(X[idx]), yv[idx]) * wv[idx]).mean()
            loss.backward(); opt.step(); sched.step()
    return net


def run(seed=18, folds=5, use_laminae=False, tag="atlasft"):
    import torch
    cache = (Path("outputs/iteration19") / f"{tag}_oof_seed{seed}.npz" if use_laminae
             else B.OUT / f"atlasft_oof_seed{seed}.npz")
    cache.parent.mkdir(parents=True, exist_ok=True)
    Xa, ya, Xc, data = design(use_laminae)
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    n_tr = len(y)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xat = torch.tensor(Xa, device=dev)
    yat = torch.tensor(ya, device=dev)
    wat = torch.ones(len(ya), device=dev)
    Xct = torch.tensor(Xc, device=dev)
    yct = torch.tensor([ci[v] for v in y], device=dev)
    print(f"atlas {Xa.shape}, challenge {Xc.shape}", flush=True)

    pre_path = (Path("outputs/iteration19") / f"{tag}_pretrained.pt" if use_laminae
                else B.OUT / "atlasft_pretrained.pt")
    if pre_path.exists():
        states = torch.load(pre_path, map_location=dev, weights_only=False)
    else:
        states = []
        for s in SEEDS:
            t0 = time.time(); torch.manual_seed(s)
            net = _net(Xa.shape[1], len(classes)).to(dev)
            _loop(net, Xat, yat, wat, PRE_EPOCHS, 2e-3)
            states.append(copy.deepcopy(net.state_dict()))
            print(f"  pretrain seed {s} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(states, pre_path)

    if str(seed) == "mouse":
        from sklearn.model_selection import GroupKFold
        g = data["meta_train"]["Mouse_ID"].astype(str).to_numpy()
        splits = list(GroupKFold(n_splits=5).split(Xc[:n_tr], y, g))
    else:
        splits = list(StratifiedKFold(folds, shuffle=True,
                                      random_state=int(seed)).split(Xc[:n_tr], y))
    out = np.zeros((n_tr, len(classes)), np.float32)
    for k, (fit, val) in enumerate(splits):
        t0 = time.time()
        Xmix = torch.cat([Xat, Xct[:n_tr][fit]])
        ymix = torch.cat([yat, yct[fit]])
        wmix = torch.cat([wat, torch.full((len(fit),), CH_WEIGHT, device=dev)])
        acc = np.zeros((len(val), len(classes)), np.float32)
        for si, s in enumerate(SEEDS):
            torch.manual_seed(1000 + s)
            net = _net(Xa.shape[1], len(classes)).to(dev)
            net.load_state_dict(states[si])
            _loop(net, Xmix, ymix, wmix, FT_EPOCHS, 3e-4)
            net.eval()
            with torch.no_grad():
                acc += torch.softmax(net(Xct[:n_tr][val]), 1).cpu().numpy()
        out[val] = acc / len(SEEDS)
        print(f"  fold {k+1}/{folds} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache, probs=out, classes=classes)
    store = B.OUT / f"experts_oof_seed{seed}.npz"
    allow = (np.load(store)["allow"] if store.exists()
             else np.ones((n_tr, len(classes)), bool))
    print(f"{tag} partition {seed}: OOF "
          f"{np.mean(classes[np.where(allow, out, -1).argmax(1)] == y):.4f}")


def test_block(use_laminae=False, tag="atlasft"):
    import torch
    Xa, ya, Xc, data = design(use_laminae)
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    n_tr = len(y)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xat = torch.tensor(Xa, device=dev); yat = torch.tensor(ya, device=dev)
    wat = torch.ones(len(ya), device=dev)
    Xct = torch.tensor(Xc, device=dev)
    yct = torch.tensor([ci[v] for v in y], device=dev)
    pre = (Path("outputs/iteration19") / f"{tag}_pretrained.pt" if use_laminae
           else B.OUT / "atlasft_pretrained.pt")
    states = torch.load(pre, map_location=dev, weights_only=False)
    Xmix = torch.cat([Xat, Xct[:n_tr]])
    ymix = torch.cat([yat, yct])
    wmix = torch.cat([wat, torch.full((n_tr,), CH_WEIGHT, device=dev)])
    out = np.zeros((len(Xc) - n_tr, len(classes)), np.float32)
    for si, s in enumerate(SEEDS):
        torch.manual_seed(2000 + s)
        net = _net(Xa.shape[1], len(classes)).to(dev)
        net.load_state_dict(states[si])
        _loop(net, Xmix, ymix, wmix, FT_EPOCHS, 3e-4)
        net.eval()
        with torch.no_grad():
            out += torch.softmax(net(Xct[n_tr:]), 1).cpu().numpy()
    dest = (Path("outputs/iteration19") / f"{tag}.npz" if use_laminae
            else B.OUT / "atlasft_test.npz")
    np.savez_compressed(dest, probs=out / len(SEEDS), classes=classes)
    print(f"wrote {dest}")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    lam = "laminae" in args
    if lam:
        args.remove("laminae")
    tag = "atlasftlam" if lam else "atlasft"
    if args and args[0] == "test":
        test_block(lam, tag)
    else:
        for s in (args or ["18"]):
            run(seed=(s if s == "mouse" else int(s)), use_laminae=lam, tag=tag)
