"""Neural and linear learners on the augmented feature stack, on the M-series GPU.

Every strong challenge-side expert in this project is a forest on the augmented stack.
A network and a linear softmax on the *same* columns are a genuinely different inductive
bias over the same information - piecewise-constant against smooth - and both train in
seconds on MPS, where the forests take minutes on eleven CPU threads.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_experts2 as E2

CONFIG = {
    "nnaug":  dict(hidden=(768, 384), epochs=120, dropout=0.35, lr=2e-3, wd=3e-2,
                   seeds=6, test_seeds=12),
    "nnaug2": dict(hidden=(1536,), epochs=90, dropout=0.45, lr=2e-3, wd=5e-2,
                   seeds=6, test_seeds=12),
    "linaug": dict(hidden=(), epochs=150, dropout=0.0, lr=3e-3, wd=1e-1,
                   seeds=4, test_seeds=8),
}


def _fit(Xtr, ytr, Xev, n_class, cfg, seeds, bs=512):
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)
    Xv = torch.tensor(Xev, device=dev)
    out = np.zeros((len(Xev), n_class), np.float32)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(Xt)
    for s in range(seeds):
        torch.manual_seed(s)
        layers, d = [], Xtr.shape[1]
        for w in cfg["hidden"]:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(),
                       nn.Dropout(cfg["dropout"])]
            d = w
        net = nn.Sequential(*layers, nn.Linear(d, n_class)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, cfg["lr"], total_steps=cfg["epochs"] * max((n + bs - 1) // bs, 1))
        for _ in range(cfg["epochs"]):
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
            out += torch.softmax(net(Xv), 1).cpu().numpy()
    return out / seeds


def run(name, seed):
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    yc = np.array([ci[v] for v in y])
    Xtr, Xte = E2.augmented4(data, seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
    cfg = CONFIG[name]
    out = np.zeros((len(y), len(classes)), np.float32)
    t0 = time.time()
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr_s, y):
        out[val] = _fit(Xtr_s[fit], yc[fit], Xtr_s[val], len(classes), cfg, cfg["seeds"])
    out /= np.maximum(out.sum(1, keepdims=True), 1e-12)
    store = B.OUT / f"experts_oof_seed{seed}.npz"
    d = dict(np.load(store, allow_pickle=True)); d[name] = out
    np.savez_compressed(store, **d)
    allow = d["allow"]
    print(f"  {name} partition {seed}: OOF "
          f"{np.mean(classes[np.where(allow, out, -1).argmax(1)] == y):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)


def run_test(name):
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    Xtr, Xte = E2.augmented4(data, 18)
    sc = StandardScaler().fit(Xtr)
    cfg = CONFIG[name]
    out = _fit(sc.transform(Xtr).astype(np.float32), np.array([ci[v] for v in y]),
               sc.transform(Xte).astype(np.float32), len(classes), cfg, cfg["test_seeds"])
    out /= np.maximum(out.sum(1, keepdims=True), 1e-12)
    store = B.OUT / "experts_test.npz"
    d = dict(np.load(store, allow_pickle=True)); d[name] = out.astype(np.float32)
    np.savez_compressed(store, **d)
    print(f"  {name} test probs written (mean max-p {out.max(1).mean():.4f})")


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CONFIG)
    for nm in names:
        for s in (18, 41, 59, 83):
            run(nm, s)
        run_test(nm)
