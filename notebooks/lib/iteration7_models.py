"""Iteration 7 - model-class sweep on atlas glia.

The representation probe showed count transform is irrelevant (0.67-0.71 for every
one of five transforms) but that plain LOGISTIC beats our ExtraTrees stack by
+3.2 pt accuracy and +13.6 pt balanced accuracy on glia. Our whole pipeline is
ET-based, so this asks how much of the glia wall is the model class.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("outputs/iteration7")
d = np.load(OUT / "atlas_glia.npz", allow_pickle=True)
counts, yg, volg, tr, te = d["counts"], d["y"], d["vol"], d["tr"], d["te"]

t = counts.sum(1, keepdims=True); t[t == 0] = 1
Z = np.log1p(counts / t * 100.0)
tot, det = counts.sum(1), (counts > 0).sum(1)
safe = np.where(volg <= 0, np.nan, volg)
qc = np.nan_to_num(np.column_stack([np.log1p(tot), det, np.log1p(np.clip(volg, 0, None)),
                                    tot / safe, det / safe]), nan=-1.0)
X = np.hstack([Z, qc]).astype(np.float32)
sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X).astype(np.float32)
print(f"{X.shape[0]} glia, {X.shape[1]} features, {len(np.unique(yg))} classes", flush=True)

rows = []
def rec(name, p, t0):
    a, b = accuracy_score(yg[te], p), balanced_accuracy_score(yg[te], p)
    print(f"  {name:44s} acc={a:.4f} bal={b:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    rows.append({"model": name, "accuracy": a, "balanced": b})

print("\n=== model class, atlas glia held-out 30% ===", flush=True)
for C in [0.1, 1.0, 10.0]:
    t0 = time.time()
    m = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(Xs[tr], yg[tr])
    rec(f"logreg C={C}", m.predict(Xs[te]), t0)

t0 = time.time()
m = LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1,
                       class_weight="balanced").fit(Xs[tr], yg[tr])
rec("logreg C=1 balanced", m.predict(Xs[te]), t0)

t0 = time.time()
m = ExtraTreesClassifier(600, max_features=0.3, min_samples_leaf=2, n_jobs=-1,
                         random_state=0).fit(X[tr], yg[tr])
rec("ET-600 max_features=0.3", m.predict(X[te]), t0)

t0 = time.time()
m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1,
                                   random_state=0).fit(X[tr], yg[tr])
rec("HistGradientBoosting-250", m.predict(X[te]), t0)

t0 = time.time()
m = KNeighborsClassifier(50, weights="distance", n_jobs=-1).fit(Xs[tr], yg[tr])
rec("kNN-50 (expression)", m.predict(Xs[te]), t0)

# ---- MLP -------------------------------------------------------------------
import torch, torch.nn as nn
import torch_device as TD
DEVICE = TD.get_device()
torch.manual_seed(0)
classes = sorted(np.unique(yg)); cid = {c: i for i, c in enumerate(classes)}
yi = np.array([cid[c] for c in yg])
dev = "cpu"
Xt = torch.tensor(Xs).to(DEVICE); yt = torch.tensor(yi).to(DEVICE)

for hidden, wd, bal in [(256, 1e-4, False), (512, 1e-4, False), (512, 1e-4, True)]:
    t0 = time.time()
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(hidden, hidden // 2),
                        nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(hidden // 2, len(classes))).to(dev)
    w = None
    if bal:
        cnt = np.bincount(yi[tr], minlength=len(classes)).astype(np.float32)
        w = torch.tensor((cnt.mean() / np.maximum(cnt, 1)) ** 0.5).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=w)
    idx = torch.tensor(tr).to(DEVICE)
    for ep in range(30):
        net.train(); perm = idx[torch.randperm(len(idx), device=DEVICE)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad(); lossf(net(Xt[b]), yt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = np.array(classes)[net(Xt[torch.tensor(te).to(DEVICE)]).argmax(1).cpu().numpy()]
    rec(f"MLP h={hidden} balanced={bal}", p, t0)

pd.DataFrame(rows).to_csv(OUT / "model_probe.csv", index=False)
print("\nwrote", OUT / "model_probe.csv", flush=True)
