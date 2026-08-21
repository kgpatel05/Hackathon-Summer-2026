"""Iteration 7 - is the +0.28 pt SSL gain real, or is it noise?

The single-CV run gave baseline 0.7882 vs +SSL 0.7910. The established outer-fold noise
floor on this task is 0.4-0.6 pt, so that delta is inside it. This settles it properly:

  * 5x3 repeated stratified CV on identical folds
  * paired McNemar against the baseline
  * a NULL CONTROL - 128 columns of Gaussian noise instead of the embedding. If random
    features move the score as much as the embedding does, the embedding is doing nothing
    and we are only measuring what happens to ExtraTrees when the feature count grows.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
from scipy import sparse
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import torch_device as TD

DEVICE = TD.get_device()

torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = Path("outputs/iteration7")
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y)); GENES = list(counts_train.columns)
glia = meta_train["Region"].isna().to_numpy()

cache = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
                allow_pickle=True)
XB = np.hstack([cache[k] for k in ["BASE_TR", "EXT_TR", "SPA_TR", "ATL_TR"]]).astype(np.float32)
CH = cache["EXPR_ALL"]

with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    ag = [g.decode() for g in h["var/_index"][:]]
    cols = np.array([ag.index(g) for g in GENES])
    X = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                          shape=(len(ids), len(ag)))
pos = {q: i for i, q in enumerate(ids)}
ch_mask = np.zeros(len(ids), bool)
for idx in [meta_train.index, meta_test.index]:
    ch_mask[[pos[q] for q in idx.astype(str)]] = True
AT = F.log_cpm(np.asarray(X[np.flatnonzero(~ch_mask)][:, cols].todense(), np.float32))

sc = StandardScaler().fit(np.vstack([AT, CH]))
Tc = torch.tensor(np.vstack([sc.transform(AT),
                             sc.transform(CH)]).astype(np.float32)).to(DEVICE)

def build_ssl(seed):
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(200, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()).to(DEVICE)
    dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 200)).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for ep in range(15):
        perm = torch.randperm(len(Tc), device=DEVICE); enc.train(); dec.train()
        for i in range(0, len(perm), 1024):
            b = Tc[perm[i:i + 1024]]
            if len(b) < 2: continue
            opt.zero_grad()
            nn.functional.mse_loss(dec(enc(b * (torch.rand_like(b) > 0.3).float())), b).backward()
            opt.step()
    enc.eval()
    with torch.no_grad():
        x = torch.tensor(sc.transform(CH[:5000]).astype(np.float32)).to(DEVICE)
        return enc(x).cpu().numpy()

t0 = time.time()
EMB = np.mean([build_ssl(s) for s in (0, 1, 2)], axis=0)   # average 3 encoder seeds
print(f"SSL embeddings (3 seeds averaged) in {time.time()-t0:.0f}s", flush=True)
NOISE = rng.standard_normal((5000, 128)).astype(np.float32)

VARIANTS = {"baseline": XB,
            "+ SSL embedding": np.hstack([XB, EMB]),
            "+ 128 random columns (null control)": np.hstack([XB, NOISE])}

folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(XB, y))
correct, results = {}, []
for name, Xf in VARIANTS.items():
    t0 = time.time(); accs, bals, ok = [], [], np.zeros((3, 5000), bool)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(Xf[tr], pd.Series(y[tr]), CLASSES, Xf[va], seeds=(0, 1))
        p = M.correct_prior(p, M.prior_vector(y[tr], CLASSES), 0.45)
        pred = np.array(CLASSES)[p.argmax(1)]
        ok[f // 5, va] = pred == y[va]
        if f % 5 == 4:
            r = f // 5
            accs.append(ok[r].mean())
            bals.append(balanced_accuracy_score(y, np.where(ok[r], y, "")))
    correct[name] = ok
    a, s = float(np.mean(accs)), float(np.std(accs))
    print(f"  {name:38s} acc={a:.4f} +/-{s:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    results.append({"variant": name, "accuracy": a, "sd": s})

print("\n=== paired McNemar vs baseline (all 3 repeats pooled) ===", flush=True)
base = correct["baseline"].ravel()
for name in list(VARIANTS)[1:]:
    other = correct[name].ravel()
    p, _table = M.paired_mcnemar(other, base)
    b_only, o_only = int((base & ~other).sum()), int((other & ~base).sum())
    print(f"  {name:38s} gain={other.mean()-base.mean():+.4f} "
          f"discordant {o_only} vs {b_only}  p={p:.4g}", flush=True)
    for r in results:
        if r["variant"] == name:
            r["mcnemar_p"] = p; r["gain"] = other.mean() - base.mean()

pd.DataFrame(results).to_csv(OUT / "pretrain_significance.csv", index=False)
print(f"\nwrote {OUT/'pretrain_significance.csv'}", flush=True)
