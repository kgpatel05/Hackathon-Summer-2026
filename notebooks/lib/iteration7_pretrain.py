"""Iteration 7 - does neural pretraining beat our frozen linear transfer?

The atlas is already used, but only as a FROZEN linear probe: a logistic fitted on
136,612 atlas cells whose 60 class probabilities are appended as features. That is the
weakest possible form of transfer. This tests the two stronger forms:

  SSL  - self-supervised masked-gene autoencoder over all 146,612 cells (200 genes,
         challenge cells included: no labels used, so this is legitimate transduction),
         then the frozen 128-d embedding is added as a feature block.
  SUP  - supervised pretraining of an MLP on the 136,612 atlas cells (60-way), then
         FINE-TUNED end to end on the 5,000 challenge cells. Unlike the linear probe
         the representation itself adapts.

Both use only the 200 released genes. Neither reads a test label.
Baseline for comparison: the submitted model, 5-fold CV on the training cells.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import torch_device as TD

DEVICE = TD.get_device()

torch.manual_seed(0)
OUT = Path("outputs/iteration7")
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y)); CID = {c: i for i, c in enumerate(CLASSES)}
GENES = list(counts_train.columns)
glia = meta_train["Region"].isna().to_numpy()

cache = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
                allow_pickle=True)
XB = np.hstack([cache[k] for k in ["BASE_TR", "EXT_TR", "SPA_TR", "ATL_TR"]]).astype(np.float32)
CH = cache["EXPR_ALL"]                      # log-CPM, 10,000 x 200

with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    ag = [g.decode() for g in h["var/_index"][:]]
    cols = np.array([ag.index(g) for g in GENES])
    X = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                          shape=(len(ids), len(ag)))
    c = [x.decode() for x in h["obs/MERFISH cell type annotation/categories"][:]]
    lab = np.array([F._normalise_label(c[i]) if i >= 0 else "NA"
                    for i in h["obs/MERFISH cell type annotation/codes"][:]])
pos = {q: i for i, q in enumerate(ids)}
ch_mask = np.zeros(len(ids), bool)
for idx in [meta_train.index, meta_test.index]:
    ch_mask[[pos[q] for q in idx.astype(str)]] = True
keep = np.flatnonzero(~ch_mask & np.isin(lab, CLASSES))
AT = np.asarray(X[keep][:, cols].todense(), np.float32)
AT = F.log_cpm(AT); AY = np.array([CID[q] for q in lab[keep]])
print(f"atlas {AT.shape}  challenge {CH.shape}", flush=True)

CORPUS = np.vstack([AT, CH])
sc = StandardScaler().fit(CORPUS)
ATs, CHs = sc.transform(AT).astype(np.float32), sc.transform(CH).astype(np.float32)
Tc = torch.tensor(np.vstack([ATs, CHs])).to(DEVICE)

def encoder(latent=128):
    return nn.Sequential(nn.Linear(200, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, latent), nn.BatchNorm1d(latent), nn.ReLU())

# ---------------------------------------------------------------- SSL pretraining
t0 = time.time()
enc = encoder().to(DEVICE)
dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                    nn.Linear(256, 200)).to(DEVICE)
opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
for ep in range(15):
    perm = torch.randperm(len(Tc), device=DEVICE); tot = 0.0
    enc.train(); dec.train()
    for i in range(0, len(perm), 1024):
        b = Tc[perm[i:i + 1024]]
        if len(b) < 2: continue
        mask = (torch.rand_like(b) > 0.3).float()
        opt.zero_grad()
        loss = nn.functional.mse_loss(dec(enc(b * mask)), b)
        loss.backward(); opt.step(); tot += loss.item()
    if ep % 5 == 4:
        print(f"  SSL epoch {ep+1:2d} loss={tot/(len(perm)//1024+1):.4f}", flush=True)
enc.eval()
with torch.no_grad():
    EMB = enc(torch.tensor(CHs).to(DEVICE)).cpu().numpy()
print(f"[SSL] embedding {EMB.shape} in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- supervised pretrain
t0 = time.time()
enc2 = encoder().to(DEVICE); head = nn.Linear(128, len(CLASSES)).to(DEVICE)
opt = torch.optim.AdamW(list(enc2.parameters()) + list(head.parameters()), lr=1e-3)
At, Ay = torch.tensor(ATs).to(DEVICE), torch.tensor(AY).to(DEVICE)
for ep in range(12):
    perm = torch.randperm(len(At), device=DEVICE); enc2.train(); head.train()
    for i in range(0, len(perm), 1024):
        b = perm[i:i + 1024]
        if len(b) < 2: continue
        opt.zero_grad()
        nn.functional.cross_entropy(head(enc2(At[b])), Ay[b]).backward(); opt.step()
print(f"[SUP] atlas pretrain done in {time.time()-t0:.0f}s", flush=True)
import copy
SUP_STATE = copy.deepcopy(enc2.state_dict())

# ---------------------------------------------------------------- evaluate
def et_cv(Xf, tag):
    oof = np.zeros((len(y), len(CLASSES)), np.float32)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y):
        p = M.fit_extra_trees(Xf[tr], pd.Series(y[tr]), CLASSES, Xf[va], seeds=(0, 1))
        oof[va] = M.correct_prior(p, M.prior_vector(y[tr], CLASSES), 0.45)
    pred = np.array(CLASSES)[oof.argmax(1)]
    a, b = accuracy_score(y, pred), balanced_accuracy_score(y, pred)
    print(f"  {tag:42s} acc={a:.4f} bal={b:.4f} glia={accuracy_score(y[glia], pred[glia]):.4f}",
          flush=True)
    return {"config": tag, "accuracy": a, "balanced": b}

CHtr = torch.tensor(CHs[:5000]).to(DEVICE)
rows = []
print("\n=== 5-fold CV on the 5,000 challenge training cells ===", flush=True)
rows.append(et_cv(XB, "baseline (submitted model)"))
rows.append(et_cv(np.hstack([XB, EMB[:5000]]), "+ SSL masked-autoencoder embedding"))

with torch.no_grad():
    enc2.load_state_dict(SUP_STATE); enc2.eval()
    SUPEMB = enc2(CHtr).numpy()
rows.append(et_cv(np.hstack([XB, SUPEMB]), "+ atlas-pretrained embedding (frozen)"))

# fine-tuned end-to-end, per fold, probabilities appended
print("\n=== fine-tuned atlas MLP, per-fold (leak-free) ===", flush=True)
FT = np.zeros((5000, len(CLASSES)), np.float32)
yi = torch.tensor([CID[q] for q in y]).to(DEVICE)
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(CHs[:5000], y):
    e = encoder(); e.load_state_dict(SUP_STATE)
    hd = nn.Linear(128, len(CLASSES)).to(DEVICE)
    op = torch.optim.AdamW(list(e.parameters()) + list(hd.parameters()), lr=3e-4)
    idx = torch.tensor(tr).to(DEVICE)
    for ep in range(40):
        e.train(); hd.train(); perm = idx[torch.randperm(len(idx), device=DEVICE)]
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            if len(b) < 2: continue
            op.zero_grad(); nn.functional.cross_entropy(hd(e(CHtr[b])), yi[b]).backward(); op.step()
    e.eval(); hd.eval()
    with torch.no_grad():
        FT[va] = torch.softmax(hd(e(CHtr[torch.tensor(va).to(DEVICE)])), 1).cpu().numpy()
pred = np.array(CLASSES)[FT.argmax(1)]
print(f"  fine-tuned MLP standalone            acc={accuracy_score(y, pred):.4f} "
      f"bal={balanced_accuracy_score(y, pred):.4f}", flush=True)
rows.append(et_cv(np.hstack([XB, FT]), "+ fine-tuned atlas MLP probabilities"))

pd.DataFrame(rows).to_csv(OUT / "pretrain.csv", index=False)
print(f"\nwrote {OUT/'pretrain.csv'}", flush=True)
