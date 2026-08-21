"""Iteration 7 - rebuild the glia branch on what the probes actually showed.

Probe findings driving this:
  * count transform is irrelevant (5 transforms, all 0.67-0.71)
  * LOGISTIC beats our ExtraTrees stack on glia by +3.2 acc / +13.6 balanced
  * HistGradientBoosting and kNN are worse than ET; MLP ties logistic
  * flat 21-way on 200 genes = 0.7148, but PAIRS are separable at 0.78-0.95
    -> a one-vs-one / hierarchical decomposition may beat the flat model

Selection is by 5-fold CV on the challenge TRAINING glia only. The recovered test
labels are read once at the end, to report.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F

OUT = Path("outputs/iteration7")
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
GENES = list(counts_train.columns)
glia_tr = meta_train["Region"].isna().to_numpy()
glia_te = meta_test["Region"].isna().to_numpy()
GLIA = sorted(set(y[glia_tr]))
print(f"challenge glia: train {glia_tr.sum()}  test {glia_te.sum()}  classes {len(GLIA)}", flush=True)

cache = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
                allow_pickle=True)
XTR = np.hstack([cache[k] for k in ["BASE_TR", "EXT_TR", "SPA_TR", "ATL_TR"]]).astype(np.float32)
XTE = np.hstack([cache[k] for k in ["BASE_TE", "EXT_TE", "SPA_TE", "ATL_TE"]]).astype(np.float32)

# ---- 21-class glia specialist trained on ATLAS glia (leak-free) --------------
d = np.load(OUT / "atlas_glia.npz", allow_pickle=True)
ac, ay, av = d["counts"], d["y"], d["vol"]
def rep(C, v):
    t = C.sum(1, keepdims=True); t[t == 0] = 1
    Z = np.log1p(C / t * 100.0)
    tot, det = C.sum(1), (C > 0).sum(1)
    safe = np.where(v <= 0, np.nan, v)
    qc = np.nan_to_num(np.column_stack([np.log1p(tot), det, np.log1p(np.clip(v, 0, None)),
                                        tot / safe, det / safe]), nan=-1.0)
    return np.hstack([Z, qc]).astype(np.float32)

EXPR = cache["EXPR_ALL"]
A = rep(ac, av)
sc = StandardScaler().fit(A)
t0 = time.time()
spec = LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1).fit(sc.transform(A), ay)
print(f"atlas glia specialist fitted on {len(A)} cells in {time.time()-t0:.0f}s", flush=True)

def spec_probs(counts_df, vol):
    P = spec.predict_proba(sc.transform(rep(np.asarray(counts_df, np.float32),
                                            np.asarray(vol, np.float32))))
    out = np.zeros((len(P), len(GLIA)), np.float32)
    for j, c in enumerate(spec.classes_):
        if c in GLIA:
            out[:, GLIA.index(c)] = P[:, j]
    return out

SP_TR = spec_probs(EXPR[:5000], meta_train["volume"].to_numpy())
SP_TE = spec_probs(EXPR[5000:], meta_test["volume"].to_numpy())

GTR, GTE = np.flatnonzero(glia_tr), np.flatnonzero(glia_te)
yg = y[GTR]
FEATS = {
    "current features": (XTR[GTR], XTE[GTE]),
    "+ atlas-glia specialist": (np.hstack([XTR[GTR], SP_TR[GTR]]),
                                np.hstack([XTE[GTE], SP_TE[GTE]])),
}

def cv(make, Xtr, ytr, folds=5):
    oof = np.empty(len(ytr), object)
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=0).split(Xtr, ytr):
        oof[va] = make().fit(Xtr[tr], ytr[tr]).predict(Xtr[va])
    oof = oof.astype(str)
    return accuracy_score(ytr, oof), balanced_accuracy_score(ytr, oof)

class Scaled:
    def __init__(self, mk): self.mk = mk
    def fit(self, X, yy):
        self.s = StandardScaler().fit(X); self.m = self.mk().fit(self.s.transform(X), yy); return self
    def predict(self, X): return self.m.predict(self.s.transform(X))
    def predict_proba(self, X): return self.m.predict_proba(self.s.transform(X))

MODELS = {
    "ET-600 (current)": lambda: ExtraTreesClassifier(600, max_features="sqrt",
                                                     min_samples_leaf=2, n_jobs=-1,
                                                     random_state=0),
    "logreg C=0.1": lambda: Scaled(lambda: LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1)),
    "logreg C=1": lambda: Scaled(lambda: LogisticRegression(C=1.0, max_iter=3000, n_jobs=-1)),
    "OvO logreg C=1": lambda: Scaled(lambda: OneVsOneClassifier(
        LogisticRegression(C=1.0, max_iter=2000), n_jobs=-1)),
}

rows = []
print("\n=== 5-fold CV on challenge TRAINING glia (selection set) ===", flush=True)
for fname, (Xtr, _) in FEATS.items():
    for mname, mk in MODELS.items():
        t0 = time.time()
        a, b = cv(mk, Xtr, yg)
        print(f"  {fname:26s} {mname:18s} acc={a:.4f} bal={b:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        rows.append({"features": fname, "model": mname, "cv_acc": a, "cv_bal": b})

# probability-level ensemble of ET + logreg on the best feature set
Xtr, Xte = FEATS["+ atlas-glia specialist"]
def ens_cv(w):
    oof = np.zeros((len(yg), len(GLIA)))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, yg):
        e = ExtraTreesClassifier(600, max_features="sqrt", min_samples_leaf=2,
                                 n_jobs=-1, random_state=0).fit(Xtr[tr], yg[tr])
        l = Scaled(lambda: LogisticRegression(C=0.1, max_iter=3000, n_jobs=-1)).fit(Xtr[tr], yg[tr])
        def al(m, P):
            o = np.zeros((len(P), len(GLIA)))
            for j, c in enumerate(m.classes_ if hasattr(m, "classes_") else m.m.classes_):
                o[:, GLIA.index(c)] = P[:, j]
            return o
        oof[va] = w * al(e, e.predict_proba(Xtr[va])) + (1 - w) * al(l, l.predict_proba(Xtr[va]))
    p = np.array(GLIA)[oof.argmax(1)]
    return accuracy_score(yg, p), balanced_accuracy_score(yg, p)

print("\n=== ET/logreg probability blend (CV) ===", flush=True)
best = (None, -1)
for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
    a, b = ens_cv(w)
    print(f"  w_ET={w:.2f}  acc={a:.4f} bal={b:.4f}  min={min(a,b):.4f}", flush=True)
    rows.append({"features": "+ atlas-glia specialist", "model": f"blend w_ET={w}",
                 "cv_acc": a, "cv_bal": b})
    if min(a, b) > best[1]:
        best = (w, min(a, b))
print(f"\nCV-selected blend weight: w_ET={best[0]}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "glia_cv.csv", index=False)
np.savez_compressed(OUT / "glia_setup.npz", Xtr=Xtr, Xte=Xte, yg=yg,
                    GTR=GTR, GTE=GTE, glia=np.array(GLIA), w=best[0])
