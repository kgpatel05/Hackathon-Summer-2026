"""Iteration 7 probe - is the glia wall a PANEL limit or a REPRESENTATION limit?

Everything we built used log1p(CPM) + ExtraTrees. With a median of 21 transcripts
per cell that is a questionable choice. This probe removes sample size from the
question entirely: it trains on ~60k atlas GLIAL cells (200 genes only, all
challenge cells excluded) and asks how far different count representations and
model classes get on a held-out atlas split.

If every representation lands near 0.68, the 200-gene panel really is the wall.
If a better representation reaches ~0.78, our stack was the wall.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F

OUT = Path("outputs/iteration7"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y_ch = meta_train[F.TARGET].astype(str)
GENES = list(counts_train.columns)
GLIA_CLASSES = sorted(y_ch[meta_train["Region"].isna()].unique())
print(f"{len(GENES)} genes, {len(GLIA_CLASSES)} glial classes", flush=True)

with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    agenes = [g.decode() for g in h["var/_index"][:]]
    lut = {g: i for i, g in enumerate(agenes)}
    cols = np.array([lut[g] for g in GENES])
    X = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                          shape=(len(ids), len(agenes)))
    cats = [c.decode() for c in h["obs/MERFISH cell type annotation/categories"][:]]
    codes = h["obs/MERFISH cell type annotation/codes"][:]
    vol = h["obs/volume"][:]

lab = np.array([F._normalise_label(cats[c]) if c >= 0 else "NA" for c in codes])
pos = {c: i for i, c in enumerate(ids)}
challenge = np.zeros(len(ids), bool)
for idx in [meta_train.index, meta_test.index]:
    challenge[[pos[c] for c in idx.astype(str)]] = True

sel = np.flatnonzero(~challenge & np.isin(lab, GLIA_CLASSES))
counts = np.asarray(X[sel][:, cols].todense(), np.float32)
yg = lab[sel]; volg = np.asarray(vol[sel], np.float32)
keep = counts.sum(1) > 0
counts, yg, volg = counts[keep], yg[keep], volg[keep]
print(f"atlas glia: {len(counts)}  median transcripts={np.median(counts.sum(1)):.0f}", flush=True)

tr, te = train_test_split(np.arange(len(counts)), test_size=0.3,
                          stratify=yg, random_state=0)

# ---------------------------------------------------------------- representations
def log_cpm(C):
    t = C.sum(1, keepdims=True); t[t == 0] = 1
    return np.log1p(C / t * 100.0)

def sqrt_cpm(C):
    t = C.sum(1, keepdims=True); t[t == 0] = 1
    return np.sqrt(C / t * 100.0)

def pearson(C, theta=100.0):
    """Analytic Pearson residuals (Lause/Berens/Kharchenko 2021)."""
    n = C.sum(1, keepdims=True); p = C.sum(0, keepdims=True) / max(C.sum(), 1)
    mu = n * p
    z = (C - mu) / np.sqrt(mu + mu**2 / theta + 1e-8)
    lim = np.sqrt(len(C))
    return np.clip(z, -lim, lim)

def freq(C):
    t = C.sum(1, keepdims=True); t[t == 0] = 1
    return C / t

def binary(C):
    return (C > 0).astype(np.float32)

def with_qc(Z, C, v):
    tot, det = C.sum(1), (C > 0).sum(1)
    safe = np.where(v <= 0, np.nan, v)
    qc = np.column_stack([np.log1p(tot), det, np.log1p(np.clip(v, 0, None)),
                          tot / safe, det / safe])
    return np.hstack([Z, np.nan_to_num(qc, nan=-1.0)]).astype(np.float32)

REPS = {"log1p_cpm (current)": log_cpm, "sqrt_cpm": sqrt_cpm,
        "pearson_resid": pearson, "frequency": freq, "binary": binary}

def score(p, name, t):
    a = accuracy_score(yg[te], p); b = balanced_accuracy_score(yg[te], p)
    print(f"  {name:44s} acc={a:.4f} bal={b:.4f}  ({t:.0f}s)", flush=True)
    return {"config": name, "accuracy": a, "balanced": b}

rows = []
print("\n=== representation x model, atlas glia held-out 30% ===", flush=True)
for rname, fn in REPS.items():
    Z = with_qc(fn(counts), counts, volg)
    for mname, mk in [("ET-400", lambda: ExtraTreesClassifier(400, max_features="sqrt",
                                                             min_samples_leaf=2, n_jobs=-1,
                                                             random_state=0)),
                      ("logreg", lambda: LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1))]:
        t0 = time.time()
        if mname == "logreg":
            sc = StandardScaler().fit(Z[tr])
            m = mk().fit(sc.transform(Z[tr]), yg[tr])
            p = m.predict(sc.transform(Z[te]))
        else:
            m = mk().fit(Z[tr], yg[tr]); p = m.predict(Z[te])
        rows.append(score(p, f"{rname} + {mname}", time.time() - t0))

pd.DataFrame(rows).to_csv(OUT / "repr_probe.csv", index=False)
np.savez_compressed(OUT / "atlas_glia.npz", counts=counts, y=yg, vol=volg, tr=tr, te=te)
print("\nwrote", OUT / "repr_probe.csv", flush=True)
