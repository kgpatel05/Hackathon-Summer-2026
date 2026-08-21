"""Iteration 7 - WHICH genes separate the confused glial pairs, and are they in the 200?

The 500-gene panel reaches 0.875 on glia; the 200-gene subset reaches ~0.71. This
asks the direct question: for each confusable pair, rank all 500 genes by how well
they separate the pair (AUC on deep atlas data, so class-level estimates are clean),
and report how many of the top discriminators survived into the 200-gene release.

If the top discriminators are mostly WITHHELD, no external reference can recover
them - the information is absent from the released features. If they are PRESENT,
we are leaving signal on the table.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F

OUT = Path("outputs/iteration7")
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y_ch = meta_train[F.TARGET].astype(str)
GENES200 = list(counts_train.columns)
IN200 = set(GENES200)
GLIA = sorted(y_ch[meta_train["Region"].isna()].unique())

with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    genes500 = [g.decode() for g in h["var/_index"][:]]
    X = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                          shape=(len(ids), len(genes500)))
    c = [x.decode() for x in h["obs/MERFISH cell type annotation/categories"][:]]
    lab = np.array([F._normalise_label(c[i]) if i >= 0 else "NA"
                    for i in h["obs/MERFISH cell type annotation/codes"][:]])

pos = {q: i for i, q in enumerate(ids)}
ch = np.zeros(len(ids), bool)
for idx in [meta_train.index, meta_test.index]:
    ch[[pos[q] for q in idx.astype(str)]] = True
sel = np.flatnonzero(~ch & np.isin(lab, GLIA))
D = np.asarray(X[sel].todense(), np.float32); L = lab[sel]
t = D.sum(1, keepdims=True); t[t == 0] = 1
Z = np.log1p(D / t * 100.0)
print(f"{len(genes500)} panel genes, {len(IN200 & set(genes500))} of them released", flush=True)
print(f"atlas glia {len(Z)}\n", flush=True)

sizes = pd.Series(L).value_counts()
print("glial class sizes (atlas):"); print(sizes.to_string(), flush=True)

PAIRS = [("oligodendrocyte_1", "oligodendrocyte_progenitor_2"),
         ("oligodendrocyte_progenitor_2", "oligodendrocyte_2"),
         ("oligodendrocyte_1", "oligodendrocyte_2"),
         ("astrocyte_1", "astrocyte_2"),
         ("meninges_1", "meninges_2"),
         ("endothelial", "astrocyte_1")]

rows = []
print("\n=== per-pair separability: 500 genes vs the released 200 ===", flush=True)
for a, b in PAIRS:
    if a not in sizes or b not in sizes:
        print(f"  {a} vs {b}: MISSING from atlas"); continue
    m = np.isin(L, [a, b]); Za, ya = Z[m], (L[m] == a).astype(int)
    auc = np.array([roc_auc_score(ya, Za[:, j]) for j in range(Za.shape[1])])
    rank = np.argsort(-np.abs(auc - 0.5))
    top20 = [genes500[j] for j in rank[:20]]
    kept = [g for g in top20 if g in IN200]
    tr, te = train_test_split(np.arange(m.sum()), test_size=0.3, stratify=ya, random_state=0)
    cols200 = [genes500.index(g) for g in GENES200]
    def acc(cols):
        mo = LogisticRegression(C=1, max_iter=2000).fit(Za[tr][:, cols], ya[tr])
        return accuracy_score(ya[te], mo.predict(Za[te][:, cols]))
    a500, a200 = acc(list(range(Za.shape[1]))), acc(cols200)
    print(f"\n  {a} vs {b}   (n={m.sum()})", flush=True)
    print(f"    binary accuracy: 500 genes {a500:.4f} | released 200 {a200:.4f}"
          f"  (gap {a500-a200:+.4f})", flush=True)
    print(f"    top-20 discriminators kept in the 200: {len(kept)}/20", flush=True)
    print(f"    best withheld: {[g for g in top20 if g not in IN200][:6]}", flush=True)
    print(f"    best released: {kept[:6]}", flush=True)
    rows.append({"pair": f"{a}|{b}", "n": int(m.sum()), "acc_500": a500,
                 "acc_200": a200, "top20_kept": len(kept)})

pd.DataFrame(rows).to_csv(OUT / "pair_separability.csv", index=False)

# --- global: how much of the 500-gene glia signal is in the released 200? ----
print("\n=== all-glia 21-way, 500 vs 200 genes (logreg, deep data) ===", flush=True)
tr, te = train_test_split(np.arange(len(Z)), test_size=0.3, stratify=L, random_state=0)
cols200 = [genes500.index(g) for g in GENES200]
for name, cols in [("500 genes", list(range(Z.shape[1]))), ("released 200", cols200)]:
    t0 = time.time()
    mo = LogisticRegression(C=1, max_iter=2000, n_jobs=-1).fit(Z[tr][:, cols], L[tr])
    p = mo.predict(Z[te][:, cols])
    from sklearn.metrics import balanced_accuracy_score
    print(f"  {name:14s} acc={accuracy_score(L[te], p):.4f} "
          f"bal={balanced_accuracy_score(L[te], p):.4f} ({time.time()-t0:.0f}s)", flush=True)
