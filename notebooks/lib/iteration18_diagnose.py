"""Iteration 18 diagnostics: is the split random, and where does the marginal go wrong?

Reads recovered test truth ONLY to characterise the problem, never to fit.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
from evaluate import load_truth

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
classes = np.array(sorted(set(y)))
truth = load_truth().reindex(meta_test.index.astype(str)).to_numpy()

print(f"train {len(y)}  test {len(truth)}  classes {len(classes)}")

# ---------------------------------------------------------------- 1. split randomness
tr = pd.Series(y).value_counts().reindex(classes).fillna(0)
te = pd.Series(truth).value_counts().reindex(classes).fillna(0)
tab = pd.DataFrame({"train": tr, "test": te})
tab["diff"] = tab.test - tab.train
chi = stats.chi2_contingency(tab[["train", "test"]].T.to_numpy() + 0.0)
print(f"\nclass-count chi2 train vs test: chi2={chi.statistic:.1f} dof={chi.dof} p={chi.pvalue:.4f}")
print("largest absolute count differences:")
print(tab.reindex(tab["diff"].abs().sort_values(ascending=False).index).head(12).to_string())

for col in ["Section_ID", "Mouse_ID", "Region", "Gender", "AP_position", "Datasets"]:
    a = meta_train[col].astype(str).value_counts()
    b = meta_test[col].astype(str).value_counts()
    idx = sorted(set(a.index) | set(b.index))
    obs = np.vstack([a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)])
    keep = obs.sum(0) > 0
    r = stats.chi2_contingency(obs[:, keep])
    print(f"  {col:14s} levels={keep.sum():4d}  chi2={r.statistic:9.1f} p={r.pvalue:.4f}")

# per-section: are train and test counts ~ equal (paired halves)?
sa = meta_train["Section_ID"].astype(str).value_counts()
sb = meta_test["Section_ID"].astype(str).value_counts()
idx = sorted(set(sa.index) | set(sb.index))
sec = pd.DataFrame({"train": sa.reindex(idx).fillna(0), "test": sb.reindex(idx).fillna(0)})
sec["tot"] = sec.train + sec.test
print(f"\nsections: {len(sec)}  mean cells/section {sec.tot.mean():.1f}")
print(f"  corr(train_n, test_n) = {sec.train.corr(sec.test):.3f}")
print(f"  |train-test| mean {np.abs(sec.train-sec.test).mean():.2f} "
      f"vs binomial expectation {np.mean(np.sqrt(sec.tot/2)*0.7979):.2f}")

# ---------------------------------------------------------------- 2. neighbour distance
print("\n--- nearest same-section challenge neighbour ---")
from scipy.spatial import cKDTree
xy_tr = meta_train[["center_x", "center_y"]].to_numpy()
xy_te = meta_test[["center_x", "center_y"]].to_numpy()
sec_tr = meta_train["Section_ID"].astype(str).to_numpy()
sec_te = meta_test["Section_ID"].astype(str).to_numpy()
d_list, same_list = [], []
for s in np.unique(sec_te):
    a = np.where(sec_tr == s)[0]
    b = np.where(sec_te == s)[0]
    if len(a) == 0:
        continue
    tree = cKDTree(xy_tr[a])
    d, j = tree.query(xy_te[b], k=1)
    d_list.append(d)
    same_list.append(y[a[j]] == truth[b])
d = np.concatenate(d_list); same = np.concatenate(same_list)
print(f"  median distance {np.median(d):.1f}; nearest-train-label agreement {same.mean():.4f}")

np.savez(Path("outputs/iteration18/diagnose.npz"),
         classes=classes, train_counts=tr.to_numpy(), test_counts=te.to_numpy())
