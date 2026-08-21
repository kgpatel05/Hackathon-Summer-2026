"""How informative is the true tissue neighbourhood, measured properly?"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration5_features as F

data = B.load_all()
classes, y = data["classes"], data["y"]
atlas = A.load()
al = atlas["labels"].astype(str)
xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
sec_a = atlas["obs_Section_ID"].astype(str)

meta = data["meta_train"]
xy_c = meta[["center_x", "center_y"]].to_numpy()
sec_c = meta["Section_ID"].astype(str).to_numpy()
print("challenge sections:", len(set(sec_c)), "| atlas sections:", len(set(sec_a)))
print("overlap:", len(set(sec_c) & set(sec_a)))

KS = [1, 3, 5, 10, 25, 50, 100]
hit = {k: np.zeros(len(y), bool) for k in KS}
share = {k: np.zeros(len(y)) for k in KS}
d1 = np.zeros(len(y))
for s in np.unique(sec_c):
    qi = np.flatnonzero(sec_c == s)
    ri = np.flatnonzero(sec_a == s)
    if len(ri) == 0:
        continue
    tree = cKDTree(xy_a[ri])
    kmax = min(max(KS), len(ri))
    d, j = tree.query(xy_c[qi], k=kmax)
    nb = al[ri[j]]
    d1[qi] = d[:, 0]
    for k in KS:
        kk = min(k, kmax)
        sub = nb[:, :kk]
        maj = pd.DataFrame(sub).mode(axis=1)[0].to_numpy()
        hit[k][qi] = maj == y[qi]
        share[k][qi] = (sub == y[qi][:, None]).mean(1)

prior = pd.Series(y).value_counts(normalize=True)
print(f"\nnearest atlas-cell distance: median {np.median(d1):.2f}  p90 {np.percentile(d1,90):.2f}")
rows = []
for k in KS:
    rows.append({"k": k, "majority_acc": hit[k].mean(),
                 "own_class_share": share[k].mean(),
                 "expected_share_if_random": float((prior**2).sum())})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

glia = meta["Region"].isna().to_numpy()
print(f"\nk=10 majority accuracy: glia {hit[10][glia].mean():.4f} "
      f"neuron {hit[10][~glia].mean():.4f}")
print("\nper-class enrichment of own label among k=10 atlas neighbours:")
df = pd.DataFrame({"y": y, "share10": share[10], "share50": share[50]})
agg = df.groupby("y").agg(n=("share10", "size"), share10=("share10", "mean"),
                          share50=("share50", "mean"))
agg["prior"] = prior.reindex(agg.index)
agg["enrich10"] = agg.share10 / agg.prior
print(agg.sort_values("enrich10", ascending=False).head(14).to_string(
    float_format=lambda v: f"{v:.4f}"))
print(agg.sort_values("enrich10").head(8).to_string(float_format=lambda v: f"{v:.4f}"))
