"""Is the hard pairwise decision information-limited, or transfer-limited?

Within-atlas cross-validated binary accuracy (26k cells) versus the same model's
accuracy when applied to challenge cells.  Also maps the published clustering hierarchy.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration5_features as F

data = B.load_all()
classes, y = data["classes"], data["y"]
atlas = A.load()
al = atlas["labels"].astype(str)

# ---------------------------------------------------------------- hierarchy map
h = pd.DataFrame({"cell_type": al,
                  "r1": atlas["obs_1st_round_cluster"].astype(str),
                  "r2": atlas["obs_2nd_round_subcluster"].astype(str),
                  "lam": atlas["obs_Laminae"].astype(str),
                  "nt": atlas["obs_Neurotransmitter"].astype(str),
                  "mk": atlas["obs_Markers"].astype(str)})
print("=== published annotation hierarchy ===")
for col in ["r1", "r2", "lam", "nt", "mk"]:
    g = h.groupby(col).cell_type.nunique()
    inv = h.groupby("cell_type")[col].nunique()
    print(f"  {col}: {h[col].nunique():3d} levels | cell types per level "
          f"min/med/max {g.min()}/{int(g.median())}/{g.max()} | "
          f"levels per cell type max {inv.max()}  "
          f"(deterministic: {(inv==1).sum()}/{len(inv)})")
pair_key = h.r1.astype(str) + "//" + h.r2.astype(str)
print(f"  (r1,r2) combos: {pair_key.nunique()} -> cell types per combo max "
      f"{h.groupby(pair_key).cell_type.nunique().max()}")
print(f"  cell type -> (r1,r2) unique: "
      f"{(h.groupby('cell_type').apply(lambda d:(d.r1+'//'+d.r2).nunique())==1).sum()}/60")
print("\n  r1 level -> cell types:")
for k, g in h.groupby("r1"):
    types = sorted(g.cell_type.unique())
    print(f"    {k:28s} n={len(g):6d}  {len(types):2d}: {', '.join(types[:6])}"
          f"{' ...' if len(types) > 6 else ''}")

# ---------------------------------------------------------------- information limit
a_expr = F.log_cpm(atlas["counts"])
a_ctx = np.hstack([a_expr,
                   np.log1p(atlas["obs_volume"])[:, None],
                   np.log1p(atlas["counts"].sum(1))[:, None],
                   (atlas["counts"] > 0).sum(1)[:, None].astype(float)])
ch_expr = F.log_cpm(data["counts_train"].to_numpy().astype(np.float32))
c_ctx = np.hstack([ch_expr,
                   np.log1p(data["meta_train"]["volume"].to_numpy())[:, None],
                   np.log1p(data["counts_train"].to_numpy().sum(1))[:, None],
                   (data["counts_train"].to_numpy() > 0).sum(1)[:, None].astype(float)])

PAIRS = [("oligodendrocyte_1", "oligodendrocyte_progenitor_2"),
         ("oligodendrocyte_2", "oligodendrocyte_progenitor_2"),
         ("astrocyte_1", "astrocyte_2"),
         ("astrocyte_1", "endothelial"),
         ("meninges_1", "meninges_2"),
         ("oligodendrocyte_precursor_cell", "oligodendrocyte_progenitor_1")]
print("\n=== binary information limit: within-atlas CV vs challenge transfer ===")
rows = []
for u, v in PAIRS:
    m = np.isin(al, [u, v])
    Xa, ya = a_ctx[m], (al[m] == v).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oofp = np.zeros(len(ya))
    for tr, va in skf.split(Xa, ya):
        et = ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                  min_samples_leaf=3, n_jobs=-1, random_state=0)
        et.fit(Xa[tr], ya[tr])
        oofp[va] = et.predict_proba(Xa[va])[:, 1]
    within = accuracy_score(ya, oofp > 0.5)
    base = max(ya.mean(), 1 - ya.mean())
    cm = np.isin(y, [u, v])
    et = ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                              min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xa, ya)
    trans = accuracy_score((y[cm] == v).astype(int),
                           et.predict_proba(c_ctx[cm])[:, 1] > 0.5)
    rows.append({"pair": f"{u}|{v}", "atlas_n": int(m.sum()), "majority": base,
                 "within_atlas_cv": within, "challenge_n": int(cm.sum()),
                 "transfer": trans})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
