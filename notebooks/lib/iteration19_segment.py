"""Is `Segment` a spatial subdivision or a label-derived cluster id?

Segment is present for 100% of neurons and 0% of glia.  If it were a spatial subdivision
of the tissue section it could be imputed for glia from position, handing the branch that
carries 85% of the remaining error its first real metadata feature.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B

data = B.load_all()
meta, y = data["meta_train"], data["y"]
neu = ~meta["Region"].isna().to_numpy()
seg = meta["Segment"].astype(str).to_numpy()
sec = meta["Section_ID"].astype(str).to_numpy()
xy = meta[["center_x", "center_y"]].to_numpy()

print(f"cells with Segment: {(seg != 'nan').sum()} / {len(seg)}   "
      f"neurons {(neu).sum()}  | Segment levels {len(set(seg[seg!='nan']))}")
print(f"glia with Segment: {int(((seg != 'nan') & ~neu).sum())}")

m = seg != "nan"
tab = pd.crosstab(pd.Series(sec[m]), pd.Series(seg[m]))
print(f"\nsections x segments: {tab.shape}; segments per section "
      f"min/median/max {tab.gt(0).sum(1).min()}/{int(tab.gt(0).sum(1).median())}/{tab.gt(0).sum(1).max()}")
print(f"sections per segment min/median/max "
      f"{tab.gt(0).sum(0).min()}/{int(tab.gt(0).sum(0).median())}/{tab.gt(0).sum(0).max()}")

keep = m & (pd.Series(seg).groupby(seg).transform("size").to_numpy() >= 20)
X = np.hstack([xy[keep],
               pd.get_dummies(pd.Series(sec[keep])).to_numpy(float),
               meta["AP_position"].astype(str).factorize()[0][keep][:, None],
               meta["Region"].astype(str).factorize()[0][keep][:, None]])
et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)
pred = cross_val_predict(et, X, seg[keep], cv=StratifiedKFold(5, shuffle=True,
                                                             random_state=0))
base = pd.Series(seg[keep]).value_counts(normalize=True).iloc[0]
print(f"\npredicting Segment from position + section + AP + Region: "
      f"{np.mean(pred == seg[keep]):.4f} (majority {base:.4f})")

X2 = np.hstack([xy[keep], pd.get_dummies(pd.Series(sec[keep])).to_numpy(float)])
pred2 = cross_val_predict(et, X2, seg[keep], cv=StratifiedKFold(5, shuffle=True,
                                                               random_state=0))
print(f"predicting Segment from position + section alone:            "
      f"{np.mean(pred2 == seg[keep]):.4f}")

lab = y[keep]
print(f"predicting Segment from the cell-type label alone:           "
      f"{pd.DataFrame({'s': seg[keep], 'y': lab}).groupby('y').s.transform(lambda v: v.value_counts().index[0]).eq(seg[keep]).mean():.4f}")
print(f"\nmutual determination: cell types per segment "
      f"{pd.DataFrame({'s': seg[keep], 'y': lab}).groupby('s').y.nunique().to_dict()}")
