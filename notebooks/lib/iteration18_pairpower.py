"""Can an atlas-trained binary arbiter beat the incumbent inside its own top-2 pair?

Measured on the released TRAINING cells only (out-of-fold incumbent, atlas models that
never see a challenge cell).  Recovered test truth is not read.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration5_features as F

data = B.load_all()
c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
classes, y = data["classes"], data["y"]
oof = B.prior_correct(c["oof_raw"], y, classes)
masked = np.where(c["oof_allow"], oof, -1.0)
order = np.argsort(-masked, axis=1)
top1, top2 = classes[order[:, 0]], classes[order[:, 1]]
idx = {cl: i for i, cl in enumerate(classes)}

atlas = A.load()
al = atlas["labels"].astype(str)
a_expr = F.log_cpm(atlas["counts"])
a_vol = np.log1p(atlas["obs_volume"].astype(np.float64))[:, None]
a_depth = np.log1p(atlas["counts"].sum(1))[:, None]
a_ngene = (atlas["counts"] > 0).sum(1)[:, None].astype(np.float64)

ch_expr = F.log_cpm(data["counts_train"].to_numpy().astype(np.float32))
ch_vol = np.log1p(data["meta_train"]["volume"].to_numpy())[:, None]
ch_depth = np.log1p(data["counts_train"].to_numpy().sum(1))[:, None]
ch_ngene = (data["counts_train"].to_numpy() > 0).sum(1)[:, None].astype(np.float64)

A_FULL = np.hstack([a_expr, a_vol, a_depth, a_ngene])
C_FULL = np.hstack([ch_expr, ch_vol, ch_depth, ch_ngene])

pairs = Counter()
in2 = (y == top1) | (y == top2)
for a, b in zip(top1[in2], top2[in2]):
    pairs[tuple(sorted((a, b)))] += 1

rows = []
for (u, v), n in pairs.most_common(18):
    sel = ((top1 == u) & (top2 == v)) | ((top1 == v) & (top2 == u))
    truth_in = np.isin(y, [u, v])
    both = sel & truth_in
    if both.sum() < 25:
        continue
    now = float(np.mean(y[both] == top1[both]))

    m = np.isin(al, [u, v])
    Xa, ya = A_FULL[m], (al[m] == v).astype(int)
    sc = StandardScaler().fit(Xa)
    t0 = time.time()
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xa), ya)
    et = ExtraTreesClassifier(n_estimators=400, max_features="sqrt",
                              min_samples_leaf=3, n_jobs=-1,
                              random_state=0).fit(Xa, ya)
    pl = lr.predict_proba(sc.transform(C_FULL[both]))[:, 1]
    pe = et.predict_proba(C_FULL[both])[:, 1]
    yv = (y[both] == v).astype(int)
    # calibrate the decision threshold to the challenge prior for this pair
    prior_v = float(np.mean(np.isin(y, [u, v]) & (y == v)) /
                    max(np.mean(np.isin(y, [u, v])), 1e-9))
    accs = {}
    for nm, p in (("lr", pl), ("et", pe), ("avg", 0.5 * (pl + pe))):
        accs[nm] = float(np.mean((p > 0.5).astype(int) == yv))
        accs[nm + "_q"] = float(np.mean(
            (p > np.quantile(p, 1 - prior_v)).astype(int) == yv))
    rows.append({"pair": f"{u}|{v}", "n": int(both.sum()), "atlas_n": int(m.sum()),
                 "incumbent": now, **accs, "t": time.time() - t0})

tab = pd.DataFrame(rows)
tab["best_atlas"] = tab[["lr", "et", "avg", "lr_q", "et_q", "avg_q"]].max(1)
tab["gain_cells"] = ((tab.best_atlas - tab.incumbent) * tab.n).round(1)
pd.set_option("display.width", 220)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nTOTAL potential cells gained (oracle over 6 arbiter variants): "
      f"{tab.gain_cells.sum():.0f} of {len(y)} = {100*tab.gain_cells.sum()/len(y):.2f} pt")
