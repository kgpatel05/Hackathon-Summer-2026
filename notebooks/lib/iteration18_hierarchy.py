"""The published `1st round cluster` is a deterministic 14-way coarsening of the 60 types.

Measure: how good is the incumbent at that level, what would an oracle give, and how
good can a dedicated coarse model get?  Training cells only; no recovered test truth.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A

data = B.load_all()
classes, y = data["classes"], data["y"]
atlas = A.load()
al = atlas["labels"].astype(str)

# deterministic map cell_type -> r1 / r2 / laminae, read from the public atlas
maps = {}
for name, col in [("r1", "obs_1st_round_cluster"), ("r2", "obs_2nd_round_subcluster"),
                  ("lam", "obs_Laminae"), ("nt", "obs_Neurotransmitter"),
                  ("mk", "obs_Markers")]:
    v = atlas[col].astype(str)
    m = pd.DataFrame({"t": al, "v": v}).groupby("t").v.agg(lambda s: s.value_counts().index[0])
    purity = pd.DataFrame({"t": al, "v": v}).groupby("t").v.agg(
        lambda s: s.value_counts(normalize=True).iloc[0])
    maps[name] = (m.reindex(classes).to_numpy(), float(purity.reindex(classes).min()))
    print(f"{name}: {len(set(maps[name][0]))} groups, min per-class purity {maps[name][1]:.4f}")

c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
oof = B.prior_correct(c["oof_raw"], y, classes)
allow = c["oof_allow"]
masked = np.where(allow, oof, 0.0)
masked = masked / np.maximum(masked.sum(1, keepdims=True), 1e-12)
pred = classes[masked.argmax(1)]
print(f"\nincumbent OOF accuracy {np.mean(pred == y):.4f}")

for name in ["r1", "r2", "lam"]:
    g, _ = maps[name]
    groups = np.array(sorted(set(g)))
    gi = {k: i for i, k in enumerate(groups)}
    col = np.array([gi[x] for x in g])
    P = np.zeros((len(y), len(groups)))
    for j in range(len(classes)):
        P[:, col[j]] += masked[:, j]
    gpred = groups[P.argmax(1)]
    gtrue = np.array([g[list(classes).index(t)] for t in y])
    acc_g = np.mean(gpred == gtrue)
    # oracle: restrict argmax to the true group
    restricted = np.where(col[None, :] == np.array([gi[t] for t in gtrue])[:, None],
                          masked, -1.0)
    oracle = np.mean(classes[restricted.argmax(1)] == y)
    # accuracy achieved when the marginal group argmax is used as a hard constraint
    hard = np.where(col[None, :] == P.argmax(1)[:, None], masked, -1.0)
    print(f"  {name}: {len(groups):3d} groups | marginal-group acc {acc_g:.4f} | "
          f"oracle-group fine acc {oracle:.4f} | self-constrained {np.mean(classes[hard.argmax(1)]==y):.4f}")

# how do errors split into within/between coarse cluster?
g, _ = maps["r1"]
gt = pd.Series(g, index=classes)
gtrue = gt.reindex(y).to_numpy()
gpred = gt.reindex(pred).to_numpy()
err = pred != y
print(f"\nOOF errors {err.sum()}: cross-r1 {int((err & (gtrue != gpred)).sum())}, "
      f"within-r1 {int((err & (gtrue == gpred)).sum())}")
glia = data["meta_train"]["Region"].isna().to_numpy()
print(f"  glia errors {int((err&glia).sum())}: cross-r1 {int((err&glia&(gtrue!=gpred)).sum())}")
print(f"  neuron errors {int((err&~glia).sum())}: cross-r1 {int((err&~glia&(gtrue!=gpred)).sum())}")

np.savez(B.OUT / "hierarchy_maps.npz",
         classes=classes, **{k: v[0] for k, v in maps.items()})
print(f"\nwrote {B.OUT/'hierarchy_maps.npz'}")
