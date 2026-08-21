"""How much accuracy is still reachable by combining the experts we already have?"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_submit as S

EPS = 1e-9

data = B.load_all()
classes, y = data["classes"], data["y"]
glia = data["meta_train"]["Region"].isna().to_numpy()
used, fits, _ = S.frozen_weights()

d = np.load(B.OUT / "experts_oof_seed18.npz", allow_pickle=True)
allow = d["allow"]
prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
lp = np.log(prior)
logs = np.stack([np.log(np.maximum(d[n], EPS)) for n in used])

z = np.zeros((len(y), len(classes)))
z[glia] = LP.apply(logs[:, glia], *fits["glia"], lp, allow[glia])
z[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], lp, allow[~glia])
pred = classes[z.argmax(1)]
print(f"pool OOF accuracy (partition 18) {np.mean(pred == y):.4f}")
print(f"  glia {np.mean(pred[glia]==y[glia]):.4f}   neuron {np.mean(pred[~glia]==y[~glia]):.4f}")

order = np.argsort(-z, axis=1)
rank = np.array([np.where(classes[order[i]] == y[i])[0][0] for i in range(len(y))])
for k in (1, 2, 3, 5):
    print(f"  top-{k} coverage {np.mean(rank < k):.4f}")

print("\nstandalone accuracy and oracle union:")
hits = {}
for n in used:
    p = np.where(allow, d[n], -1)
    hits[n] = classes[p.argmax(1)] == y
    print(f"  {n:12s} {hits[n].mean():.4f}")
any_hit = np.any(np.stack(list(hits.values())), axis=0)
print(f"\noracle union over {len(used)} experts: {any_hit.mean():.4f}")
print(f"  pool captures {np.mean(pred == y):.4f}; unrealised union headroom "
      f"{100*(any_hit.mean()-np.mean(pred==y)):.1f} pt")
n_correct = np.stack(list(hits.values())).sum(0)
tab = pd.crosstab(pd.cut(n_correct, [-1, 0, 3, 8, 15, 20, 30]), pred == y)
print("\nnumber of experts that are right, vs whether the pool is right:")
print(tab.to_string())
