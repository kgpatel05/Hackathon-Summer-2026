"""How much headroom lives in the top-2 decision, and can the atlas resolve those pairs?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A

data = B.load_all()
c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
classes, y = data["classes"], data["y"]
oof = B.prior_correct(c["oof_raw"], y, classes)
masked = np.where(c["oof_allow"], oof, -1.0)
order = np.argsort(-masked, axis=1)
top1 = classes[order[:, 0]]
top2 = classes[order[:, 1]]
p1 = np.take_along_axis(oof, order[:, :1], 1)[:, 0]
p2 = np.take_along_axis(oof, order[:, 1:2], 1)[:, 0]

in2 = (y == top1) | (y == top2)
print(f"OOF top-1 {np.mean(y==top1):.4f}   top-2 coverage {in2.mean():.4f}")
print(f"among top-2-covered cells, top1 correct: {np.mean(y[in2]==top1[in2]):.4f} "
      f"(n={in2.sum()})")
print(f"headroom if the pair were resolved perfectly: "
      f"+{100*(in2.mean()-np.mean(y==top1)):.2f} pt")

pair = np.array([tuple(sorted((a, b))) for a, b in zip(top1, top2)], dtype=object)
key = np.array(["|".join(p) for p in pair])
cnt = Counter(key[in2])
print(f"\ndistinct top-2 pairs among covered cells: {len(cnt)}")
rows = []
for k, n in cnt.most_common(25):
    m = (key == k) & in2
    rows.append({"pair": k, "n": int(m.sum()), "acc_now": float(np.mean(y[m] == top1[m])),
                 "margin": float(np.median(p1[m] - p2[m]))})
tab = pd.DataFrame(rows)
tab["errors"] = (tab.n * (1 - tab.acc_now)).round(0)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\ntop-25 pairs cover {tab.n.sum()} cells and {int(tab.errors.sum())} recoverable errors "
      f"of {int((in2 & (y!=top1)).sum())} total")

# atlas support for these pairs
atlas = A.load()
al = atlas["labels"]
support = Counter(al)
print("\natlas cell counts for the classes in the top pairs:")
seen = []
for k in tab.pair:
    for cls in k.split("|"):
        if cls not in seen:
            seen.append(cls)
print(pd.Series({s: support[s] for s in seen}).to_string())
