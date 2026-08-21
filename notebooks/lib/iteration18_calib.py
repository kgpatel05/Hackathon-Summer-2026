"""Is the incumbent posterior calibrated?  If so, decision-level methods cannot help.

Also: how far is the metadata prior from being fully exploited, and how much of the
error sits in cells whose posterior is genuinely near-tied?
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B

data = B.load_all()
classes, y = data["classes"], data["y"]
c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
p = B.prior_correct(c["oof_raw"], y, classes)
allow = c["oof_allow"]
p = np.where(allow, p, 0.0); p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
pred = classes[p.argmax(1)]
conf = p.max(1)
hit = (pred == y).astype(float)
print(f"OOF accuracy {hit.mean():.4f} | mean max-posterior {conf.mean():.4f} "
      f"(gap {conf.mean()-hit.mean():+.4f})")

bins = np.quantile(conf, np.linspace(0, 1, 11))
bins[-1] += 1e-6
b = np.digitize(conf, bins[1:-1])
tab = pd.DataFrame({"bin": b, "conf": conf, "hit": hit}).groupby("bin").agg(
    n=("hit", "size"), mean_conf=("conf", "mean"), accuracy=("hit", "mean"))
tab["excess"] = tab.accuracy - tab.mean_conf
print("\ncalibration by confidence decile:")
print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nBrier-style ceiling: if perfectly calibrated and we always take argmax, the "
      f"expected accuracy is mean(max p) = {conf.mean():.4f}")

glia = data["meta_train"]["Region"].isna().to_numpy()
for name, m in [("glia", glia), ("neuron", ~glia)]:
    print(f"  {name}: acc {hit[m].mean():.4f}  mean max-p {conf[m].mean():.4f}")

# --- metadata: how much is Region/EI/Segment worth, hard and soft?
meta = data["meta_train"]
key = (meta["Region"].astype(str) + "|" + meta["Excitatory_vs_Inhibitory"].astype(str)
       + "|" + meta["Segment"].astype(str)).to_numpy()
print(f"\nmetadata key levels: {len(set(key))}  "
      f"(Segment levels {meta['Segment'].nunique()})")
df = pd.DataFrame({"key": key, "y": y})
grp = df.groupby("key").y.agg(["size", "nunique",
                               lambda s: s.value_counts(normalize=True).iloc[0]])
grp.columns = ["n", "n_classes", "purity"]
print(grp.sort_values("n", ascending=False).head(12).to_string(
    float_format=lambda v: f"{v:.3f}"))
print(f"weighted mean purity of the metadata key: "
      f"{(grp.n*grp.purity).sum()/grp.n.sum():.4f}")
neur = ~meta["Region"].isna().to_numpy()
print(f"neuron-only weighted purity: "
      f"{(grp.n*grp.purity)[grp.index.isin(set(key[neur]))].sum()/grp.n[grp.index.isin(set(key[neur]))].sum():.4f}")

# how many errors are near-ties?
gap = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
for t in [0.02, 0.05, 0.10, 0.20]:
    m = gap < t
    print(f"  gap<{t:.2f}: {m.sum():5d} cells, accuracy {hit[m].mean():.4f}, "
          f"holds {int((1-hit[m]).sum()):4d} of {int((1-hit).sum())} errors")
