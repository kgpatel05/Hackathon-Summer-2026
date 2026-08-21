"""Iteration 8b - are the rare dorsal-horn classes separable, or just never predicted?

WHY THIS IS NOW THE TARGET. From the 19 August leaderboard we are 3rd at 0.7784 with
first place at 0.8138. The arithmetic:

    neurons  1632/1823 = 0.8952   191 errors
    glia     2260/3177 = 0.7114   917 errors

    perfect neurons + our current glia = 4083/5000 = 0.8166  >  0.8138

Glia carry 83% of the errors but they are information-limited: the released panel
contains none of Plp1, Mbp, Mog, Sox10, Pdgfra, Cspg4, Aqp4, Gfap, Cldn5 or Pecam1 - it
is a neuron-subtyping panel (35 protocadherins, semaphorins, neuropeptides, GPCRs). The
191 neuron errors sit where the panel is DENSE, and they are extraordinarily
concentrated: 121 cells in classes below 0.5 recall account for 106 of them, and four
classes sit at exactly zero recall.

Zero recall has two very different causes and they need different fixes:
  (a) the class is not separable on these features -> nothing to do;
  (b) the class is separable but so rare that the argmax never selects it -> a decision
      rule problem, worth up to 106 cells.

This measures which. One-vs-rest AUC is the right diagnostic because it is threshold-free:
it asks whether the model ranks the class's own cells above the others, independent of
whether it ever wins an argmax.

Restricted to Region == 1 (the 26 dorsal-horn classes), since Region is a deterministic
function of the label and the metadata mask already enforces it - so this is exactly the
subproblem the model actually faces for these cells.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y_all = meta_train[F.TARGET].astype(str).to_numpy()
c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X_all = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
                   c["ATL_TR"]]).astype(np.float32)

dh = (meta_train["Region"].astype(str) == "1.0").to_numpy() | \
     (meta_train["Region"] == 1).to_numpy()
X, y = X_all[dh], y_all[dh]
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
print(f"dorsal horn (Region==1): {len(y)} training cells, {len(CLASSES)} classes",
      flush=True)
print(f"class sizes: min={pd.Series(y).value_counts().min()} "
      f"median={int(pd.Series(y).value_counts().median())} "
      f"max={pd.Series(y).value_counts().max()}", flush=True)

# out-of-fold probabilities on the dorsal-horn subproblem
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(y, y))
oof = np.zeros((len(y), len(CLASSES)), np.float32)
t0 = time.time()
for tr, va in folds:
    oof[va] = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
print(f"OOF computed in {time.time()-t0:.0f}s", flush=True)

prior = M.prior_vector(pd.Series(y), CLASSES)
corrected = M.correct_prior(oof, prior, 0.45)
pred = CLASS_ARR[corrected.argmax(1)]
print(f"\ndorsal-horn 26-way accuracy: {(pred == y).mean():.4f}", flush=True)

rows = []
for k, cls in enumerate(CLASSES):
    hit = (y == cls)
    if hit.sum() < 2:
        continue
    auc = roc_auc_score(hit, oof[:, k])
    recall = (pred[hit] == cls).mean()
    # rank of the true class among the 60 for this class's own cells
    rank = (corrected[hit] > corrected[hit, k][:, None]).sum(1).mean() + 1
    rows.append({"class": cls, "n": int(hit.sum()), "auc": auc, "recall": recall,
                 "mean_rank_of_true": rank,
                 "mean_prob_of_true": oof[hit, k].mean()})
df = pd.DataFrame(rows).sort_values("recall")
pd.set_option("display.width", 200)
print("\n=== ranked by recall (worst first) ===", flush=True)
print(df.head(14).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

sep = df[(df.recall < 0.5) & (df.auc > 0.80)]
print(f"\nclasses with recall<0.5 but AUC>0.80 -- separable but never selected:",
      flush=True)
if len(sep):
    print(sep.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"\n  {sep.n.sum()} training cells live in these classes", flush=True)
else:
    print("  none", flush=True)

hopeless = df[(df.recall < 0.5) & (df.auc <= 0.70)]
print(f"\nclasses with recall<0.5 and AUC<=0.70 -- genuinely not separable:", flush=True)
print(hopeless[["class", "n", "auc", "recall"]].to_string(index=False,
      float_format=lambda v: f"{v:.4f}") if len(hopeless) else "  none", flush=True)

df.to_csv(OUT / "dorsal_horn_separability.csv", index=False)
print(f"\nwrote {OUT/'dorsal_horn_separability.csv'}", flush=True)
