"""Iteration 7 - the last decision-free lever: average over near-equal configurations.

Everything else tested in §10 tried to add information. This adds none - it only reduces
variance, which is the one thing still legitimately available:

  * more ExtraTrees seeds (we use 5; variance falls as 1/sqrt(n_seeds))
  * averaging probabilities over several feature-block subsets that CV rates as
    near-equal, rather than betting everything on the single best subset

Neither involves a choice made against the test set. Selection is 5x3 repeated CV on the
training cells, paired McNemar against the submitted configuration.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
glia = meta_train["Region"].isna().to_numpy()
c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
B = {k: c[f"{k.upper()}_TR"] for k in ["base", "ext", "spa", "nic", "atl"]}

SUBSETS = {
    "submitted  base+ext+spa+atl": ["base", "ext", "spa", "atl"],
    "base+ext+atl":                ["base", "ext", "atl"],
    "base+ext+spa+nic+atl":        ["base", "ext", "spa", "nic", "atl"],
    "base+ext+spa":                ["base", "ext", "spa"],
}
X = {n: np.hstack([B[k] for k in ks]).astype(np.float32) for n, ks in SUBSETS.items()}
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(y, y))

def run(Xf, seeds):
    P = np.zeros((3, 5000, len(CLASSES)), np.float32)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(Xf[tr], pd.Series(y[tr]), CLASSES, Xf[va], seeds=seeds)
        P[f // 5, va] = M.correct_prior(p, M.prior_vector(y[tr], CLASSES), 0.45)
    return P

def report(P, tag):
    accs = [accuracy_score(y, np.array(CLASSES)[P[r].argmax(1)]) for r in range(3)]
    pred = np.array(CLASSES)[P.mean(0).argmax(1)]
    b = balanced_accuracy_score(y, pred)
    print(f"  {tag:42s} acc={np.mean(accs):.4f} +/-{np.std(accs):.4f} bal={b:.4f} "
          f"glia={accuracy_score(y[glia], pred[glia]):.4f}", flush=True)
    return np.mean(accs), np.array([np.array(CLASSES)[P[r].argmax(1)] == y for r in range(3)])

rows = []
print("=== seed count, submitted feature set ===", flush=True)
store = {}
for n_seeds in [2, 5, 10, 20]:
    t0 = time.time()
    P = run(X["submitted  base+ext+spa+atl"], tuple(range(n_seeds)))
    a, ok = report(P, f"{n_seeds} ET seeds")
    store[f"seeds{n_seeds}"] = (P, ok)
    rows.append({"config": f"{n_seeds} seeds", "accuracy": a, "seconds": time.time() - t0})

print("\n=== each feature subset alone (5 seeds) ===", flush=True)
subP = {}
for n, Xf in X.items():
    P = run(Xf, (0, 1, 2, 3, 4))
    a, ok = report(P, n)
    subP[n] = P
    rows.append({"config": n, "accuracy": a})

print("\n=== averaging probabilities across subsets ===", flush=True)
for combo in [["submitted  base+ext+spa+atl", "base+ext+atl"],
              ["submitted  base+ext+spa+atl", "base+ext+spa+nic+atl"],
              ["submitted  base+ext+spa+atl", "base+ext+atl", "base+ext+spa+nic+atl"],
              list(SUBSETS)]:
    P = np.mean([subP[k] for k in combo], axis=0)
    a, ok = report(P, f"mean of {len(combo)}: " + ", ".join(k.split()[-1] for k in combo))
    store["ens" + str(len(combo))] = (P, ok)
    rows.append({"config": "mean of " + "+".join(combo), "accuracy": a})

print("\n=== paired McNemar vs the submitted configuration ===", flush=True)
base_ok = store["seeds5"][1].ravel()
for tag, (P, ok) in store.items():
    if tag == "seeds5":
        continue
    o = ok.ravel()
    p, _ = M.paired_mcnemar(o, base_ok)
    print(f"  {tag:12s} gain={o.mean()-base_ok.mean():+.4f}  "
          f"discordant {int((o & ~base_ok).sum())} vs {int((base_ok & ~o).sum())}  p={p:.4g}",
          flush=True)

pd.DataFrame(rows).to_csv(OUT / "ensemble.csv", index=False)
print(f"\nwrote {OUT/'ensemble.csv'}", flush=True)
