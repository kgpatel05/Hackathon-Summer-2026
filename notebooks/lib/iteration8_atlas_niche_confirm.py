"""Iteration 8d-confirm - the atlas-niche block on an INDEPENDENT fold partition.

The screen (fold seed 7) gave +0.27 pt at p=4.3e-4, passing Holm. That is an order of
magnitude stronger than the section-profile candidate in §10p-1, which reached p=0.034 and
then REVERSED SIGN on a new partition. Same discipline applies: one hypothesis, full
power, a fold seed the screen never saw.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
SEEDS = tuple(range(20))
ctr, mtr, cte, mte = F.load_challenge()
y = mtr[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y)); CA = np.array(CLASSES)
glia = mtr["Region"].isna().to_numpy()
c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz", allow_pickle=True)
an = np.load(OUT / "atlas_niche.npz")["k50"][:len(y)]
CORE = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["ATL_TR"]]).astype(np.float32)
A = np.hstack([CORE, c["NIC_TR"]]).astype(np.float32)
B = np.hstack([CORE, c["NIC_TR"], an]).astype(np.float32)
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=23).split(y, y))
print(f"submitted {A.shape} | + atlas niche {B.shape} | {len(folds)} folds seed 23", flush=True)

def run(X, tag):
    t0 = time.time(); ok = np.zeros((5, len(y)), bool)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
        ok[f // 5, va] = CA[p.argmax(1)] == y[va]
    a = ok.mean(1)
    print(f"  {tag:24s} acc={a.mean():.4f} +/-{a.std():.4f} glia={ok[:, glia].mean():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok

okA = run(A, "submitted"); okB = run(B, "+ atlas niche k=50")
p, _ = M.paired_mcnemar(okB.ravel(), okA.ravel())
gain = okB.mean() - okA.mean()
b_only = int((okB.ravel() & ~okA.ravel()).sum()); a_only = int((okA.ravel() & ~okB.ravel()).sum())
print(f"\n=== paired McNemar, ONE pre-registered comparison, fold seed 23 ===", flush=True)
print(f"  gain {gain:+.4f}\n  discordant {b_only} for +atlas-niche vs {a_only} for submitted", flush=True)
print(f"  p    {p:.4g}", flush=True)
print(f"\n  VERDICT: {'ADOPT' if gain > 0 and p < 0.05 else 'DO NOT ADOPT'}", flush=True)
pd.DataFrame([{"gain": gain, "p": p, "acc_base": okA.mean(), "acc_cand": okB.mean(),
               "glia_base": okA[:, glia].mean(), "glia_cand": okB[:, glia].mean(),
               "fold_seed": 23, "adopt": bool(gain > 0 and p < 0.05)}]).to_csv(
    OUT / "atlas_niche_confirm.csv", index=False)
