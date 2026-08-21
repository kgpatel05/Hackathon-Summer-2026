"""Iteration 7 - is the niche block a real gain, or the best number out of a scan of four?

The subset scan showed base+ext+spa+nic+atl at 0.7929 against the submitted
base+ext+spa+atl at 0.7905. That is +0.24 pt, larger than anything else found in
iteration 7 - but it is also the maximum of four configurations, which is exactly the
selection pressure that made CV 1.3 pt optimistic in the first place.

So this is a single pre-registered comparison, decided before looking:
  * 5x5 repeated stratified CV (25 folds, up from 15)
  * 20 ET seeds, so seed noise is negligible (sd was 0.0002 at 20 seeds)
  * identical folds, paired McNemar
  * ADOPT ONLY IF p < 0.05

Context that argues against: in iteration 5's scan the niche block was neutral
(S3 0.7807, S4 0.7816 vs S2 0.7819). It only looks useful once the atlas block is present.
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

A = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["ATL_TR"]]).astype(np.float32)
B = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
               c["ATL_TR"]]).astype(np.float32)
SEEDS = tuple(range(20))
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7).split(y, y))
print(f"submitted {A.shape} | +niche {B.shape} | {len(folds)} folds, {len(SEEDS)} seeds",
      flush=True)

def run(X, tag):
    t0 = time.time(); ok = np.zeros((5, 5000), bool)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(y[tr], CLASSES), 0.45)
        ok[f // 5, va] = np.array(CLASSES)[p.argmax(1)] == y[va]
    accs = ok.mean(1)
    print(f"  {tag:26s} acc={accs.mean():.4f} +/-{accs.std():.4f} "
          f"glia={ok.ravel().reshape(5,-1)[:, glia].mean():.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    return ok

print("\n=== 5x5 repeated CV, 20 seeds ===", flush=True)
okA = run(A, "submitted")
okB = run(B, "+ niche block")

p, table = M.paired_mcnemar(okB.ravel(), okA.ravel())
gain = okB.mean() - okA.mean()
b_only, a_only = int((okB.ravel() & ~okA.ravel()).sum()), int((okA.ravel() & ~okB.ravel()).sum())
print(f"\n=== paired McNemar ===", flush=True)
print(f"  gain      {gain:+.4f}", flush=True)
print(f"  discordant {b_only} for +niche vs {a_only} for submitted", flush=True)
print(f"  p         {p:.4g}", flush=True)
print(f"\n  VERDICT: {'ADOPT' if p < 0.05 and gain > 0 else 'DO NOT ADOPT'} "
      f"(threshold p<0.05, pre-registered)", flush=True)

pd.DataFrame([{"comparison": "+niche vs submitted", "gain": gain, "mcnemar_p": p,
               "discordant_niche": b_only, "discordant_submitted": a_only,
               "folds": len(folds), "seeds": len(SEEDS),
               "adopt": bool(p < 0.05 and gain > 0)}]).to_csv(
    OUT / "niche_test.csv", index=False)
