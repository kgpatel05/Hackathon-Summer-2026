"""Tier 2b - are the Extra Trees hyperparameters still right at 529 features?

`ET_KWARGS = n_estimators=600, max_features="sqrt", min_samples_leaf=2` was tuned back
when the feature stack was ~440 columns. It is now 529, and `sqrt` samples only 23 of
them at each split. Three of the five blocks (ext, atlas, niche) are dense, informative,
and highly correlated within themselves, so a 23-column sample may be missing them too
often.

WHY THIS IS RUN AS NESTED CV RATHER THAN A GRID SEARCH: the maximum of a grid evaluated
on the same folds used to report it is selection-optimistic - that is precisely the
mistake that made CV 1.25 pt optimistic against the test set, and §10o-2 measured it
again when per-fold alpha tuning LOST 0.08 pt to a frozen constant. So the grid is
searched on inner folds only, and the outer folds see just one number: what re-tuning
actually delivers, honestly.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt re-tuning only if nested CV beats the frozen configuration at p < 0.05, paired on
  identical folds. The frozen configuration wins ties.
"""
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
OUTER_SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS, N_INNER = 5, 3, 3

GRID = [dict(n_estimators=600, max_features=mf, min_samples_leaf=leaf, n_jobs=-1)
        for mf, leaf in product(["sqrt", 0.1, 0.2], [1, 2, 4])]
FROZEN = dict(M.ET_KWARGS)

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
               c["ATL_TR"]]).astype(np.float32)
n = len(X)
print(f"train={n} features={X.shape[1]} grid={len(GRID)} configs", flush=True)
print(f"frozen: {FROZEN}", flush=True)
print(f"sqrt(529) = {int(np.sqrt(X.shape[1]))} features per split; "
      f"0.1 = {int(0.1*X.shape[1])}, 0.2 = {int(0.2*X.shape[1])}", flush=True)


def fit_predict(kwargs, Xtr, ytr, Xev, seeds):
    stacked = np.zeros((len(Xev), len(CLASSES)), np.float32)
    for s in seeds:
        model = ExtraTreesClassifier(random_state=s, **kwargs).fit(Xtr, ytr)
        stacked += M.align_proba(model, Xev, CLASSES)
    return stacked / len(seeds)


folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
ok_frozen = np.zeros((N_REPEATS, n), bool)
ok_tuned = np.zeros((N_REPEATS, n), bool)
picked = []
t0 = time.time()

for f, (tr, va) in enumerate(folds):
    prior = M.prior_vector(pd.Series(y[tr]), CLASSES)

    # ---- inner search: one seed per config, 3 inner folds
    scores = np.zeros(len(GRID))
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=11)
    for itr, iva in inner.split(tr, y[tr]):
        a, b = tr[itr], tr[iva]
        for g, kwargs in enumerate(GRID):
            p = fit_predict(kwargs, X[a], pd.Series(y[a]), X[b], seeds=(0,))
            p = M.correct_prior(p, M.prior_vector(pd.Series(y[a]), CLASSES), 0.45)
            scores[g] += (CLASS_ARR[p.argmax(1)] == y[b]).sum()
    best = GRID[int(scores.argmax())]
    picked.append(f"mf={best['max_features']},leaf={best['min_samples_leaf']}")

    # ---- outer evaluation
    for kwargs, store in ((FROZEN, ok_frozen), (best, ok_tuned)):
        p = fit_predict(kwargs, X[tr], pd.Series(y[tr]), X[va], seeds=OUTER_SEEDS)
        p = M.correct_prior(p, prior, 0.45)
        store[f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    print(f"  fold {f+1}/{len(folds)} picked {picked[-1]} ({time.time()-t0:.0f}s)",
          flush=True)

print(f"\n=== nested {N_SPLITS}x{N_REPEATS} CV ({time.time()-t0:.0f}s) ===", flush=True)
for tag, ok in (("frozen (submitted)", ok_frozen), ("re-tuned per fold", ok_tuned)):
    a = ok.mean(1)
    print(f"  {tag:22s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[:, glia].mean():.4f}", flush=True)
print(f"\n  configs chosen: {pd.Series(picked).value_counts().to_dict()}", flush=True)

p, _ = M.paired_mcnemar(ok_tuned.ravel(), ok_frozen.ravel())
gain = ok_tuned.mean() - ok_frozen.mean()
print(f"\n=== paired McNemar ===", flush=True)
print(f"  gain {gain:+.4f}  p={p:.4g}", flush=True)
print(f"\n  VERDICT: {'ADOPT' if (gain > 0 and p < 0.05) else 'DO NOT ADOPT'} "
      f"(pre-registered: p<0.05, frozen wins ties)", flush=True)

pd.DataFrame([{"comparison": "re-tuned vs frozen", "acc_frozen": ok_frozen.mean(),
               "acc_tuned": ok_tuned.mean(), "gain": gain, "mcnemar_p": p,
               "folds": len(folds), "grid": len(GRID),
               "configs_chosen": pd.Series(picked).value_counts().to_dict(),
               "adopt": bool(gain > 0 and p < 0.05)}]).to_csv(
    OUT / "et_retune.csv", index=False)
print(f"\nwrote {OUT/'et_retune.csv'}", flush=True)
