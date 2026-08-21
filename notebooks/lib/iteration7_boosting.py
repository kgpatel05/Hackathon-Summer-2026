"""Tier 2c - does tuned gradient boosting add anything, alone or blended?

Prior evidence, both partial:
  * iteration 5 scanned an UNTUNED XGBoost on base+external: 0.7578, below the ET.
  * §10b probed HistGradientBoosting-250 on 86k atlas glia: 0.6624, well below logistic
    regression at 0.7133 and ET at 0.6822.

Neither tested a properly configured booster on the full 529-feature stack, which is the
question a reviewer would actually ask. Boosting also has a real reason to differ from
Extra Trees here: ET splits at random thresholds on a random feature subset, which
suppresses variance but wastes the dense, highly informative reference-transfer blocks;
a booster fits those greedily. Different bias, so possibly complementary errors - and
blending, not replacement, is the realistic route to a gain.

Out-of-fold probabilities are computed ONCE per model, then every blend is evaluated from
storage. This costs one pass instead of one per weight.

lightgbm and catboost are not installed in this environment and were not installed for
this test; xgboost 3.0.2 and sklearn's HistGradientBoosting cover the same model class.

REDUCED POWER, stated up front: boosting 60 classes is expensive, so this uses 5x2
repeated CV (10 folds) rather than the 25 folds used elsewhere. A borderline result here
would need re-running at full power before it could be adopted.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt only at Holm-corrected p < 0.05 across the 6 comparisons against the ET baseline.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS = 5, 2

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
y_idx = np.searchsorted(CLASS_ARR, y)
K = len(CLASSES)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
               c["ATL_TR"]]).astype(np.float32)
n = len(X)
print(f"train={n} features={X.shape[1]} classes={K}", flush=True)

folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
P = {m: np.zeros((N_REPEATS, n, K), np.float32) for m in ("et", "xgb", "hgb")}
# per-cell copy of its own fold's training prior, so prior correction never sees a
# validation label even though blending happens after the CV loop
PRIOR = np.zeros((N_REPEATS, n, K), np.float32)
t0 = time.time()

for f, (tr, va) in enumerate(folds):
    r = f // N_SPLITS
    PRIOR[r, va] = M.prior_vector(pd.Series(y[tr]), CLASSES)[None, :]
    P["et"][r, va] = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)

    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
                        tree_method="hist", objective="multi:softprob",
                        num_class=K, n_jobs=-1, random_state=0, verbosity=0)
    xgb.fit(X[tr], y_idx[tr])
    P["xgb"][r, va] = xgb.predict_proba(X[va])

    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)
    hgb.fit(X[tr], y[tr])
    P["hgb"][r, va] = M.align_proba(hgb, X[va], CLASSES)
    print(f"  fold {f+1}/{len(folds)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\nOOF probabilities computed in {time.time()-t0:.0f}s", flush=True)

BLENDS = {
    "ET (submitted)":            {"et": 1.0},
    "XGBoost alone":             {"xgb": 1.0},
    "HistGradientBoosting alone": {"hgb": 1.0},
    "0.8 ET + 0.2 XGB":          {"et": 0.8, "xgb": 0.2},
    "0.7 ET + 0.3 XGB":          {"et": 0.7, "xgb": 0.3},
    "0.5 ET + 0.5 XGB":          {"et": 0.5, "xgb": 0.5},
    "0.6 ET + 0.2 XGB + 0.2 HGB": {"et": 0.6, "xgb": 0.2, "hgb": 0.2},
}

results = {}
print("\n=== 5x2 repeated CV ===", flush=True)
for name, weights in BLENDS.items():
    ok = np.zeros((N_REPEATS, n), bool)
    for r in range(N_REPEATS):
        mix = sum(w * P[m][r] for m, w in weights.items())
        mix = mix / (PRIOR[r] ** 0.45)
        mix /= np.clip(mix.sum(1, keepdims=True), 1e-12, None)
        ok[r] = CLASS_ARR[mix.argmax(1)] == y
    results[name] = ok
    a = ok.mean(1)
    print(f"  {name:28s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[:, glia].mean():.4f}", flush=True)

base = results["ET (submitted)"]
rows = []
for name, ok in results.items():
    if name == "ET (submitted)":
        continue
    p, _ = M.paired_mcnemar(ok.ravel(), base.ravel())
    rows.append({"variant": name, "accuracy": ok.mean(), "glia": ok[:, glia].mean(),
                 "gain_vs_et": ok.mean() - base.mean(), "p_vs_et": p})
rows.sort(key=lambda r: r["p_vs_et"])
m = len(rows)
print("\n=== paired McNemar vs ET (Holm-corrected) ===", flush=True)
for i, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (m - i)
    r["passes"] = bool(r["gain_vs_et"] > 0 and r["p_vs_et"] < r["holm_threshold"])
    print(f"  {r['variant']:28s} {r['gain_vs_et']:+.4f} p={r['p_vs_et']:.3g} "
          f"(Holm {r['holm_threshold']:.4f})"
          f"{'   <== PASSES' if r['passes'] else ''}", flush=True)

print(f"\n  VERDICT: {'ADOPT' if any(r['passes'] for r in rows) else 'DO NOT ADOPT'} "
      f"(pre-registered: Holm p<0.05; note reduced power, 10 folds)", flush=True)
pd.DataFrame(rows).to_csv(OUT / "boosting.csv", index=False)
print(f"\nwrote {OUT/'boosting.csv'}", flush=True)
