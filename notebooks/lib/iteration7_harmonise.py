"""Tier 1a - is the 11-point per-mouse spread a technical batch effect?

DIAGNOSTIC THAT MOTIVATES THIS (measured on the submitted prediction, §10n follow-up):

  per-mouse accuracy      0.7423 (F2) -> 0.8263 (F5), spread 8.4 pt
  per-mouse GLIA accuracy 0.6544 (F2) -> 0.7643 (F5), spread 11.0 pt
  corr(accuracy, glia fraction) = 0.036

The spread is not composition - mice that are harder are not the mice with more glia.
That points at technical variation in capture efficiency between animals. If so,
removing the per-mouse mean of each gene should help.

ARGUMENT AGAINST, stated up front: `Mouse_ID` is already a one-hot feature, so the
trees can condition on animal already. Harmonisation does not add information; it makes
an existing effect representable in ONE split instead of requiring the tree to split on
mouse before every gene threshold. That is a real but second-order benefit.

RISK, also stated up front: cell-type composition genuinely differs between animals, so
per-group centring removes real biological signal along with the technical offset. A
mouse with more oligodendrocytes really does have a higher mean Plp1. This is why the
section-level variant is expected to be worse - at ~70 cells per section the composition
estimate is pure noise.

All group statistics are computed over all 10,000 cells (train + test) using only the
group label, never a class label. That is transductive but leak-free.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt a variant only if it beats the submitted baseline at Holm-corrected p < 0.05
  across the 5 comparisons, AND beats the random-group null control.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS = 5, 5

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
N_GENES = counts_train.shape[1]
BASE_TR, BASE_TE = c["BASE_TR"], c["BASE_TE"]
OTHER_TR = np.hstack([c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], c["ATL_TR"]])
# the first N_GENES columns of the base block are log1p expression
EXPR_ALL = np.vstack([BASE_TR[:, :N_GENES], BASE_TE[:, :N_GENES]]).astype(np.float64)
REST_TR = BASE_TR[:, N_GENES:]
n_tr = len(BASE_TR)

mouse = meta_all["Mouse_ID"].astype(str).to_numpy()
section = meta_all["Section_ID"].astype(str).to_numpy()
rng = np.random.default_rng(0)
# a random partition with exactly the same group sizes as Mouse_ID
random_group = np.empty(len(mouse), object)
order = rng.permutation(len(mouse))
start = 0
for g, size in pd.Series(mouse).value_counts().items():
    random_group[order[start:start + size]] = f"rand_{g}"
    start += size
random_group = random_group.astype(str)


def group_centre(matrix, groups, scale=False):
    out = matrix.copy()
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        block = matrix[rows]
        out[rows] = block - block.mean(0)
        if scale:
            out[rows] /= block.std(0) + 1e-6
    return out


def group_rank(matrix, groups):
    """Per-group rank transform of each gene to [0, 1] - quantile-matches the animals."""
    out = np.empty_like(matrix)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        block = matrix[rows]
        ranks = block.argsort(0).argsort(0).astype(np.float64)
        out[rows] = ranks / max(len(rows) - 1, 1)
    return out


VARIANTS = {
    "baseline (submitted)": EXPR_ALL,
    "mouse gene-centred": group_centre(EXPR_ALL, mouse),
    "mouse gene z-scored": group_centre(EXPR_ALL, mouse, scale=True),
    "section gene-centred": group_centre(EXPR_ALL, section),
    "mouse rank (quantile)": group_rank(EXPR_ALL, mouse),
    "random group (null)": group_centre(EXPR_ALL, random_group),
}
print(f"train={n_tr} genes={N_GENES} variants={len(VARIANTS)} "
      f"folds={N_SPLITS*N_REPEATS} seeds={len(SEEDS)}", flush=True)
for name, expr in VARIANTS.items():
    print(f"  {name:24s} mean|x|={np.abs(expr[:n_tr]).mean():.4f}", flush=True)

folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))


def run(expr, tag):
    X = np.hstack([expr[:n_tr].astype(np.float32), REST_TR, OTHER_TR]).astype(np.float32)
    t0 = time.time()
    ok = np.zeros((N_REPEATS, n_tr), bool)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
        ok[f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    accs = ok.mean(1)
    print(f"  {tag:24s} acc={accs.mean():.4f} +/-{accs.std():.4f} "
          f"glia={ok[:, glia].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return ok


print(f"\n=== {N_SPLITS}x{N_REPEATS} repeated CV ===", flush=True)
results = {name: run(expr, name) for name, expr in VARIANTS.items()}

base = results["baseline (submitted)"]
null = results["random group (null)"]
print("\n=== paired McNemar vs baseline (identical folds) ===", flush=True)
rows = []
for name, ok in results.items():
    if name == "baseline (submitted)":
        continue
    p_base, _ = M.paired_mcnemar(ok.ravel(), base.ravel())
    p_null, _ = M.paired_mcnemar(ok.ravel(), null.ravel())
    rows.append({"variant": name, "accuracy": ok.mean(), "glia": ok[:, glia].mean(),
                 "gain_vs_baseline": ok.mean() - base.mean(), "p_vs_baseline": p_base,
                 "gain_vs_null": ok.mean() - null.mean(), "p_vs_null": p_null})

# Holm-Bonferroni across the 5 comparisons
rows.sort(key=lambda r: r["p_vs_baseline"])
m = len(rows)
for i, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (m - i)
    r["passes"] = bool(r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < r["holm_threshold"]
                       and r["gain_vs_null"] > 0 and r["p_vs_null"] < 0.05)
for r in rows:
    print(f"  {r['variant']:24s} {r['gain_vs_baseline']:+.4f} p={r['p_vs_baseline']:.3g} "
          f"(Holm {r['holm_threshold']:.4f}) | vs null {r['gain_vs_null']:+.4f} "
          f"p={r['p_vs_null']:.3g}{'   <== PASSES' if r['passes'] else ''}", flush=True)

verdict = any(r["passes"] for r in rows)
print(f"\n  VERDICT: {'ADOPT' if verdict else 'DO NOT ADOPT'} "
      f"(pre-registered: Holm p<0.05 vs baseline AND beat the null control)", flush=True)
pd.DataFrame(rows).to_csv(OUT / "harmonise.csv", index=False)
print(f"\nwrote {OUT/'harmonise.csv'}", flush=True)
