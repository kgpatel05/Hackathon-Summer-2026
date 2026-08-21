"""Iteration 8c - a separate prior-correction exponent for each Region group.

WHAT §8b ESTABLISHED. Inside the dorsal horn, every class at zero recall has AUC 0.95-0.99
and its own cells rank ~2nd of 26. They are separable and simply never win an argmax.
That is a decision-rule failure, not an information failure, and it is worth up to 106
cells - more than half the 177 needed for first place.

WHY ONE ALPHA CANNOT FIX IT. `correct_prior` divides by prior**alpha with a single global
alpha = 0.45, chosen to maximise overall accuracy. 63.5% of cells are glia, whose 21
classes are large and evenly populated and therefore want a small alpha. The dorsal horn
is 26 classes with sizes 6-124 and wants a much larger one. A single constant is a
compromise dominated by the glia.

Region is a deterministic function of the label (§8a), so the 60-way problem decomposes
into disjoint subproblems - Region 1 = 26 dorsal-horn classes, 2 = DM_ex_Zfhx3 alone,
3 = 4 medial, 4 = 7 medioventral, 5 = VH_in_Chat alone, missing = the 21 non-neuronal
classes. Each subproblem gets its own alpha.

Note that a group-CONDITIONAL PRIOR would be a no-op: prior_g(k) = prior(k)/P(g), and
P(g)**alpha is constant within the group, so it cannot change an argmax. A group-specific
EXPONENT is what actually re-weights the classes against each other.

HONEST PROTOCOL. Six free parameters selected on data is exactly the over-selection that
has burned this project repeatedly (per-fold alpha lost 0.08 pt in §10o-2; the section
profile reversed sign on a new fold seed in §10p-1). So the exponents are chosen on INNER
folds only and scored on outer folds the selection never saw, and the comparison against
the frozen global alpha = 0.45 is paired on identical folds.

PRE-REGISTERED: adopt only if nested CV beats global alpha=0.45 at p < 0.05. The global
constant wins ties.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS, N_INNER = 5, 5, 3
GRID = (0.0, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5)
BASE_ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
K = len(CLASSES)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
               c["ATL_TR"]]).astype(np.float32)
n = len(X)

region = meta_train["Region"].astype(str).to_numpy()
region_te = meta_test["Region"].astype(str).to_numpy()
GROUPS = sorted(set(region))
print(f"train={n} groups={GROUPS}", flush=True)
for g in GROUPS:
    m = region == g
    print(f"  Region {g:5s}: {m.sum():5d} cells, {len(set(y[m])):2d} classes", flush=True)


def apply_alpha(probs, prior, groups, alpha_by_group):
    out = np.empty_like(probs)
    for g, a in alpha_by_group.items():
        m = groups == g
        if m.any():
            out[m] = M.correct_prior(probs[m], prior, a)
    return out


def pick_alphas(probs, prior, groups, truth):
    """Best exponent per group, chosen independently since the groups are disjoint."""
    chosen = {}
    for g in GROUPS:
        m = groups == g
        if not m.any():
            chosen[g] = BASE_ALPHA
            continue
        best, best_a = -1.0, BASE_ALPHA
        for a in GRID:
            acc = (CLASS_ARR[M.correct_prior(probs[m], prior, a).argmax(1)] == truth[m]).mean()
            if acc > best:
                best, best_a = acc, a
        chosen[g] = best_a
    return chosen


folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
ok_base = np.zeros((N_REPEATS, n), bool)
ok_group = np.zeros((N_REPEATS, n), bool)
picks = []
t0 = time.time()

for f, (tr, va) in enumerate(folds):
    prior = M.prior_vector(pd.Series(y[tr]), CLASSES)

    # inner OOF probabilities -> choose exponents without touching the outer fold
    inner_oof = np.zeros((len(tr), K), np.float32)
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=11)
    for itr, iva in inner.split(tr, y[tr]):
        inner_oof[iva] = M.fit_extra_trees(X[tr[itr]], pd.Series(y[tr[itr]]), CLASSES,
                                           X[tr[iva]], seeds=(0, 1, 2))
    chosen = pick_alphas(inner_oof, prior, region[tr], y[tr])
    picks.append(chosen)

    p = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
    ok_base[f // N_SPLITS, va] = CLASS_ARR[
        M.correct_prior(p, prior, BASE_ALPHA).argmax(1)] == y[va]
    ok_group[f // N_SPLITS, va] = CLASS_ARR[
        apply_alpha(p, prior, region[va], chosen).argmax(1)] == y[va]
    if f % 5 == 0:
        print(f"  fold {f+1}/{len(folds)} chose {chosen} ({time.time()-t0:.0f}s)",
              flush=True)

print(f"\n=== nested {N_SPLITS}x{N_REPEATS} CV ({time.time()-t0:.0f}s) ===", flush=True)
for tag, ok in (("global alpha=0.45", ok_base), ("per-Region alpha", ok_group)):
    a = ok.mean(1)
    print(f"  {tag:20s} acc={a.mean():.4f} +/-{a.std():.4f} glia={ok[:, glia].mean():.4f} "
          f"neurons={ok[:, ~glia].mean():.4f}", flush=True)

gain = ok_group.mean() - ok_base.mean()
p_val, _ = M.paired_mcnemar(ok_group.ravel(), ok_base.ravel())
print(f"\n  gain {gain:+.4f}  p={p_val:.4g}", flush=True)
for g in GROUPS:
    vals = pd.Series([p[g] for p in picks]).value_counts().to_dict()
    print(f"    Region {g:5s} chose {vals}", flush=True)
adopt = gain > 0 and p_val < 0.05
print(f"\n  VERDICT: {'ADOPT' if adopt else 'DO NOT ADOPT'} "
      f"(pre-registered p<0.05, global constant wins ties)", flush=True)

pd.DataFrame([{"acc_global": ok_base.mean(), "acc_group": ok_group.mean(), "gain": gain,
               "mcnemar_p": p_val, "adopt": bool(adopt),
               "picks": str(pd.Series([str(p) for p in picks]).value_counts().to_dict())}
              ]).to_csv(OUT / "group_alpha.csv", index=False)
print(f"\nwrote {OUT/'group_alpha.csv'}", flush=True)
