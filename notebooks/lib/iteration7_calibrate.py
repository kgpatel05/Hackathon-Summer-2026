"""Tier 1b - is the alpha=0.45 prior power transform leaving anything on the table?

The submitted model divides the Extra Trees probabilities by prior**0.45 and renormalises.
That constant was hand-tuned by CV. Two things make it worth revisiting properly:

  * train and test are an IID split of the SAME cells - all 10 mice, 6 batches and 108
    sections appear in both in near-identical proportion. So the true prior shift is
    ZERO, and alpha is not correcting a shift at all. What it is really doing is undoing
    the shrinkage that averaged decision trees apply to rare classes. A calibrator aimed
    at that directly should do better than a one-parameter power law.
  * 4 classes sit at zero recall and 299 cells live in classes below 0.5 recall.

NOTE ON WHAT CANNOT WORK: temperature scaling is monotone and identical across classes,
so it cannot change an argmax and therefore cannot change accuracy by a single cell. Only
PER-CLASS transforms are tested here.

Every calibrator is fitted on inner-CV out-of-fold probabilities from the outer fold's
training cells only. It never sees the outer validation cells.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt only at Holm-corrected p < 0.05 across the 5 comparisons against the submitted
  alpha=0.45 baseline, on identical folds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
OUTER_SEEDS, INNER_SEEDS = tuple(range(10)), (0, 1, 2, 3, 4)
N_SPLITS, N_REPEATS, N_INNER = 5, 5, 3
ALPHA_GRID = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
EPS = 1e-9

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
y_idx = np.searchsorted(CLASS_ARR, y)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
               c["ATL_TR"]]).astype(np.float32)
n = len(X)
print(f"train={n} features={X.shape[1]} classes={len(CLASSES)}", flush=True)


# ------------------------------------------------------------------ calibrators
def fit_vector_scaling(probs, target, iters=300):
    """logits = a_c * log p_c + b_c, fitted by multinomial NLL. Changes the argmax."""
    lp = torch.log(torch.tensor(probs, dtype=torch.float32) + EPS)
    t = torch.tensor(target, dtype=torch.long)
    a = torch.ones(probs.shape[1], requires_grad=True)
    b = torch.zeros(probs.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lp * a + b, t)
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().numpy(), b.detach().numpy()


def apply_vector_scaling(probs, ab):
    a, b = ab
    z = np.log(probs + EPS) * a + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_isotonic(probs, target):
    models = []
    for k in range(probs.shape[1]):
        hit = (target == k).astype(float)
        if hit.sum() == 0 or hit.sum() == len(hit):
            models.append(None)
            continue
        models.append(IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                      .fit(probs[:, k], hit))
    return models


def apply_isotonic(probs, models):
    out = np.zeros_like(probs)
    for k, mdl in enumerate(models):
        out[:, k] = probs[:, k] if mdl is None else mdl.predict(probs[:, k])
    total = out.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return out / total


def fit_platt(probs, target):
    models = []
    for k in range(probs.shape[1]):
        hit = (target == k).astype(int)
        if hit.sum() < 2 or hit.sum() == len(hit):
            models.append(None)
            continue
        models.append(LogisticRegression(C=1.0, max_iter=1000)
                      .fit(np.log(probs[:, [k]] + EPS), hit))
    return models


def apply_platt(probs, models):
    out = np.zeros_like(probs)
    for k, mdl in enumerate(models):
        out[:, k] = (probs[:, k] if mdl is None
                     else mdl.predict_proba(np.log(probs[:, [k]] + EPS))[:, 1])
    total = out.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return out / total


VARIANTS = ["alpha=0.45 (submitted)", "no correction (alpha=0)", "alpha tuned per fold",
            "vector scaling", "per-class isotonic", "per-class Platt"]

folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
ok = {v: np.zeros((N_REPEATS, n), bool) for v in VARIANTS}
alpha_chosen = []
t0 = time.time()

for f, (tr, va) in enumerate(folds):
    # inner OOF probabilities on the outer-training cells, for fitting calibrators
    inner_oof = np.zeros((len(tr), len(CLASSES)), np.float32)
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=11)
    for itr, iva in inner.split(tr, y[tr]):
        inner_oof[iva] = M.fit_extra_trees(X[tr[itr]], pd.Series(y[tr[itr]]), CLASSES,
                                           X[tr[iva]], seeds=INNER_SEEDS)
    y_in = y_idx[tr]

    prior = M.prior_vector(pd.Series(y[tr]), CLASSES)
    best_alpha = max(ALPHA_GRID, key=lambda a: (
        CLASS_ARR[M.correct_prior(inner_oof, prior, a).argmax(1)] == y[tr]).mean())
    alpha_chosen.append(best_alpha)
    ab = fit_vector_scaling(inner_oof, y_in)
    iso = fit_isotonic(inner_oof, y_in)
    platt = fit_platt(inner_oof, y_in)

    raw = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=OUTER_SEEDS)
    out = {
        "alpha=0.45 (submitted)": M.correct_prior(raw, prior, 0.45),
        "no correction (alpha=0)": raw,
        "alpha tuned per fold": M.correct_prior(raw, prior, best_alpha),
        "vector scaling": apply_vector_scaling(raw, ab),
        "per-class isotonic": apply_isotonic(raw, iso),
        "per-class Platt": apply_platt(raw, platt),
    }
    for v, p in out.items():
        ok[v][f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    if f % 5 == 0:
        print(f"  fold {f+1}/{len(folds)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== {N_SPLITS}x{N_REPEATS} nested CV, {len(OUTER_SEEDS)} seeds "
      f"({time.time()-t0:.0f}s) ===", flush=True)
for v in VARIANTS:
    a = ok[v].mean(1)
    print(f"  {v:26s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[v][:, glia].mean():.4f}", flush=True)
print(f"\n  alpha chosen per fold: {pd.Series(alpha_chosen).value_counts().to_dict()}",
      flush=True)

base = ok["alpha=0.45 (submitted)"]
rows = []
for v in VARIANTS:
    if v == "alpha=0.45 (submitted)":
        continue
    p, _ = M.paired_mcnemar(ok[v].ravel(), base.ravel())
    rows.append({"variant": v, "accuracy": ok[v].mean(), "glia": ok[v][:, glia].mean(),
                 "gain_vs_baseline": ok[v].mean() - base.mean(), "p_vs_baseline": p})
rows.sort(key=lambda r: r["p_vs_baseline"])
m = len(rows)
print("\n=== paired McNemar vs alpha=0.45 (Holm-corrected) ===", flush=True)
for i, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (m - i)
    r["passes"] = bool(r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < r["holm_threshold"])
    print(f"  {r['variant']:26s} {r['gain_vs_baseline']:+.4f} p={r['p_vs_baseline']:.3g} "
          f"(Holm {r['holm_threshold']:.4f})"
          f"{'   <== PASSES' if r['passes'] else ''}", flush=True)

print(f"\n  VERDICT: {'ADOPT' if any(r['passes'] for r in rows) else 'DO NOT ADOPT'} "
      f"(pre-registered: Holm p<0.05)", flush=True)
pd.DataFrame(rows).to_csv(OUT / "calibrate.csv", index=False)
print(f"\nwrote {OUT/'calibrate.csv'}", flush=True)
