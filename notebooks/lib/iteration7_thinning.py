"""Tier 1c - binomial count thinning, as train-time augmentation and at test time.

This is the one Tier-1 idea that was never fairly tested. It appears exactly once in the
project, in iteration 4 (0.7418), inside the run that the reference-label-column bug
invalidated - so its only measurement is meaningless.

The mechanism is specific to this dataset: median 18-21 transcripts per cell. At that
depth the dominant nuisance is sampling noise in which transcripts happened to be
captured, not biological variation. Binomial thinning (keep each transcript with
probability p) generates cells that are biologically identical and technically noisier,
which is exactly the invariance the classifier should have.

  train-aug   append thinned copies of the training cells, same labels
  TTA         average predictions over thinned copies of the evaluation cells
  NULL        append EXACT DUPLICATES of the training cells instead of thinned copies.
              This is the control that matters: duplication alone changes the bootstrap
              weighting inside every tree, so if plain duplication captures the gain then
              thinning contributed nothing.

The base, reference-transfer and atlas-transfer blocks are all recomputed from the
thinned counts. The spatial and niche blocks are not - they describe a cell's
neighbourhood rather than its own transcripts, and thinning them would model a different
experiment. Noted as a limitation rather than hidden.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt only at Holm-corrected p < 0.05 across the 4 comparisons against the submitted
  baseline, AND only if the variant also beats the duplicate-rows null control.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS = 5, 3
P_KEEP = 0.7
N_THIN = 4              # thinned replicates of the training file

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
GENES = list(counts_train.columns)
glia = meta_train["Region"].isna().to_numpy()
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
n = len(y)

raw = counts_train.to_numpy()
raw = np.rint(raw).astype(np.int64)
print(f"train={n} genes={len(GENES)} median transcripts/cell={np.median(raw.sum(1)):.0f}",
      flush=True)

rng = np.random.default_rng(0)
thinned = [pd.DataFrame(rng.binomial(raw, P_KEEP), index=counts_train.index,
                        columns=counts_train.columns) for _ in range(N_THIN)]
print(f"built {N_THIN} thinned replicates at p={P_KEEP}; "
      f"median transcripts now {np.median(thinned[0].to_numpy().sum(1)):.0f}", flush=True)

# ---------------------------------------------------------------- feature blocks
t0 = time.time()
matrices = [counts_train] + thinned
encoder = OneHotEncoder(handle_unknown="ignore").fit(
    pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str))
BASE = [F.base_block(m, meta_train, encoder) for m in matrices]
print(f"[base]  {BASE[0].shape} x{len(BASE)} ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
EXT, _, _ = F.reference_transfer(GENES, CLASSES, matrices, label_column="voting")
print(f"[ext]   ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
ATL, n_ref = F.atlas_transfer(GENES, CLASSES, matrices)
print(f"[atlas] reference cells={n_ref} ({time.time()-t0:.0f}s)", flush=True)

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
SPA, NIC = c["SPA_TR"], c["NIC_TR"]          # neighbourhood blocks, not thinned
X = [np.hstack([BASE[i], EXT[i], SPA, NIC, ATL[i]]).astype(np.float32)
     for i in range(len(matrices))]
print(f"feature stack {X[0].shape} x{len(X)}", flush=True)

folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
prior_alpha = 0.45


def run(tag, n_aug, duplicate, tta):
    """n_aug extra training copies (thinned, or duplicated if `duplicate`); tta replicates."""
    t0 = time.time()
    ok = np.zeros((N_REPEATS, n), bool)
    for f, (tr, va) in enumerate(folds):
        src = [0] * n_aug if duplicate else list(range(1, n_aug + 1))
        Xtr = np.vstack([X[0][tr]] + [X[i][tr] for i in src])
        ytr = np.tile(y[tr], n_aug + 1)
        prior = M.prior_vector(pd.Series(y[tr]), CLASSES)
        evals = [0] + list(range(1, tta + 1))
        p = np.mean([M.fit_extra_trees(Xtr, pd.Series(ytr), CLASSES, X[i][va],
                                       seeds=SEEDS) for i in evals], axis=0) \
            if tta else M.fit_extra_trees(Xtr, pd.Series(ytr), CLASSES, X[0][va],
                                          seeds=SEEDS)
        p = M.correct_prior(p, prior, prior_alpha)
        ok[f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    a = ok.mean(1)
    print(f"  {tag:34s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[:, glia].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return ok


print(f"\n=== {N_SPLITS}x{N_REPEATS} repeated CV, {len(SEEDS)} seeds ===", flush=True)
results = {}
results["baseline (submitted)"] = run("baseline (submitted)", 0, False, 0)
results["train-aug x2 thinned"] = run("train-aug x2 thinned", 2, False, 0)
results["TTA x5"] = run("TTA x5", 0, False, N_THIN)
results["train-aug x2 + TTA x5"] = run("train-aug x2 + TTA x5", 2, False, N_THIN)
results["duplicate rows x2 (null)"] = run("duplicate rows x2 (null)", 2, True, 0)

base = results["baseline (submitted)"]
null = results["duplicate rows x2 (null)"]
rows = []
for name, ok in results.items():
    if name == "baseline (submitted)":
        continue
    p_base, _ = M.paired_mcnemar(ok.ravel(), base.ravel())
    p_null, _ = M.paired_mcnemar(ok.ravel(), null.ravel())
    rows.append({"variant": name, "accuracy": ok.mean(), "glia": ok[:, glia].mean(),
                 "gain_vs_baseline": ok.mean() - base.mean(), "p_vs_baseline": p_base,
                 "gain_vs_null": ok.mean() - null.mean(), "p_vs_null": p_null})
rows.sort(key=lambda r: r["p_vs_baseline"])
m = len(rows)
print("\n=== paired McNemar vs baseline (Holm-corrected) ===", flush=True)
for i, r in enumerate(rows):
    r["holm_threshold"] = 0.05 / (m - i)
    r["passes"] = bool(r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < r["holm_threshold"]
                       and r["gain_vs_null"] > 0 and r["p_vs_null"] < 0.05)
    print(f"  {r['variant']:26s} {r['gain_vs_baseline']:+.4f} p={r['p_vs_baseline']:.3g} "
          f"(Holm {r['holm_threshold']:.4f}) | vs null {r['gain_vs_null']:+.4f} "
          f"p={r['p_vs_null']:.3g}{'   <== PASSES' if r['passes'] else ''}", flush=True)

print(f"\n  VERDICT: {'ADOPT' if any(r['passes'] for r in rows) else 'DO NOT ADOPT'} "
      f"(pre-registered: Holm p<0.05 vs baseline AND beat the duplicate-rows null)",
      flush=True)
pd.DataFrame(rows).to_csv(OUT / "thinning.csv", index=False)
print(f"\nwrote {OUT/'thinning.csv'}", flush=True)
