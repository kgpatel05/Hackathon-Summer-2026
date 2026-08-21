"""Iteration 9 - final arbitration of the two surviving atlas-neighbourhood channels.

TWO CANDIDATES, BOTH POSITIVE ON TWO PARTITIONS, NEITHER CERTIFIED
-----------------------------------------------------------------
  atlas niche EXPRESSION (S11c, 30 cols)   screen +0.27 p=4.3e-4   confirm +0.11 p=0.129
  atlas neighbourhood COMPOSITION (61 cols) screen +0.52 p=0.012   confirm +0.38 p=0.067

Both restore the tissue context the organisers' 1-in-27 subsample destroyed (70 cells
per section in the challenge file against 964 in the parent atlas), but through
different channels: mean EXPRESSION of the true neighbours over the 200 released genes,
versus the class HISTOGRAM of those neighbours' public annotations.

WHY THE CONFIRMATIONS MISSED, AND WHY THAT IS NOT EVIDENCE OF ABSENCE
--------------------------------------------------------------------
Both protocols correctly used ONE out-of-fold prediction per cell - never flattened
repeated-CV rows, which would inflate significance by reusing the same cell. That is the
right call, and it caps the test at 5,000 independent cells. Composition's confirmation
was 58 wins against 39 losses: z = (58-39)/sqrt(97) = 1.93, p = 0.067. An effect of ~20
net cells is simply at the resolution limit of 5,000 paired observations. Re-rolling
partitions until one clears 0.05 would be exactly the p-hacking this project has refused
five times.

So the decision rule moves from significance to EFFECT SIZE across independent
partitions, pre-registered here before any number below exists.

PRE-REGISTERED RULE (fixed before running)
------------------------------------------
Three FRESH fold partitions (seeds 41, 59, 83), never used by the screen (7) or the
confirmations (23). 20 ET seeds. One out-of-fold prediction per cell per partition, so
each partition yields an independent 5,000-cell McNemar.

Four configurations: baseline, +composition, +niche, +both. Plus the row-shuffled
composition NULL, carried on every partition.

ADOPT the configuration that satisfies ALL of:
  (a) gain > 0 on all three fresh partitions;
  (b) mean gain across the three > +0.20 pt;
  (c) mean gain exceeds the NULL's mean gain by more than 0.20 pt;
  (d) it is the best such configuration by mean gain.
Otherwise DO NOT ADOPT and the submission stays at 0.7796.

An effect-size rule is the honest instrument at the power limit; a p-value rule here
would just be a noisier version of the same question. The null control is what protects
against fooling ourselves, and it is carried on every partition rather than once.

LEGITIMACY
----------
Both blocks read the parent atlas restricted to cells that are NOT in the challenge (all
10,000 removed before any neighbour search), use only the 200 RELEASED genes for
expression, and never read any challenge or test label. Composition additionally reads
the atlas cell-type annotation of EXTERNAL neighbours - the same resource and column the
already-submitted atlas_transfer block trains on. Disclosure for review: those neighbour
annotations were derived by the study from 500 genes, so the block carries 500-gene
information about a cell's MICROENVIRONMENT, never about its own transcriptome.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
PARTITIONS = (41, 59, 83)
SEEDS = tuple(range(20))
ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
n = len(y)

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
BASE = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
                  c["ATL_TR"]]).astype(np.float32)
COMP = np.load(OUT / "atlas_composition_cache.npz")["k10"][:n].astype(np.float32)
NICHE = np.load("outputs/iteration8/atlas_niche.npz")["k50"][:n].astype(np.float32)

# row-shuffled null: same block, same marginals, wrong cells - shuffled WITHIN section so
# the null keeps the section-level structure and only destroys the per-cell assignment
rng = np.random.default_rng(12345)
sections = meta_train["Section_ID"].astype(str).to_numpy()
NULLC = COMP.copy()
for s in np.unique(sections):
    idx = np.flatnonzero(sections == s)
    NULLC[idx] = COMP[rng.permutation(idx)]

CONFIGS = {
    "baseline": BASE,
    "+ composition": np.hstack([BASE, COMP]),
    "+ niche": np.hstack([BASE, NICHE]),
    "+ both": np.hstack([BASE, COMP, NICHE]),
    "+ composition SHUFFLED (null)": np.hstack([BASE, NULLC]),
}
print(f"cells={n} classes={len(CLASSES)}", flush=True)
for k, v in CONFIGS.items():
    print(f"  {k:32s} {v.shape[1]} features", flush=True)
print(f"partitions={PARTITIONS} seeds={len(SEEDS)}", flush=True)


def oof_correct(X, seed):
    """One out-of-fold prediction per cell -> 5,000 independent paired observations."""
    ok = np.zeros(n, bool)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y):
        p = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), ALPHA)
        ok[va] = CLASS_ARR[p.argmax(1)] == y[va]
    return ok


rows = []
for seed in PARTITIONS:
    print(f"\n=== partition seed {seed} ===", flush=True)
    t0 = time.time()
    correct = {name: oof_correct(X, seed) for name, X in CONFIGS.items()}
    base = correct["baseline"]
    for name, ok in correct.items():
        gain = ok.mean() - base.mean()
        if name == "baseline":
            p_val, w, l = 1.0, 0, 0
        else:
            p_val, _ = M.paired_mcnemar(ok, base)
            w = int((ok & ~base).sum()); l = int((base & ~ok).sum())
        rows.append({"partition": seed, "config": name, "accuracy": ok.mean(),
                     "glia": ok[glia].mean(), "neurons": ok[~glia].mean(),
                     "gain_pt": 100 * gain, "wins": w, "losses": l, "p": p_val})
        print(f"  {name:32s} acc={ok.mean():.4f} glia={ok[glia].mean():.4f} "
              f"gain={100*gain:+.2f}pt  {w}w/{l}l  p={p_val:.4g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "arbitrate.csv", index=False)

print("\n=== mean gain across the three fresh partitions ===", flush=True)
summary = df[df.config != "baseline"].groupby("config").agg(
    mean_gain=("gain_pt", "mean"), min_gain=("gain_pt", "min"),
    all_positive=("gain_pt", lambda s: bool((s > 0).all())),
    mean_glia=("glia", "mean"))
print(summary.to_string(), flush=True)

null_mean = summary.loc["+ composition SHUFFLED (null)", "mean_gain"]
cands = summary.drop(index="+ composition SHUFFLED (null)")
eligible = cands[(cands.all_positive) & (cands.mean_gain > 0.20)
                 & (cands.mean_gain - null_mean > 0.20)]
print(f"\n  null mean gain: {null_mean:+.2f} pt", flush=True)
if len(eligible):
    winner = eligible.mean_gain.idxmax()
    print(f"\n  VERDICT: ADOPT '{winner}' "
          f"(mean {eligible.loc[winner,'mean_gain']:+.2f} pt, "
          f"min {eligible.loc[winner,'min_gain']:+.2f} pt)", flush=True)
else:
    print("\n  VERDICT: DO NOT ADOPT (no configuration met the pre-registered rule)",
          flush=True)
print(f"\nwrote {OUT/'arbitrate.csv'}", flush=True)
