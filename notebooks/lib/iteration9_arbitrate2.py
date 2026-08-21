"""Iteration 9 - is the metadata-conditioned ET atlas block STILL worth anything now that
the neighbourhood blocks are in?

THE CANDIDATE. The deployed atlas block is one L2 logistic on expression alone (0.5992
standalone). An ExtraTrees fitted on the same atlas cells with expression + QC + METADATA
reaches 0.6790 standalone. Its screen was strong and replicated:

    variant                          partition 7        partition 23
    replace logistic with ET         0.7956 (p=0.036)   0.7962 (p=0.397)
    CONCAT both blocks               0.7972 (p=0.0016)  0.8006 (p=0.0073)
    NULL row-shuffled ET block       0.7900 (p=0.585)   0.7910 (p=0.428)

THE REASON TO DOUBT IT NOW. That screen ran against the OLD 529-feature baseline. The
diagnostic behind the block is explicit that ExtraTrees is a WORSE expression model than
the logistic (-3.4 pt); the whole +7.5 pt standalone gain comes from conditioning on
metadata and POSITION. An atlas model conditioned on center_x/center_y/Section ID learns
"at this location in this section, cells tend to be type T" - which is a spatial lookup of
local cell-type composition, i.e. the SAME CHANNEL the newly adopted atlas_composition
block now encodes directly and far more sharply.

So the honest question is not "does it help?" but "does it help ON TOP OF composition?"
If the two are redundant the gain will collapse, and adding 60-120 correlated columns to
a 620-feature stack for nothing is a cost, not a wash.

PRE-REGISTERED RULE (fixed before running). Baseline is the CURRENT deployed stack
(620 features, including atlas-comp and atlas-niche). Three fresh fold partitions
(41/59/83), 20 ET seeds, one out-of-fold prediction per cell so each partition is an
independent 5,000-cell McNemar. The row-shuffled ET block is carried on every partition.

  ADOPT the configuration that satisfies ALL of:
    (a) gain > 0 on all three partitions;
    (b) mean gain > +0.20 pt;
    (c) mean gain exceeds the NULL's mean gain by more than 0.20 pt;
    (d) best such configuration by mean gain.
  Otherwise DO NOT ADOPT.

Identical instrument to iteration9_arbitrate.py, so the two results are comparable.
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
PARTITIONS = (41, 59, 83)
SEEDS = tuple(range(20))
ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
n = len(y)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
COMP = F.atlas_composition(meta_all, CLASSES, k=10)[:n].astype(np.float32)
ANIC = F.atlas_niche(meta_all, list(counts_train.columns), k=50, n_components=30)[:n]
DEPLOYED = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
                      COMP, ANIC.astype(np.float32), c["ATL_TR"]]).astype(np.float32)

blk = np.load(OUT / "atlas_et_block.npz", allow_pickle=True)
ET_FINE = blk["ATL_ET_TR"][:n].astype(np.float32)
ET_COARSE = blk["COARSE_TR"][:n].astype(np.float32)
print(f"deployed stack {DEPLOYED.shape} | ET atlas fine {ET_FINE.shape} "
      f"coarse {ET_COARSE.shape}", flush=True)

rng = np.random.default_rng(9876)
sections = meta_train["Section_ID"].astype(str).to_numpy()
NULL_ET = ET_FINE.copy()
for s in np.unique(sections):
    idx = np.flatnonzero(sections == s)
    NULL_ET[idx] = ET_FINE[rng.permutation(idx)]

CONFIGS = {
    "deployed (620)": DEPLOYED,
    "+ ET atlas fine": np.hstack([DEPLOYED, ET_FINE]),
    "+ ET atlas fine + coarse": np.hstack([DEPLOYED, ET_FINE, ET_COARSE]),
    "+ ET atlas SHUFFLED (null)": np.hstack([DEPLOYED, NULL_ET]),
}
for k, v in CONFIGS.items():
    print(f"  {k:30s} {v.shape[1]} features", flush=True)


def oof_correct(X, seed):
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
    base = correct["deployed (620)"]
    for name, ok in correct.items():
        gain = ok.mean() - base.mean()
        if name == "deployed (620)":
            p_val, w, l = 1.0, 0, 0
        else:
            p_val, _ = M.paired_mcnemar(ok, base)
            w = int((ok & ~base).sum()); l = int((base & ~ok).sum())
        rows.append({"partition": seed, "config": name, "accuracy": ok.mean(),
                     "glia": ok[glia].mean(), "gain_pt": 100 * gain,
                     "wins": w, "losses": l, "p": p_val})
        print(f"  {name:30s} acc={ok.mean():.4f} glia={ok[glia].mean():.4f} "
              f"gain={100*gain:+.2f}pt  {w}w/{l}l  p={p_val:.4g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "arbitrate2.csv", index=False)
summary = df[df.config != "deployed (620)"].groupby("config").agg(
    mean_gain=("gain_pt", "mean"), min_gain=("gain_pt", "min"),
    all_positive=("gain_pt", lambda s: bool((s > 0).all())))
print("\n=== mean gain across the three fresh partitions ===", flush=True)
print(summary.to_string(), flush=True)
null_mean = summary.loc["+ ET atlas SHUFFLED (null)", "mean_gain"]
cands = summary.drop(index="+ ET atlas SHUFFLED (null)")
elig = cands[(cands.all_positive) & (cands.mean_gain > 0.20)
             & (cands.mean_gain - null_mean > 0.20)]
print(f"\n  null mean gain: {null_mean:+.2f} pt", flush=True)
if len(elig):
    w = elig.mean_gain.idxmax()
    print(f"\n  VERDICT: ADOPT '{w}' (mean {elig.loc[w,'mean_gain']:+.2f} pt, "
          f"min {elig.loc[w,'min_gain']:+.2f} pt)", flush=True)
else:
    print("\n  VERDICT: DO NOT ADOPT - redundant with the neighbourhood blocks", flush=True)
print(f"\nwrote {OUT/'arbitrate2.csv'}", flush=True)
