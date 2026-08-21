"""Iteration 11 - subtract neighbour spillover from each cell's counts.

LITERATURE MOTIVATION
---------------------
Imaging-based spatial transcriptomics misassigns transcripts across cell boundaries
during segmentation. The contamination is real, frequent, and strongly distance-
dependent (MisTIC, biorxiv 2025.12.11.693759). ResolVI (biorxiv 2025.01.20.634005)
writes the generative model explicitly:

    observed_i  =  true_i  +  SUM_j w_ij * true_j  +  background

i.e. a cell's observed profile is its own expression plus a distance-weighted leak from
its physical neighbours.

WHY THIS DATASET IS THE WORST CASE, AND THE BEST OPPORTUNITY
------------------------------------------------------------
Median 21 transcripts per cell. In tissue that dense, a leak of even 20% is ~4 of the
21 counts a classifier sees. The cells worst affected are small ones wedged among
larger neighbours - which is exactly the glia, holding 917 of our 1,108 errors. And the
released panel is a NEURON panel (35 protocadherins, neuropeptides, GPCRs), so a glial
cell sitting next to a neuron picks up counts on precisely the genes that define
neuronal identity, with nothing of its own to compete.

WHY THIS IS NOT THE NICHE OR COMPOSITION BLOCK AGAIN
----------------------------------------------------
Those blocks ADD the neighbourhood as extra columns. This SUBTRACTS it from the cell's
own profile. The distinction is not cosmetic for our model: ExtraTrees split on single
axes, so `x_i - lambda * c_i` is not expressible from `x_i` and `c_i` as separate
features without an impractical number of splits. Supplying the residual directly is a
real gain in representational reach for a tree ensemble, not a restatement.

We can also do this better than a normal pipeline could. Ordinary decontamination must
GUESS a cell's neighbours from the same noisy file. The challenge is a 1-in-27 subsample
of a public atlas, so we know each cell's ACTUAL physical neighbours - all 26 that were
withheld - and can read their expression directly.

THE ESTIMATOR
-------------
For challenge cell i with observed counts x_i and depth d_i = sum(x_i):

    p_i   = distance-weighted mean of the k nearest NON-challenge atlas neighbours'
            count profiles, in the same section, normalised to sum to 1
    x~_i  = clip(x_i - rho * d_i * p_i, 0, None)

rho is the assumed contaminated fraction of a cell's depth. The block is log1p(x~_i),
200 columns, appended to the deployed 694-feature stack.

NULL CONTROL. The identical construction with p_i taken from a DIFFERENT randomly chosen
section. It has the same width, the same sparsity, the same subtract-and-clip
non-linearity, and destroys only the fact that the neighbours are the cell's real ones.
If the mechanism is spillover, the real profile must beat this; if the gain is just
"subtracting something roughly transcriptome-shaped sharpens the counts", it will not.

PRE-REGISTERED RULE (fixed before any number exists)
----------------------------------------------------
Three fresh fold partitions (101/137/173), never used by any earlier experiment
(7, 23, 41, 59, 83, 131, 211, 251). 20 ET seeds. One out-of-fold prediction per cell, so
each partition is an independent 5,000-cell McNemar. Same instrument as
iteration9_arbitrate.py so results are directly comparable.

  ADOPT the configuration satisfying ALL of:
    (a) gain > 0 on all three partitions;
    (b) mean gain > +0.20 pt;
    (c) mean gain exceeds the NULL's mean gain by more than 0.20 pt;
    (d) best such configuration by mean gain.
  Otherwise DO NOT ADOPT.

LEGITIMACY
----------
Reads the parent atlas restricted to cells that are NOT in the challenge (all 10,000
removed before any neighbour search), over the 200 RELEASED genes only. No challenge or
test label is read, and no withheld gene is read for any cell. Same standing as the
atlas blocks already deployed.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration11")
OUT.mkdir(parents=True, exist_ok=True)
PARTITIONS = (101, 137, 173)
SEEDS = tuple(range(20))
K_NBR = 10
RHOS = (0.15, 0.30)
ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
GENES = list(counts_train.columns)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
n = len(y)

# ---------------------------------------------------------------- neighbour profiles
t0 = time.time()
with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    atlas_genes = [g.decode() for g in h["var/_index"][:]]
    cols = np.array([{g: i for i, g in enumerate(atlas_genes)}[g] for g in GENES])
    X_atlas = sparse.csr_matrix(
        (h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
        shape=(len(ids), len(atlas_genes)))
    sec_cat = [c.decode() for c in h["obs/Section ID/categories"][:]]
    sec_codes = h["obs/Section ID/codes"][:]
    ax = h["obs/center_x"][:].astype(float)
    ay = h["obs/center_y"][:].astype(float)

atlas_sec = np.array([sec_cat[c] if c >= 0 else "NA" for c in sec_codes])
position = {c: i for i, c in enumerate(ids)}
is_challenge = np.zeros(len(ids), bool)
is_challenge[[position[c] for c in meta_all.index.astype(str)]] = True
donors = np.flatnonzero(~is_challenge)
assert len(donors) == len(ids) - len(meta_all), "donor pool contains challenge cells"
print(f"atlas {X_atlas.shape}, donors {len(donors)} ({time.time()-t0:.0f}s)", flush=True)

q_sec = meta_all["Section_ID"].astype(str).to_numpy()
q_xy = meta_all[["center_x", "center_y"]].to_numpy(float)


def neighbour_profile(section_map):
    """Distance-weighted mean neighbour count profile, normalised to sum 1 per cell.

    `section_map` sends each challenge section to the donor section to draw from - the
    identity for the real block, a permutation for the null.
    """
    out = np.zeros((len(q_sec), len(GENES)), np.float32)
    donor_sec = atlas_sec[donors]
    for section in np.unique(q_sec):
        rows = np.flatnonzero(q_sec == section)
        pool = donors[donor_sec == section_map[section]]
        if len(pool) < 5 or len(rows) == 0:
            continue
        tree = cKDTree(np.column_stack([ax[pool], ay[pool]]))
        k_eff = min(K_NBR, len(pool))
        dist, nn = tree.query(q_xy[rows], k=k_eff)
        dist = np.atleast_2d(dist); nn = np.atleast_2d(nn)
        w = 1.0 / (dist + 1e-6)
        w /= w.sum(1, keepdims=True)
        block = np.asarray(X_atlas[pool][:, cols].todense(), np.float32)
        out[rows] = np.einsum("nk,nkg->ng", w.astype(np.float32), block[nn])
    total = out.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return out / total


ident = {s: s for s in np.unique(q_sec)}
rng = np.random.default_rng(4242)
shuffled_sections = rng.permutation(np.unique(q_sec))
wrong = {s: shuffled_sections[i] for i, s in enumerate(np.unique(q_sec))}

t0 = time.time()
P_REAL = neighbour_profile(ident)
P_NULL = neighbour_profile(wrong)
print(f"neighbour profiles built ({time.time()-t0:.0f}s)", flush=True)

RAW = np.vstack([counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32)])
DEPTH = RAW.sum(1, keepdims=True)
print(f"median depth {np.median(DEPTH):.0f} transcripts/cell", flush=True)


def decontaminated(profile, rho):
    return np.log1p(np.clip(RAW - rho * DEPTH * profile, 0, None)).astype(np.float32)[:n]


# ---------------------------------------------------------------- deployed stack
c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
COMP = F.atlas_composition(meta_all, CLASSES, k=10)[:n].astype(np.float32)
ANIC = F.atlas_niche(meta_all, GENES, k=50, n_components=30)[:n].astype(np.float32)
blk = np.load("outputs/iteration9/atlas_et_block.npz", allow_pickle=True)
DEPLOYED = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], COMP, ANIC,
                      c["ATL_TR"], blk["ATL_ET_TR"][:n],
                      blk["COARSE_TR"][:n]]).astype(np.float32)
print(f"deployed stack {DEPLOYED.shape}", flush=True)

CONFIGS = {"deployed (694)": DEPLOYED}
for rho in RHOS:
    CONFIGS[f"+ decontaminated rho={rho}"] = np.hstack([DEPLOYED, decontaminated(P_REAL, rho)])
CONFIGS[f"+ WRONG-SECTION rho={RHOS[-1]} (null)"] = np.hstack(
    [DEPLOYED, decontaminated(P_NULL, RHOS[-1])])
for k, v in CONFIGS.items():
    print(f"  {k:38s} {v.shape[1]} features", flush=True)

# how much do the counts actually move?
for rho in RHOS:
    d = decontaminated(P_REAL, rho)
    zeroed = (np.clip(RAW[:n] - rho * DEPTH[:n] * P_REAL[:n], 0, None) == 0).mean()
    print(f"  rho={rho}: {100*zeroed:.1f}% of count entries driven to zero", flush=True)


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
    base = correct["deployed (694)"]
    for name, ok in correct.items():
        gain = ok.mean() - base.mean()
        if name == "deployed (694)":
            p_val, w, l = 1.0, 0, 0
        else:
            p_val, _ = M.paired_mcnemar(ok, base)
            w = int((ok & ~base).sum()); l = int((base & ~ok).sum())
        rows.append({"partition": seed, "config": name, "accuracy": ok.mean(),
                     "glia": ok[glia].mean(), "neurons": ok[~glia].mean(),
                     "gain_pt": 100 * gain, "wins": w, "losses": l, "p": p_val})
        print(f"  {name:38s} acc={ok.mean():.4f} glia={ok[glia].mean():.4f} "
              f"gain={100*gain:+.2f}pt  {w}w/{l}l  p={p_val:.4g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "decontaminate.csv", index=False)
summary = df[df.config != "deployed (694)"].groupby("config").agg(
    mean_gain=("gain_pt", "mean"), min_gain=("gain_pt", "min"),
    all_positive=("gain_pt", lambda s: bool((s > 0).all())),
    mean_glia=("glia", "mean"))
print("\n=== mean gain across three fresh partitions ===", flush=True)
print(summary.to_string(), flush=True)
null_name = [k for k in CONFIGS if "null" in k][0]
null_mean = summary.loc[null_name, "mean_gain"]
cands = summary.drop(index=null_name)
elig = cands[(cands.all_positive) & (cands.mean_gain > 0.20)
             & (cands.mean_gain - null_mean > 0.20)]
print(f"\n  null mean gain: {null_mean:+.2f} pt", flush=True)
if len(elig):
    w = elig.mean_gain.idxmax()
    print(f"\n  VERDICT: ADOPT '{w}' (mean {elig.loc[w,'mean_gain']:+.2f} pt, "
          f"min {elig.loc[w,'min_gain']:+.2f} pt)", flush=True)
else:
    print("\n  VERDICT: DO NOT ADOPT", flush=True)
print(f"\nwrote {OUT/'decontaminate.csv'}", flush=True)
