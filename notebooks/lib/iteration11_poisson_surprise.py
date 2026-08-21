"""Iteration 11b - per-gene Poisson surprise against expected neighbour leak.

WHY THE SUBTRACTION VERSION FAILED (iteration11_decontaminate.py)
-----------------------------------------------------------------
    partition 101   rho=0.15 -0.24 pt   rho=0.30 -0.20 pt   wrong-section null -0.04 pt
    partition 137   rho=0.15 -0.08 pt   rho=0.30 +0.12 pt   wrong-section null +0.22 pt

The null matched or beat the real profile on both, and the diagnostic says why: at
rho=0.15 the decontaminated matrix is 92.6% zeros and at rho=0.30 it is 92.7% - virtually
identical, because a 15-30% leak on a 21-transcript cell is ~3 counts spread over 200
genes, so per-gene corrections are 0.1-0.3 of a count. Subtraction barely moved the
representation, so the test could not discriminate real neighbours from wrong ones.

THE RIGHT STATISTIC IS A RATIO, NOT A DIFFERENCE
-------------------------------------------------
Whether an observed count is evidence of intrinsic expression depends entirely on how
much leak was EXPECTED at that gene:

    x = 1, expected leak 0.05  ->  20x more than leak explains. Strong evidence.
    x = 1, expected leak 0.90  ->  entirely consistent with spillover. No evidence.

Subtraction collapses both to ~0.5 and throws the distinction away. A ratio keeps it as a
20-fold difference. This matters precisely because MERFISH counts are tiny: at depth 21
almost every informative gene sits at 1-3 counts, which is the regime where "is this more
than leakage would produce?" is the entire question.

The principled form is the Poisson upper-tail probability under a leak-only null:

    e_ig       = rho * depth_i * p_ig            expected leak counts at gene g
    surprise   = -log P(X >= x_ig | Poisson(e_ig))

x = 0 gives P = 1 and surprise 0, correctly encoding "no evidence either way" rather than
a spurious negative. Large counts at genes with negligible expected leak give large
surprise. This is a per-gene likelihood-ratio test for intrinsic expression, and it is
exactly the statistic the ResolVI generative model implies
(observed = true + weighted neighbour leak + background).

Also tested: the simpler log-ratio log((x + 0.5) / (e + 0.5)), which keeps the ratio
behaviour without the Poisson tail, as a check on whether the tail matters or just the
ratio.

NULL CONTROL. Identical construction with p_ig drawn from a DIFFERENT randomly chosen
section - same width, same non-linearity, same dependence on depth, destroying only the
fact that the neighbours are this cell's real ones. Carried on every partition.

PRE-REGISTERED RULE (fixed before any number exists)
----------------------------------------------------
Three fresh partitions (211 is taken; using 197/229/263), 20 ET seeds, one out-of-fold
prediction per cell, same instrument as iteration9_arbitrate.py.

  ADOPT the configuration satisfying ALL of:
    (a) gain > 0 on all three partitions;
    (b) mean gain > +0.20 pt;
    (c) mean gain exceeds the NULL's mean gain by more than 0.20 pt;
    (d) best such configuration by mean gain.
  Otherwise DO NOT ADOPT.

LEGITIMACY. Parent atlas restricted to non-challenge cells, 200 released genes only. No
challenge or test label, no withheld gene. Same standing as the deployed atlas blocks.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import poisson
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration11")
OUT.mkdir(parents=True, exist_ok=True)
PARTITIONS = (197, 229, 263)
SEEDS = tuple(range(20))
K_NBR = 10
RHO = 0.30
ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
GENES = list(counts_train.columns)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
n = len(y)

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
assert len(donors) == len(ids) - len(meta_all)
q_sec = meta_all["Section_ID"].astype(str).to_numpy()
q_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
print(f"atlas loaded, {len(donors)} donors ({time.time()-t0:.0f}s)", flush=True)


def neighbour_profile(section_map):
    out = np.zeros((len(q_sec), len(GENES)), np.float32)
    donor_sec = atlas_sec[donors]
    for section in np.unique(q_sec):
        rows = np.flatnonzero(q_sec == section)
        pool = donors[donor_sec == section_map[section]]
        if len(pool) < 5 or len(rows) == 0:
            continue
        tree = cKDTree(np.column_stack([ax[pool], ay[pool]]))
        dist, nn = tree.query(q_xy[rows], k=min(K_NBR, len(pool)))
        dist = np.atleast_2d(dist); nn = np.atleast_2d(nn)
        w = 1.0 / (dist + 1e-6)
        w /= w.sum(1, keepdims=True)
        block = np.asarray(X_atlas[pool][:, cols].todense(), np.float32)
        out[rows] = np.einsum("nk,nkg->ng", w.astype(np.float32), block[nn])
    total = out.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return out / total


sections = np.unique(q_sec)
ident = {s: s for s in sections}
rng = np.random.default_rng(777)
perm = rng.permutation(sections)
wrong = {s: perm[i] for i, s in enumerate(sections)}

t0 = time.time()
P_REAL, P_NULL = neighbour_profile(ident), neighbour_profile(wrong)
RAW = np.vstack([counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32)])
DEPTH = RAW.sum(1, keepdims=True)
print(f"profiles built ({time.time()-t0:.0f}s); median depth {np.median(DEPTH):.0f}",
      flush=True)


def surprise(profile):
    """-log P(X >= x | Poisson(expected leak)). Zero counts give exactly 0."""
    e = np.clip(RHO * DEPTH * profile, 1e-9, None)
    sf = poisson.sf(RAW - 1, e)                 # P(X >= x)
    out = -np.log(np.clip(sf, 1e-12, 1.0))
    out[RAW == 0] = 0.0
    return out.astype(np.float32)[:n]


def logratio(profile):
    e = RHO * DEPTH * profile
    return np.log((RAW + 0.5) / (e + 0.5)).astype(np.float32)[:n]


t0 = time.time()
S_REAL, S_NULL, L_REAL = surprise(P_REAL), surprise(P_NULL), logratio(P_REAL)
print(f"statistics built ({time.time()-t0:.0f}s)", flush=True)
nz = RAW[:n] > 0
print(f"  surprise on non-zero entries: mean {S_REAL[nz].mean():.2f} "
      f"p90 {np.percentile(S_REAL[nz], 90):.2f} max {S_REAL[nz].max():.2f}", flush=True)
print(f"  null surprise, same entries : mean {S_NULL[nz].mean():.2f} "
      f"p90 {np.percentile(S_NULL[nz], 90):.2f}", flush=True)
print(f"  corr(real, null) over non-zero entries: "
      f"{np.corrcoef(S_REAL[nz], S_NULL[nz])[0,1]:.3f}", flush=True)

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
COMP = F.atlas_composition(meta_all, CLASSES, k=10)[:n].astype(np.float32)
ANIC = F.atlas_niche(meta_all, GENES, k=50, n_components=30)[:n].astype(np.float32)
blk = np.load("outputs/iteration9/atlas_et_block.npz", allow_pickle=True)
DEPLOYED = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], COMP, ANIC,
                      c["ATL_TR"], blk["ATL_ET_TR"][:n],
                      blk["COARSE_TR"][:n]]).astype(np.float32)

CONFIGS = {
    "deployed (694)": DEPLOYED,
    "+ Poisson surprise": np.hstack([DEPLOYED, S_REAL]),
    "+ log-ratio": np.hstack([DEPLOYED, L_REAL]),
    "+ WRONG-SECTION surprise (null)": np.hstack([DEPLOYED, S_NULL]),
}
for k, v in CONFIGS.items():
    print(f"  {k:34s} {v.shape[1]} features", flush=True)


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
                     "glia": ok[glia].mean(), "gain_pt": 100 * gain,
                     "wins": w, "losses": l, "p": p_val})
        print(f"  {name:34s} acc={ok.mean():.4f} glia={ok[glia].mean():.4f} "
              f"gain={100*gain:+.2f}pt  {w}w/{l}l  p={p_val:.4g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "poisson_surprise.csv", index=False)
summary = df[df.config != "deployed (694)"].groupby("config").agg(
    mean_gain=("gain_pt", "mean"), min_gain=("gain_pt", "min"),
    all_positive=("gain_pt", lambda s: bool((s > 0).all())))
print("\n=== mean gain across three fresh partitions ===", flush=True)
print(summary.to_string(), flush=True)
null_mean = summary.loc["+ WRONG-SECTION surprise (null)", "mean_gain"]
cands = summary.drop(index="+ WRONG-SECTION surprise (null)")
elig = cands[(cands.all_positive) & (cands.mean_gain > 0.20)
             & (cands.mean_gain - null_mean > 0.20)]
print(f"\n  null mean gain: {null_mean:+.2f} pt", flush=True)
if len(elig):
    w = elig.mean_gain.idxmax()
    print(f"\n  VERDICT: ADOPT '{w}' (mean {elig.loc[w,'mean_gain']:+.2f} pt)", flush=True)
else:
    print("\n  VERDICT: DO NOT ADOPT", flush=True)
print(f"\nwrote {OUT/'poisson_surprise.csv'}", flush=True)
