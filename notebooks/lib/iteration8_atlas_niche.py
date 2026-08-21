"""Iteration 8d - rebuild the niche block against the FULL atlas, not the subsample.

The niche block currently averages the expression of each cell's 15 nearest neighbours
*within the challenge file*. But §10c established that the challenge is a 1-in-27
subsample: median 35 cells per section in the training file, 70 across train+test,
against 964 in the parent atlas. So the "neighbours" the block averages are hundreds of
microns away and describe almost nothing about the cell's actual microenvironment.

§10c tested neighbour LABEL voting against the full atlas and got 0.2164 - physical
neighbours barely beat chance at predicting a cell's type. That killed label propagation,
and I wrongly treated it as killing everything spatial. It does not: a cell's local
microenvironment is a covariate, not a label to copy. White matter and grey matter have
genuinely different neighbourhood expression profiles even when the neighbours' individual
identities are uninformative, and the oligodendrocyte/astrocyte confusions that dominate
our errors are exactly a white/grey distinction.

So this recomputes the same niche feature with the tissue restored - the mean expression
of each challenge cell's k true physical neighbours among the 136,612 non-challenge atlas
cells in the same section.

LEGITIMACY: uses the parent atlas restricted to the 200 RELEASED genes, plus coordinates.
No withheld gene, no label of any kind, and no challenge cell is used as a neighbour. It
rides on the same atlas dependency already present in the submitted model, which the
ablation prices at +0.46 pt.

PRE-REGISTERED: adopt only if it beats the submitted configuration at p < 0.05 on
identical folds, paired McNemar.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
N_SPLITS, N_REPEATS = 5, 5
KS = (15, 50)
N_PC = 30

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
glia = meta_train["Region"].isna().to_numpy()
GENES = list(counts_train.columns)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
n_tr = len(meta_train)

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
CORE_TR = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["ATL_TR"]]).astype(np.float32)
NIC_OLD = c["NIC_TR"]

# ---------------------------------------------------------------- atlas neighbours
t0 = time.time()
with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    atlas_genes = [g.decode() for g in h["var/_index"][:]]
    lookup = {g: i for i, g in enumerate(atlas_genes)}
    cols = np.array([lookup[g] for g in GENES])
    X_atlas = sparse.csr_matrix(
        (h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
        shape=(len(ids), len(atlas_genes)))
    obs = {k: h[f"obs/{k}"][:] for k in ("center_x", "center_y")
           if f"obs/{k}" in h}
    sec_cat = [s.decode() for s in h["obs/Section ID/categories"][:]] \
        if "obs/Section ID/categories" in h else None
    sec_codes = h["obs/Section ID/codes"][:] if sec_cat else None
print(f"atlas loaded {X_atlas.shape} ({time.time()-t0:.0f}s)", flush=True)
print(f"  coords available: {sorted(obs)} | section categories: "
      f"{'yes' if sec_cat else 'NO'}", flush=True)

pos = {cid: i for i, cid in enumerate(ids)}
challenge_rows = np.array([pos[cid] for cid in meta_all.index.astype(str)])
is_challenge = np.zeros(len(ids), bool)
is_challenge[challenge_rows] = True

atlas_sec = np.array([sec_cat[k] if k >= 0 else "NA" for k in sec_codes])
atlas_xy = np.column_stack([obs["center_x"], obs["center_y"]]).astype(float)
EXPR_ATLAS = None

ch_sec = meta_all["Section_ID"].astype(str).to_numpy()
ch_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
per_sec = pd.Series(atlas_sec[~is_challenge]).value_counts()
print(f"  non-challenge atlas cells per section: median "
      f"{per_sec.median():.0f} (challenge file has "
      f"{pd.Series(ch_sec).value_counts().median():.0f})", flush=True)


def atlas_niche(k):
    """Mean 200-gene log-CPM of each challenge cell's k nearest NON-challenge atlas
    neighbours in its own section."""
    global EXPR_ATLAS
    out = np.zeros((len(ch_sec), len(GENES)), np.float32)
    hit = 0
    for section in np.unique(ch_sec):
        q = np.flatnonzero(ch_sec == section)
        ref = np.flatnonzero((atlas_sec == section) & ~is_challenge)
        if len(ref) < 2:
            continue
        k_eff = min(k, len(ref))
        nn = NearestNeighbors(n_neighbors=k_eff).fit(atlas_xy[ref])
        _, idx = nn.kneighbors(ch_xy[q])
        block = np.asarray(X_atlas[ref][:, cols].todense(), np.float32)
        block = F.log_cpm(block)
        out[q] = block[idx].mean(1)
        hit += len(q)
    print(f"  k={k}: {hit}/{len(ch_sec)} challenge cells matched to atlas neighbours",
          flush=True)
    return PCA(n_components=N_PC, random_state=0).fit_transform(out).astype(np.float32)


NICHE = {}
for k in KS:
    t0 = time.time()
    NICHE[k] = atlas_niche(k)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- evaluate
folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
CONFIGS = {"submitted (challenge-file niche)": NIC_OLD}
for k in KS:
    CONFIGS[f"atlas niche k={k}"] = NICHE[k][:n_tr]
    CONFIGS[f"atlas niche k={k} + old"] = np.hstack([NICHE[k][:n_tr], NIC_OLD])

results = {}
print(f"\n=== {N_SPLITS}x{N_REPEATS} CV, {len(SEEDS)} seeds ===", flush=True)
for name, block in CONFIGS.items():
    t0 = time.time()
    Xf = np.hstack([CORE_TR, block]).astype(np.float32)
    ok = np.zeros((N_REPEATS, n_tr), bool)
    for f, (tr, va) in enumerate(folds):
        p = M.fit_extra_trees(Xf[tr], pd.Series(y[tr]), CLASSES, Xf[va], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
        ok[f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    results[name] = ok
    a = ok.mean(1)
    print(f"  {name:34s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[:, glia].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

base = results["submitted (challenge-file niche)"]
rows = []
for name, ok in results.items():
    if name == "submitted (challenge-file niche)":
        continue
    p_val, _ = M.paired_mcnemar(ok.ravel(), base.ravel())
    rows.append({"variant": name, "accuracy": ok.mean(), "glia": ok[:, glia].mean(),
                 "gain": ok.mean() - base.mean(), "p": p_val})
rows.sort(key=lambda r: r["p"])
m = len(rows)
print("\n=== paired McNemar vs submitted (Holm-corrected) ===", flush=True)
for i, r in enumerate(rows):
    r["holm"] = 0.05 / (m - i)
    r["passes"] = bool(r["gain"] > 0 and r["p"] < r["holm"])
    print(f"  {r['variant']:34s} {r['gain']:+.4f} p={r['p']:.3g} (Holm {r['holm']:.4f})"
          f"{'   <== PASSES' if r['passes'] else ''}", flush=True)
print(f"\n  VERDICT: {'ADOPT' if any(r['passes'] for r in rows) else 'DO NOT ADOPT'}",
      flush=True)

np.savez_compressed(OUT / "atlas_niche.npz", **{f"k{k}": NICHE[k] for k in KS})
pd.DataFrame(rows).to_csv(OUT / "atlas_niche.csv", index=False)
print(f"\nwrote {OUT/'atlas_niche.csv'} and the cached blocks", flush=True)
