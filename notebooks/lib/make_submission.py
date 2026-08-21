"""Build the competition submission: prediction/prediction.csv.

Self-contained. Rebuilds every feature block from source, fits on ALL 5,000 labelled
challenge cells, predicts the 5,000 test cells, and writes the submission.

WHAT THIS USES
  * data/counts_train.csv, meta_train.csv  - the 200 released genes + metadata + labels
  * data/counts_test.csv, meta_test.csv    - the 200 released genes + metadata
  * data/external/SNI_merged_0531.h5ad     - public reference, DIFFERENT mice/batches,
                                             200 shared genes, cell-type labels only
  * data/external/MERFISH_spinal_cord_0531.h5ad - public parent atlas, restricted to the
                                             200 RELEASED genes, with all 10,000 challenge
                                             cells removed before fitting

WHAT THIS DOES NOT USE
  * none of the 300 genes the organisers withheld
  * no cell-type label of any test cell
  * no other team's predictions

Usage:
    python3 notebooks/lib/make_submission.py              # accepted 694-feature model
    python3 notebooks/lib/make_submission.py no-atlas-et  # pre-Iteration-9 model
    python3 notebooks/lib/make_submission.py no-atlas     # no parent-atlas blocks
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_atlas_model as I9A

USE_ATLAS = "no-atlas" not in sys.argv
USE_ATLAS_ET = USE_ATLAS and "no-atlas-et" not in sys.argv
ALPHA = 0.45
# 20 seeds rather than 5: averaging more seeds of the same estimator is strictly
# variance-reducing (fold sd falls 0.0012 -> 0.0002). Costs compute, cannot hurt.
SEEDS = tuple(range(20))
SUBMISSION = Path("prediction/prediction.csv")
OUT = Path("outputs/merfish_hackathon_iteration5_full_model")
(OUT / "predictions").mkdir(parents=True, exist_ok=True)

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str)
CLASSES = sorted(y.unique())
CLASS_ARR = np.array(CLASSES)
GENES = list(counts_train.columns)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
print(f"train={len(y)} test={len(meta_test)} genes={len(GENES)} classes={len(CLASSES)}",
      flush=True)

# ---------------------------------------------------------------- feature blocks
t0 = time.time()
encoder = OneHotEncoder(handle_unknown="ignore").fit(
    pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str))
BASE_TR = F.base_block(counts_train, meta_train, encoder)
BASE_TE = F.base_block(counts_test, meta_test, encoder)
print(f"[base]    {BASE_TR.shape} ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
(EXT_TR, EXT_TE), REF_X, REF_Y = F.reference_transfer(
    GENES, CLASSES, [counts_train, counts_test], label_column="voting")
assert EXT_TR.shape[1] == 60, "reference transfer must emit 60 class columns"
assert set(REF_Y) <= set(CLASSES), "reference taxonomy does not match the challenge"
print(f"[ext]     SNI reference cells={len(REF_X)} ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
neuron_all = (~meta_all["Region"].isna()).to_numpy() & (meta_all["Region"] == 1).to_numpy()
SPATIAL = F.registered_spatial(meta_all, neuron_all)
SPA_TR, SPA_TE = SPATIAL[: len(meta_train)], SPATIAL[len(meta_train):]
print(f"[spatial] registered {SPATIAL.shape} ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
EXPR_ALL = F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]))
NICHE = F.niche_expression(EXPR_ALL, meta_all, k=15, n_components=30)
NIC_TR, NIC_TE = NICHE[: len(meta_train)], NICHE[len(meta_train):]
print(f"[niche]   {NICHE.shape} ({time.time()-t0:.0f}s)", flush=True)

# Iteration 9: rebuild each cell's real microenvironment from the parent atlas. The
# challenge is a 1-in-27 subsample (70 cells/section vs 964), so the in-file niche block
# above describes neighbours hundreds of microns away. These two blocks restore the true
# neighbourhood through two channels - the neighbours' class histogram, and their mean
# expression. Validated on three fresh fold partitions (41/59/83): +0.54/+0.46/+0.58 pt,
# mean +0.53, against a within-section shuffled null at +0.00 pt.
t0 = time.time()
COMP = F.atlas_composition(meta_all, CLASSES, k=10)
COMP_TR, COMP_TE = COMP[: len(meta_train)], COMP[len(meta_train):]
print(f"[comp]    atlas neighbour composition {COMP.shape} ({time.time()-t0:.0f}s)",
      flush=True)

t0 = time.time()
ANIC = F.atlas_niche(meta_all, GENES, k=50, n_components=30)
ANIC_TR, ANIC_TE = ANIC[: len(meta_train)], ANIC[len(meta_train):]
print(f"[aniche]  atlas neighbour expression {ANIC.shape} ({time.time()-t0:.0f}s)",
      flush=True)

blocks_tr = [BASE_TR, EXT_TR, SPA_TR, NIC_TR, COMP_TR, ANIC_TR]
blocks_te = [BASE_TE, EXT_TE, SPA_TE, NIC_TE, COMP_TE, ANIC_TE]
names = ["base", "ext", "spatial", "niche", "atlas-comp", "atlas-niche"]
if USE_ATLAS:
    t0 = time.time()
    (ATL_TR, ATL_TE), n_ref = F.atlas_transfer(GENES, CLASSES, [counts_train, counts_test])
    blocks_tr.append(ATL_TR); blocks_te.append(ATL_TE); names.append("atlas")
    print(f"[atlas]   parent-atlas cells={n_ref} (challenge cells removed) "
          f"({time.time()-t0:.0f}s)", flush=True)

# Iteration 9: a second, complementary parent-atlas transfer block.  Unlike the
# expression-only logistic above, this ExtraTrees reference model learns interactions
# among the same released 200 genes, QC, mouse, section and position on 136,612 external
# cells.  All 10,000 challenge cells are removed before fitting.  On top of the adopted
# 620-feature neighbourhood stack, fine + coarse atlas probabilities improved three fresh
# fold partitions by +0.74/+0.78/+1.16 pt (mean +0.89), versus +0.19 for the shuffled null.
if USE_ATLAS_ET:
    t0 = time.time()
    if I9A.BLOCK_CACHE.exists():
        atlas_et = np.load(I9A.BLOCK_CACHE, allow_pickle=True)
        if list(atlas_et["classes"].astype(str)) != CLASSES:
            raise ValueError("Iteration-9 atlas cache taxonomy does not match training")
        ATL_ET_TR, ATL_ET_TE = atlas_et["ATL_ET_TR"], atlas_et["ATL_ET_TE"]
        ATL_COARSE_TR, ATL_COARSE_TE = atlas_et["COARSE_TR"], atlas_et["COARSE_TE"]
        print(f"[atlas-et] loaded {I9A.BLOCK_CACHE} ({time.time()-t0:.0f}s)", flush=True)
    else:
        ATL_ET_TR, ATL_ET_TE, ATL_COARSE_TR, ATL_COARSE_TE = I9A.build_block(
            GENES, CLASSES, counts_train, meta_train, counts_test, meta_test
        )
        print(f"[atlas-et] rebuilt from public atlas ({time.time()-t0:.0f}s)", flush=True)
    blocks_tr.append(ATL_ET_TR)
    blocks_te.append(ATL_ET_TE)
    blocks_tr.append(ATL_COARSE_TR)
    blocks_te.append(ATL_COARSE_TE)
    names.extend(["atlas-et", "atlas-et-coarse"])

X_TR = np.hstack(blocks_tr).astype(np.float32)
X_TE = np.hstack(blocks_te).astype(np.float32)
print(f"\nconfig: blocks={'+'.join(names)} alpha={ALPHA} seeds={len(SEEDS)} "
      f"features={X_TR.shape[1]}", flush=True)

# ---------------------------------------------------------------- fit and predict
t0 = time.time()
probs = M.fit_extra_trees(X_TR, y, CLASSES, X_TE, seeds=SEEDS)
probs = M.correct_prior(probs, M.prior_vector(y, CLASSES), ALPHA)

# Hard metadata-compatibility mask. Region, Excitatory_vs_Inhibitory and Segment are each
# a deterministic function of the label on the training cells, so a class never observed
# with a cell's metadata value is impossible. Built from training labels only; CV
# 0.7909 -> 0.7915 (p=6e-05), and it redirects 6 test cells that the model had called
# alpha_motoneuron despite metadata that marks them glial.
MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]
allow = np.ones((len(meta_test), len(CLASSES)), bool)
for col in MASK_COLS:
    tr_vals = meta_train[col].astype(str).to_numpy()
    seen = [set(tr_vals[y.to_numpy() == cls]) for cls in CLASSES]
    known = set(tr_vals)
    for i, v in enumerate(meta_test[col].astype(str).to_numpy()):
        if v in known:
            allow[i] &= np.array([v in s_k for s_k in seen])
allow[~allow.any(1)] = True
n_blocked = int((~allow).sum())
pred = CLASS_ARR[np.where(allow, probs, -1.0).argmax(1)]
n_redirected = int((pred != CLASS_ARR[probs.argmax(1)]).sum())
print(f"metadata mask: {n_blocked} (cell, class) pairs blocked, "
      f"{n_redirected} predictions redirected", flush=True)
print(f"fitted on all {len(y)} labelled cells in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- validate + write
example = pd.read_csv(SUBMISSION, nrows=0)
target_col = example.columns[1]
submission = pd.DataFrame({"Cell_ID": meta_test.index.astype(str), target_col: pred})

assert len(submission) == 5000, f"expected 5000 rows, got {len(submission)}"
assert not submission.Cell_ID.duplicated().any(), "duplicate Cell_IDs"
assert np.array_equal(submission.Cell_ID.to_numpy(),
                      meta_test.index.astype(str).to_numpy()), "row order != meta_test"
assert set(pred) <= set(CLASSES), "predicted a label outside the training taxonomy"
assert submission[target_col].notna().all(), "null prediction"

backup = SUBMISSION.with_suffix(".csv.previous")
if SUBMISSION.exists():
    shutil.copy(SUBMISSION, backup)
text = submission.to_csv(index=False).rstrip("\n")
SUBMISSION.write_text(text)
model_copy = OUT / "predictions" / (
    "prediction_iteration9_atlas_et.csv" if USE_ATLAS_ET else "prediction_iteration5_model.csv"
)
model_copy.write_text(text)

print(f"\nwrote {SUBMISSION}  (previous version saved to {backup})")
print(f"wrote {model_copy}")
print(f"columns: {list(submission.columns)}")
print(f"distinct labels predicted: {submission[target_col].nunique()} / {len(CLASSES)}")
print("\ntop predicted classes:")
print(submission[target_col].value_counts().head(8).to_string())
