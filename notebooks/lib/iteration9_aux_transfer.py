"""Iteration 9 - the parent atlas's AUXILIARY ANNOTATION COLUMNS as standalone
transfer targets, and the (1st round cluster, 2nd round subcluster) refinement.

-------------------------------------------------------------------------------
MECHANISM
-------------------------------------------------------------------------------
`atlas_transfer` fits ONE L2 multinomial logistic per cell type on 136,612 atlas
cells over the 200 released genes. A cell type that is transcriptionally
MULTIMODAL therefore gets a single linear weight vector covering several distinct
subpopulations, and the softmax is forced onto their mean direction.

The atlas ships its own two-level clustering. Keyed jointly, (1st round cluster,
2nd round subcluster) yields 81 groups that STRICTLY REFINE the 60-class target:
H(cell_type | joint) = 0.014 bits, and only 1 of the 81 groups is impure. The
refinement lands on exactly 11 cell types - six of them glial, the error
bottleneck: endothelial splits into 6 modes, oligodendrocyte-progenitor-2 into 4,
OPC-1 into 3, Schwann cell / pericyte / peripheral glia into 2 each, plus five
dorsal-horn neuron types.

Fitting the SAME logistic at 81-way granularity gives every mode its own linear
discriminant, then pooling the 81 posteriors back to 60 columns yields a
MIXTURE-OF-LINEAR-EXPERTS readout for precisely the multimodal glial classes -
same information, same legitimacy footprint, a strictly more expressive decision
surface than one hyperplane per class.

-------------------------------------------------------------------------------
PRIOR EVIDENCE (diagnostics run before this script; numbers are load-bearing)
-------------------------------------------------------------------------------
1. Six aux columns are EXACT DETERMINISTIC COARSENINGS of the target, measured
   over all 146,621 atlas cells - H(aux | cell_type) in bits:

       1st round cluster        0.000     (0 of 60 cell types split)
       Neurotransmitter         0.000     (0 of 60)
       Region                   0.000     (0 of 60)
       Excitatory_vs_Inhibitory 0.000     (0 of 60)
       Laminae                  0.006     (1 of 60: DH_in_Cdh3)
       Markers                  0.105     (1 of 60: endothelial)

   They are label metadata attached post-hoc to the 60 clusters, NOT independent
   measurements that cross-cut the taxonomy. A transfer model predicting one of
   them is an exact marginalisation of the 60-way posterior - a rank-deficient
   linear projection carrying zero new information. This is the mathematical
   reason SCORECARD 7d saw -0.30 pt when Laminae/Markers/Neurotransmitter were
   added, and it is why NONE of them gets an arm here.

2. Axial level / Datasets / Gender / Mouse ID are near-INDEPENDENT of cell type
   (MI/H(y) = 0.005 / 0.008 / 0.001 / 0.006) and all four are already observed
   columns of the challenge metadata. Nothing to transfer.

3. The subclusters ARE learnable on the 200 released genes - within-cell-type
   70/30 holdout accuracy vs the majority-subcluster rate:

       pericyte            1.000 vs 0.526      peripheral glia  0.977 vs 0.650
       DH_in_Rorb          0.954 vs 0.596      OPC-1            0.680 vs 0.415
       DH_in_Pdyn          0.738 vs 0.580      endothelial      0.508 vs 0.310

   So the finer granularity is real and visible on the released panel. The
   premise of this angle survives step 3; what follows is where it is tested.

4. NEGATIVE screen already run (3 x 50/50 splits, 2 ET seeds, deltas vs the
   submitted 529-feature stack at 0.7714):

       replace ATL60 with the raw 82 subcluster columns   -0.34 pt
       ATL60 + raw 82 subcluster columns                  -0.12 pt
       replace ATL60 with MAX-pooled 60                   +0.06 pt
       replace ATL60 with SUM-pooled 60                   +0.08 pt
       ATL60 + MAX-pooled 60                              +0.17 pt

   And standalone (no ET, argmax of the transfer block alone, challenge-train):
   81-way pooled back to 60 = 0.5784 vs 60-way = 0.5824; on the 11 refined cell
   types themselves 0.6216 vs 0.6322. Argmax agreement 95.5%, mean |dP| 0.0011.

5. THE NULL CONTROL ALREADY FIRES AT THE STANDALONE LEVEL. Fitting all three
   transfer blocks on the full 136,612-cell atlas reference and taking the argmax
   directly on the 5,000 challenge training cells:

       plain 60-way ATL block                           0.6040
       81-way subclusters, SUM-pooled back to 60        0.6016
       81-way subclusters, MAX-pooled back to 60        0.5992
       RANDOM refinement of matched shape, MAX-pooled   0.5972

   The atlas's biological subcluster structure buys 0.2-0.4 pt over a random
   shattering of each cell type into groups of the same sizes - and BOTH lose to
   not refining at all. Whatever the refinement contributes is nearly exhausted by
   the mere act of splitting classes, which is the definition of a null result.

   The prior expectation is therefore NEGATIVE. SCORECARD 8c established that the
   glial error is a GROSS-IDENTITY error - 22% of glia land in the wrong broad
   cluster, while within-cluster accuracy is already 89.2%. Subclustering refines
   BELOW the level where the error lives, which is the same failure mode as the
   glia specialist, pairwise arbitration and stacking. This script exists to give
   the one arm that screened positive (+0.17 pt, below the ~0.3 pt measurability
   floor) a full-power test rather than to ratify it.

-------------------------------------------------------------------------------
ARMS
-------------------------------------------------------------------------------
   A  submitted 529-feature stack                                (baseline)
   C  replace the 60-col ATL block with MAX-pooled subcluster 60  (529 wide)
   F  ATL60 + MAX-pooled subcluster 60                            (589 wide)
   G  ATL60 + a separately fitted 14-way '1st round cluster' block (543 wide)
   N  ATL60 + MAX-pooled 60 from a RANDOM refinement              (NULL CONTROL)

   Arm G is the single aux column with a real prior-evidence story: SCORECARD 8c
   showed coarse-cluster assignment IS the failing step, and a separately fitted
   14-way logistic is not numerically identical to marginalising the 60-way one
   (different regularisation path, far better conditioned per class). It is still
   an exact coarsening, so it is expected to fail; it is included so the angle is
   tested rather than only argued about.

   Arm N is the null control and the point of the whole script. It shatters each
   cell type into random subgroups with the SAME group-size distribution as the
   real refinement, then runs the identical pool-and-append pipeline. If N moves
   accuracy as much as C/F, the gain is from feature-block width and ET column
   subsampling, not from the atlas's biological subcluster structure.

-------------------------------------------------------------------------------
PRE-REGISTERED DECISION RULE  (fixed before the run; do not renegotiate after)
-------------------------------------------------------------------------------
   5 x 5 RepeatedStratifiedKFold, fold seed 23, 20 ET seeds, alpha = 0.45.
   Four candidate arms (C, F, G, N) vs A, paired exact McNemar on identical
   folds, HOLM correction over the 4 comparisons.

   ADOPT an arm iff ALL FOUR hold:
     (i)   gain > +0.0030 absolute  (the stated measurability floor; anything
           smaller is fold noise, since fold assignment alone moves CV +/-0.3 pt)
     (ii)  Holm-adjusted p < 0.05
     (iii) the NULL arm N does NOT itself satisfy (i) and (ii)
     (iv)  the arm REPLICATES on fold seed 101 with the same sign and gain
           > +0.0030 (run with --replicate; a second partition is mandatory
           because SCORECARD 10p-1 documents a candidate that passed at p=0.034
           and then reversed sign on a new partition)

   Otherwise DO NOT ADOPT. Recording a clean negative is the expected outcome and
   is a complete result.

-------------------------------------------------------------------------------
LEGITIMACY
-------------------------------------------------------------------------------
   * Only the 200 RELEASED genes are ever read, for atlas and challenge cells
     alike. The 300 withheld genes are never touched.
   * All 10,000 challenge cells (train and test) are removed from the atlas
     before any transfer model is fitted - identical exclusion to atlas_transfer.
   * The atlas annotation columns are read for ATLAS cells only. No challenge
     cell's label, in any column, is read as a feature.
   * Challenge test labels are never read. Nothing under outputs/quarantine/ is
     opened. prediction/prediction.csv and make_submission.py are not modified.

Usage:  python notebooks/lib/iteration9_aux_transfer.py [--replicate]
"""
import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "aux_transfer_blocks.npz"
FEATURE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")

SEEDS = tuple(range(20))
ALPHA = 0.45
FOLD_SEED = 101 if "--replicate" in sys.argv else 23
MIN_GAIN = 0.0030


# ----------------------------------------------------------------------
# Atlas read: released genes only, challenge cells excluded
# ----------------------------------------------------------------------
def read_atlas(gene_order, challenge_ids):
    """Return (expression over released genes, cell type, joint subcluster key,
    1st-round cluster) for every atlas cell that is NOT a challenge cell."""
    with h5py.File(F.PARENT_ATLAS, "r") as h:
        ids = np.array([x.decode() for x in h["obs/_index"][:]])
        atlas_genes = [g.decode() for g in h["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        columns = np.array([lookup[g] for g in gene_order])   # 200 released genes only
        matrix = sparse.csr_matrix(
            (h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )

        def column(key):
            cats = np.array([c.decode() for c in h[f"obs/{key}/categories"][:]])
            codes = h[f"obs/{key}/codes"][:]
            return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "NA")

        cell_type = np.array([F._normalise_label(v)
                              for v in column("MERFISH cell type annotation")])
        first = column("1st round cluster")
        second = column("2nd round subcluster")

    # Append the cell type to the joint key so every group is pure by construction
    # (1 of 81 groups is otherwise impure) and pooling back to 60 is exact.
    joint = np.array([f"{a}|{b}|{c}" for a, b, c in zip(first, second, cell_type)])

    position = {c: i for i, c in enumerate(ids)}
    challenge = np.zeros(len(ids), bool)
    challenge[[position[c] for c in challenge_ids]] = True
    keep = ~challenge

    expression = np.asarray(matrix[np.flatnonzero(keep)][:, columns].todense(), np.float32)
    return expression, cell_type[keep], joint[keep], first[keep]


def transfer_block(reference, labels, matrices, C=0.1):
    """L2 logistic on the 200 released genes, atlas cells only - mirrors
    iteration5_features.atlas_transfer, with an arbitrary target column."""
    model = LogisticRegression(C=C, max_iter=1500, n_jobs=2)
    model.fit(F.zscore(F.log_cpm(reference)), labels)
    order = list(model.classes_)
    blocks = [model.predict_proba(F.zscore(F.log_cpm(m))).astype(np.float32)
              for m in matrices]
    return blocks, order


def pool(block, order, classes, how="max"):
    """Collapse a refined transfer block back onto the 60 target columns."""
    index = {c: i for i, c in enumerate(classes)}
    groups = [[] for _ in classes]
    for j, name in enumerate(order):
        groups[index[name.split("|")[-1]]].append(j)
    reduce = np.max if how == "max" else np.sum
    out = np.zeros((len(block), len(classes)), np.float32)
    for i, g in enumerate(groups):
        if g:
            out[:, i] = reduce(block[:, g], axis=1)
    return out


def random_refinement(cell_type, joint, rng):
    """NULL CONTROL: shatter each cell type into random subgroups matching the
    real refinement's group-size distribution, ignoring biology entirely."""
    fake = np.empty(len(cell_type), object)
    for label in np.unique(cell_type):
        rows = np.flatnonzero(cell_type == label)
        sizes = pd.Series(joint[rows]).value_counts().to_numpy()
        shuffled = rng.permutation(rows)
        cut = np.cumsum(sizes)[:-1]
        for k, chunk in enumerate(np.split(shuffled, cut)):
            fake[chunk] = f"R{k}|{label}"
    return fake.astype(str)


# ----------------------------------------------------------------------
def build_blocks():
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    gene_order = list(counts_train.columns)
    classes = sorted(meta_train[F.TARGET].astype(str).unique())
    challenge_ids = list(meta_train.index.astype(str)) + list(meta_test.index.astype(str))

    expression, cell_type, joint, first = read_atlas(gene_order, challenge_ids)
    usable = (expression.sum(1) > 0) & np.isin(cell_type, classes)
    expression, cell_type = expression[usable], cell_type[usable]
    joint, first = joint[usable], first[usable]
    print(f"atlas reference {expression.shape}, {len(set(joint))} joint subclusters, "
          f"{len(set(first))} coarse clusters", flush=True)

    matrices = [counts_train.to_numpy(), counts_test.to_numpy()]
    store = {}

    t0 = time.time()
    (sub_tr, sub_te), sub_order = transfer_block(expression, joint, matrices)
    store["SUB_TR"], store["SUB_TE"] = sub_tr, sub_te
    store["SUB_ORDER"] = np.array(sub_order)
    print(f"  subcluster block {sub_tr.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    (c14_tr, c14_te), c14_order = transfer_block(expression, first, matrices)
    store["C14_TR"], store["C14_TE"] = c14_tr, c14_te
    store["C14_ORDER"] = np.array(c14_order)
    print(f"  coarse-cluster block {c14_tr.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    fake = random_refinement(cell_type, joint, np.random.default_rng(0))
    (nul_tr, nul_te), nul_order = transfer_block(expression, fake, matrices)
    store["NUL_TR"], store["NUL_TE"] = nul_tr, nul_te
    store["NUL_ORDER"] = np.array(nul_order)
    print(f"  null-refinement block {nul_tr.shape} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(CACHE, **store)
    return store, classes


def main():
    counts_train, meta_train, _, _ = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.array(classes)
    glia = meta_train["Region"].isna().to_numpy()

    if CACHE.exists():
        store = dict(np.load(CACHE, allow_pickle=True))
        print(f"loaded cached blocks from {CACHE}", flush=True)
    else:
        store, classes = build_blocks()

    cache = np.load(FEATURE_CACHE, allow_pickle=True)
    pre = np.hstack([cache["BASE_TR"], cache["EXT_TR"],
                     cache["SPA_TR"], cache["NIC_TR"]]).astype(np.float32)
    atl = cache["ATL_TR"].astype(np.float32)

    sub_max = pool(store["SUB_TR"], [str(s) for s in store["SUB_ORDER"]], classes, "max")
    nul_max = pool(store["NUL_TR"], [str(s) for s in store["NUL_ORDER"]], classes, "max")

    arms = {
        "A submitted 529":      np.hstack([pre, atl]),
        "C replace MAX60":      np.hstack([pre, sub_max]),
        "F ATL60 + MAX60":      np.hstack([pre, atl, sub_max]),
        "G ATL60 + cluster14":  np.hstack([pre, atl, store["C14_TR"]]),
        "N NULL refinement":    np.hstack([pre, atl, nul_max]),
    }
    folds = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=FOLD_SEED).split(y, y))
    print(f"\nfold seed {FOLD_SEED}, {len(folds)} folds, {len(SEEDS)} ET seeds, "
          f"alpha={ALPHA}", flush=True)
    for name, X in arms.items():
        print(f"  {name:22s} {X.shape}", flush=True)

    def run(X, tag):
        t0 = time.time()
        ok = np.zeros((5, len(y)), bool)
        for f, (tr, va) in enumerate(folds):
            probs = M.fit_extra_trees(X[tr], pd.Series(y[tr]), classes, X[va], seeds=SEEDS)
            probs = M.correct_prior(probs, M.prior_vector(pd.Series(y[tr]), classes), ALPHA)
            ok[f // 5, va] = class_array[probs.argmax(1)] == y[va]
        acc = ok.mean(1)
        print(f"  {tag:22s} acc={acc.mean():.4f} +/-{acc.std():.4f} "
              f"glia={ok[:, glia].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        return ok

    print("\n=== 5 x 5 CV ===", flush=True)
    results = {name: run(X, name) for name, X in arms.items()}
    base = results["A submitted 529"]

    rows = []
    for name in ["C replace MAX60", "F ATL60 + MAX60", "G ATL60 + cluster14",
                 "N NULL refinement"]:
        cand = results[name]
        p, _ = M.paired_mcnemar(base.ravel(), cand.ravel())
        rows.append({
            "arm": name,
            "gain": float(cand.mean() - base.mean()),
            "p_raw": float(p),
            "wins": int((cand.ravel() & ~base.ravel()).sum()),
            "losses": int((base.ravel() & ~cand.ravel()).sum()),
            "acc": float(cand.mean()),
            "glia": float(cand[:, glia].mean()),
        })

    # Holm over the 4 comparisons
    order = np.argsort([r["p_raw"] for r in rows])
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, rows[i]["p_raw"] * (len(rows) - rank))
        running = max(running, adjusted)          # Holm is monotone in rank
        rows[i]["p_holm"] = running

    null_row = next(r for r in rows if r["arm"].startswith("N"))
    null_passes = null_row["gain"] > MIN_GAIN and null_row["p_holm"] < 0.05

    print(f"\n=== paired McNemar vs A, Holm over {len(rows)} comparisons, "
          f"fold seed {FOLD_SEED} ===", flush=True)
    print(f"  baseline A acc={base.mean():.4f} glia={base[:, glia].mean():.4f}", flush=True)
    for r in rows:
        r["adopt"] = bool(r["gain"] > MIN_GAIN and r["p_holm"] < 0.05
                          and not null_passes and not r["arm"].startswith("N"))
        print(f"  {r['arm']:22s} gain {r['gain']:+.4f}  wins {r['wins']:4d} / "
              f"losses {r['losses']:4d}  p_raw {r['p_raw']:.4g}  "
              f"p_holm {r['p_holm']:.4g}  {'PASS' if r['adopt'] else 'fail'}", flush=True)

    if null_passes:
        print("\n  NULL CONTROL FIRED: a random refinement of the same shape moves "
              "accuracy as much as the real one.\n  Any gain here is feature-block "
              "width, not atlas subcluster biology. DO NOT ADOPT anything.", flush=True)

    survivors = [r["arm"] for r in rows if r["adopt"]]
    if survivors and FOLD_SEED == 23:
        print(f"\n  VERDICT: {survivors} passed the screen. "
              "Rule (iv) requires replication - rerun with --replicate "
              "(fold seed 101) before adopting.", flush=True)
    elif survivors:
        print(f"\n  VERDICT: ADOPT {survivors} (replication partition).", flush=True)
    else:
        print("\n  VERDICT: DO NOT ADOPT. The auxiliary annotation columns and the "
              "81-way subcluster refinement add nothing.", flush=True)

    tag = "replicate" if FOLD_SEED != 23 else "screen"
    frame = pd.DataFrame(rows)
    frame["fold_seed"] = FOLD_SEED
    frame["acc_base"] = base.mean()
    frame["null_fired"] = null_passes
    frame.to_csv(OUT / f"aux_transfer_{tag}.csv", index=False)
    print(f"\nwrote {OUT / f'aux_transfer_{tag}.csv'}", flush=True)


if __name__ == "__main__":
    main()
