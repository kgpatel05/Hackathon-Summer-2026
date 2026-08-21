"""Iteration 9 - atlas neighbourhood cell-type COMPOSITION as a feature block.

--------------------------------------------------------------------------------
MECHANISM (why this is not "spatial features, again")
--------------------------------------------------------------------------------
Spinal glial subtypes are not spatially random; several of them are defined by the
tissue compartment they sit in.

  * astrocyte_1 / astrocyte_2 is the protoplasmic (grey matter) vs fibrous (white
    matter) split - a textbook anatomical distinction, not a transcriptional nuance.
  * oligodendrocyte_2 is concentrated in white-matter tracts; oligodendrocyte_1 and
    oligodendrocyte_progenitor_2 are distributed differently within the same lineage.
  * meninges_1/2/3 form a surface sheet: their own type is 13x / 27x / 24x enriched
    among their physical neighbours relative to its global rate.

The challenge file cannot express any of this because the organisers released a
1-in-27 subsample: ~70 cells per section against 964 in the parent atlas (SCORECARD
S10c). At that density a cell's nearest in-file neighbour is hundreds of microns
away, which is exactly why spatial kNN scored 0.204 and why the GNN work lost to a
random-graph control (S10n). Restoring the 26 missing neighbours per cell from the
public parent atlas restores the compartment signal.

--------------------------------------------------------------------------------
WHY THE EXISTING NEGATIVE RESULTS DO NOT COVER THIS
--------------------------------------------------------------------------------
Three prior experiments look like this one and are all different questions:

  S10c  neighbour LABEL VOTING over the full atlas -> 0.2164 standalone.  That is an
        ARGMAX over the neighbour histogram, evaluated ALONE.  A decision rule that
        must win a 60-way argmax can never fire for meninges_2 at 0.65% prevalence,
        however enriched it is locally.  Used as a FEATURE next to expression, the
        same histogram lets the model say "expression is torn between meninges_1 and
        meninges_2, and 18% of my true neighbours are meninges_2 against a 0.65%
        base rate".
  S10n  edge homophily 0.0936 vs 0.0559 chance -> "only 1.7x".  That global average
        is dominated by neurons (36% of cells, spatially interdigitated) and by the
        two diffuse abundant glial types (OPC_2, astrocyte_1, both ~1.3x).  It hides
        3.3x for oligodendrocyte_2, 4.1x for astrocyte_2 and 13-27x for the meninges.
        The mean over classes was read as a statement about every class.
  S11c  atlas niche EXPRESSION (mean of the 200 released genes over true neighbours,
        PCA'd) -> +0.27 pt screen, +0.11 pt confirm, rejected.  Mean expression over
        50 cells on a NEURON panel is dominated by how many neurons are nearby; it is
        a blunt encoding of compartment.  The neighbours' cell-type ANNOTATIONS are
        the study's own 500-gene calls, so the composition vector is a sharper and
        strictly different summary of the same neighbourhood.

That last point is also why S10e's data-processing-inequality argument does not
apply.  It says a classifier g(X, f(X)) built from imputed values f(X) cannot exceed
I(Y; X), and it is correct.  Neighbour composition is NOT a function of this cell's
X.  It is a function of OTHER cells' 500-gene-derived annotations, so it can and does
carry information about Y that the cell's own 200 genes do not.  This is the one
channel the measured "200-gene ceiling" argument never closed.

--------------------------------------------------------------------------------
PRIOR EVIDENCE FROM CHEAP DIAGNOSTICS (this iteration, 5,000 training cells)
--------------------------------------------------------------------------------
Single best atlas-neighbour composition feature, one-vs-one AUC, against the released
200-gene AUC for the same pair reported in SCORECARD S10d:

  pair                                     atlas-nbr  released-200  majority floor
  astrocyte_1 vs astrocyte_2                 0.954       0.8919         0.748
  oligodendrocyte_1 vs oligodendrocyte_2     0.866       0.9455         0.598
  OPC_2 vs oligodendrocyte_2                 0.806       0.9128         0.652
  OPC_2 vs oligodendrocyte_1                 0.633       0.7846         0.736

Conditional check (4-fold logistic, astrocyte_1 vs astrocyte_2, n=769):
  full 529-feature stack        AUC 0.962
  full stack + composition      AUC 0.971

Screen already run (3-fold, 1 ET seed, alpha=0.45, identical folds):
  baseline 529 features         acc 0.7906   neurons 0.8999   glia 0.7260
  + composition k=30 (60 cols)  acc 0.7934   neurons 0.8934   glia 0.7342
  + composition, ROW-SHUFFLED   acc 0.7904   neurons 0.8999   glia 0.7257   <- null

And the mechanism check that matters, because SCORECARD S8c established that the
residual glial error is a COARSE-identity error, not a fine-subtype error:

  coarse "1st round cluster" accuracy on glia   0.8170 -> 0.8237  (+0.67 pt)
  exact accuracy WITHIN a correctly called cluster  0.8886 -> 0.8914  (+0.28 pt)

The block moves the coarse call, which is the error S8c said was the real one, and
leaves fine splitting alone.  Per-pair: astrocyte_1/astrocyte_2 0.758 -> 0.789,
meninges_1/meninges_2 0.550 -> 0.569, and OPC_2/oligodendrocyte_1 0.652 -> 0.660,
i.e. the pair with no anatomical separation (AUC 0.633) barely moves - the gain
lands where the mechanism says it should, not uniformly.

Neurons lost 0.65 pt in that screen, but the NULL control lost 0.32 pt at identical
width, so roughly half of that is block-width noise rather than the block.  A
narrowed 22-column variant was also screened and was worse on glia (0.7320 vs
0.7342), so the full 60-column block is what is registered here.

--------------------------------------------------------------------------------
LEGITIMACY
--------------------------------------------------------------------------------
Uses: coordinates + Section_ID of challenge cells (released), and the public parent
atlas restricted to cells that are NOT in the challenge (all 10,000 removed before
anything is computed).  Reads the atlas cell-type annotation only for those external
cells - the identical resource and the identical column that the already-submitted
`atlas_transfer` block trains on.  No challenge cell's label is read, no test label is
read, and none of the 300 withheld genes is read for any cell, challenge or atlas.

The one thing to disclose on review: a neighbour's public annotation was derived by
the study from 500 genes, so this feature carries 500-gene information about the
MICROENVIRONMENT (never about the target cell's own transcriptome).  That is the same
standing as using those cells as labelled training examples, which the submission
already does.

--------------------------------------------------------------------------------
PRE-REGISTERED DECISION RULE  (fixed before any number below is produced)
--------------------------------------------------------------------------------
Protocol follows S11e: ONE out-of-fold prediction per cell per partition, exact
McNemar on 5,000 independent cells - never on flattened repeated-CV rows.

  SCREEN   partition seed 7, 5-fold, 5 ET seeds.  Variants: k=10, k=30, k=100, plus
           the row-shuffled null at the winning k.  Holm-Bonferroni across the three
           real variants.  A variant survives iff
             (a) gain > 0, (b) Holm-adjusted p < 0.05, (c) it beats the null control
             variant's gain.
  CONFIRM  the single surviving k, partition seed 23 (independent fold assignment),
           20 ET seeds, one hypothesis, no correction.
  ADOPT    iff CONFIRM gain > 0 AND CONFIRM p < 0.05.

Anything else is DO NOT ADOPT, recorded, and the submission is left alone.  This rule
exists because S10p-1 and S11c both produced screen winners that reversed or halved
on a second partition; changing fold assignment alone moves the baseline 0.28 pt.

Usage:
    python3 notebooks/lib/iteration9_freshidea.py screen
    python3 notebooks/lib/iteration9_freshidea.py confirm      # k read from screen
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

ALPHA = 0.45
K_GRID = (10, 30, 100)
SCREEN_SEED, CONFIRM_SEED = 7, 23
N_SPLITS = 5
SCREEN_ET_SEEDS = tuple(range(5))
CONFIRM_ET_SEEDS = tuple(range(20))
NULL_RNG = 0

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "atlas_composition_cache.npz"
SCREEN_JSON = OUT / "freshidea_screen.json"
PARENT_ATLAS = Path("data/external/MERFISH_spinal_cord_0531.h5ad")


# ----------------------------------------------------------------------
# The feature block
# ----------------------------------------------------------------------
def atlas_composition(meta_all: pd.DataFrame, classes, ks=K_GRID) -> dict[int, np.ndarray]:
    """Cell-type composition of each cell's k nearest PARENT-ATLAS neighbours.

    Donor pool = parent-atlas cells that are not challenge cells.  All 10,000
    challenge cells (train and test alike) are removed before the KD-trees are built,
    so no challenge label - and in particular no test label - can enter the feature.
    Neighbours are taken within the cell's own tissue section; challenge coordinates
    are bit-identical to the atlas ones (SCORECARD S6a), so the join is exact.

    Returns {k: (n_cells, 61)} - 60 challenge classes plus one column for atlas
    annotations outside the challenge taxonomy, so each row sums to 1.
    """
    with h5py.File(PARENT_ATLAS, "r") as handle:
        ids = np.array([x.decode() for x in handle["obs/_index"][:]])
        cat = [c.decode() for c in handle["obs/MERFISH cell type annotation/categories"][:]]
        codes = handle["obs/MERFISH cell type annotation/codes"][:]
        sec_cat = [c.decode() for c in handle["obs/Section ID/categories"][:]]
        sec_codes = handle["obs/Section ID/codes"][:]
        ax = handle["obs/center_x"][:].astype(float)
        ay = handle["obs/center_y"][:].astype(float)

    atlas_label = np.array([F._normalise_label(cat[c]) if c >= 0 else "NA" for c in codes])
    atlas_section = np.array([sec_cat[c] if c >= 0 else "NA" for c in sec_codes])

    index_of = {c: i for i, c in enumerate(classes)}
    other = len(classes)                       # the 61st column
    atlas_code = np.array([index_of.get(l, other) for l in atlas_label])

    position = {c: i for i, c in enumerate(ids)}
    missing = [c for c in meta_all.index.astype(str) if c not in position]
    if missing:
        raise ValueError(f"{len(missing)} challenge cells absent from the parent atlas")
    challenge_rows = np.array([position[c] for c in meta_all.index.astype(str)])

    is_challenge = np.zeros(len(ids), bool)
    is_challenge[challenge_rows] = True
    donors = np.flatnonzero(~is_challenge)
    assert len(donors) == len(ids) - len(meta_all), "donor pool still contains challenge cells"

    query_section = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    out = {k: np.zeros((len(meta_all), other + 1), np.float32) for k in ks}

    donor_section = atlas_section[donors]
    for section in np.unique(query_section):
        pool = donors[donor_section == section]
        rows = np.flatnonzero(query_section == section)
        if len(pool) < 10 or len(rows) == 0:
            continue
        tree = cKDTree(np.column_stack([ax[pool], ay[pool]]))
        pool_code = atlas_code[pool]
        for k in ks:
            _, neighbours = tree.query(query_xy[rows], k=min(k, len(pool)))
            neighbours = np.atleast_2d(neighbours)
            taken = pool_code[neighbours]
            for j in range(other + 1):
                out[k][rows, j] = (taken == j).mean(1)
    return out


def shuffled_null(block: np.ndarray, meta_all: pd.DataFrame, seed=NULL_RNG) -> np.ndarray:
    """Permute composition rows WITHIN each section.

    This is the control the idea has to beat.  It preserves the block's width, its
    marginal distribution and its section-level composition - everything except the
    within-section spatial assignment, which is the only thing the mechanism claims.
    If the real block does not beat this, the gain was extra columns, exactly as the
    random graph (S10n) and the duplicate rows (S10o-3) turned out to be.
    """
    rng = np.random.default_rng(seed)
    out = block.copy()
    section = meta_all["Section_ID"].astype(str).to_numpy()
    for s in np.unique(section):
        rows = np.flatnonzero(section == s)
        out[rows] = block[rng.permutation(rows)]
    return out


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def build_base_stack():
    """The 529-feature submitted stack, from cache when available."""
    cache = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str)
    classes = sorted(y.unique())
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        X = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], c["ATL_TR"]])
        return X.astype(np.float32), y.to_numpy(str), classes, meta_train, meta_all

    genes = list(counts_train.columns)
    encoder = OneHotEncoder(handle_unknown="ignore").fit(
        pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str))
    base = F.base_block(counts_train, meta_train, encoder)
    (ext, _), _, _ = F.reference_transfer(genes, classes, [counts_train, counts_test])
    neuron_all = (meta_all["Region"] == 1).to_numpy()
    spa = F.registered_spatial(meta_all, neuron_all)[: len(meta_train)]
    expr_all = F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]))
    nic = F.niche_expression(expr_all, meta_all, k=15, n_components=30)[: len(meta_train)]
    (atl, _), _ = F.atlas_transfer(genes, classes, [counts_train, counts_test])
    X = np.hstack([base, ext, spa, nic, atl]).astype(np.float32)
    return X, y.to_numpy(str), classes, meta_train, meta_all


def oof_correct(X, y, classes, fold_seed, et_seeds):
    """One out-of-fold prediction per cell.  Returns a boolean correctness vector."""
    prior = M.prior_vector(y, classes)
    probs = np.zeros((len(y), len(classes)), np.float32)
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=fold_seed)
    for train_idx, eval_idx in splitter.split(X, y):
        probs[eval_idx] = M.fit_extra_trees(
            X[train_idx], pd.Series(y[train_idx]), classes, X[eval_idx], seeds=et_seeds)
    pred = np.array(classes)[M.correct_prior(probs, prior, ALPHA).argmax(1)]
    return pred == y


def report(name, correct, baseline, neuron):
    gain = 100 * (correct.mean() - baseline.mean())
    p, table = M.paired_mcnemar(correct, baseline)
    wins, losses = table[0][1], table[1][0]
    print(f"  {name:26s} acc={correct.mean():.5f}  neu={correct[neuron].mean():.4f}  "
          f"glia={correct[~neuron].mean():.4f}  gain={gain:+.2f} pt  "
          f"{wins}w/{losses}l  p={p:.4g}", flush=True)
    return dict(name=name, accuracy=float(correct.mean()), gain_pt=float(gain),
                wins=int(wins), losses=int(losses), p=float(p),
                neurons=float(correct[neuron].mean()), glia=float(correct[~neuron].mean()))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    t0 = time.time()
    X, y, classes, meta_train, meta_all = build_base_stack()
    neuron = (~meta_train["Region"].isna()).to_numpy()
    print(f"base stack {X.shape}  neurons={neuron.sum()}  glia={(~neuron).sum()}", flush=True)

    if CACHE.exists():
        z = np.load(CACHE)
        comp_all = {k: z[f"k{k}"] for k in K_GRID}
    else:
        comp_all = atlas_composition(meta_all, classes)
        np.savez_compressed(CACHE, **{f"k{k}": v for k, v in comp_all.items()})
    comp = {k: v[: len(meta_train)] for k, v in comp_all.items()}
    print(f"composition block built ({time.time()-t0:.0f}s); "
          f"{comp[K_GRID[0]].shape[1]} columns per k", flush=True)

    if mode == "screen":
        seeds, fold_seed = SCREEN_ET_SEEDS, SCREEN_SEED
        print(f"\nSCREEN  partition seed {fold_seed}, {len(seeds)} ET seeds")
        baseline = oof_correct(X, y, classes, fold_seed, seeds)
        results = [report("baseline 529", baseline, baseline, neuron)]
        for k in K_GRID:
            c = oof_correct(np.hstack([X, comp[k]]), y, classes, fold_seed, seeds)
            results.append(report(f"+ composition k={k}", c, baseline, neuron))

        real = results[1:]
        order = np.argsort([r["p"] for r in real])
        holm_pass, m = {}, len(real)
        for rank, idx in enumerate(order):
            threshold = 0.05 / (m - rank)
            holm_pass[real[idx]["name"]] = (real[idx]["p"] < threshold
                                            and real[idx]["gain_pt"] > 0)
            real[idx]["holm_threshold"] = threshold
        best = max(real, key=lambda r: r["gain_pt"])
        best_k = int(best["name"].split("=")[1])

        null_correct = oof_correct(
            np.hstack([X, shuffled_null(comp[best_k], meta_train)]),
            y, classes, fold_seed, seeds)
        null_result = report(f"NULL shuffled k={best_k}", null_correct, baseline, neuron)

        survives = (holm_pass.get(best["name"], False)
                    and best["gain_pt"] > null_result["gain_pt"])
        print(f"\nbest = {best['name']}  Holm-pass={holm_pass.get(best['name'])}  "
              f"beats-null={best['gain_pt'] > null_result['gain_pt']}")
        print("SCREEN VERDICT:", "PROCEED TO CONFIRM" if survives else "STOP - DO NOT ADOPT")
        SCREEN_JSON.write_text(json.dumps(
            dict(results=results, null=null_result, best_k=best_k,
                 survives=bool(survives)), indent=2))

    elif mode == "confirm":
        if not SCREEN_JSON.exists():
            raise SystemExit("run `screen` first - the confirm k is fixed by the screen")
        screen = json.loads(SCREEN_JSON.read_text())
        if not screen["survives"]:
            raise SystemExit("screen verdict was STOP; confirming anyway would be "
                             "shopping for a partition that agrees")
        k = screen["best_k"]
        print(f"\nCONFIRM  partition seed {CONFIRM_SEED}, {len(CONFIRM_ET_SEEDS)} ET seeds, k={k}")
        baseline = oof_correct(X, y, classes, CONFIRM_SEED, CONFIRM_ET_SEEDS)
        report("baseline 529", baseline, baseline, neuron)
        cand = oof_correct(np.hstack([X, comp[k]]), y, classes,
                           CONFIRM_SEED, CONFIRM_ET_SEEDS)
        result = report(f"+ composition k={k}", cand, baseline, neuron)
        adopt = result["gain_pt"] > 0 and result["p"] < 0.05
        print("\nCONFIRM VERDICT:", "ADOPT" if adopt else "DO NOT ADOPT")
        print("(pre-registered rule: adopt iff gain > 0 and p < 0.05 on this partition)")
        (OUT / "freshidea_confirm.json").write_text(
            json.dumps(dict(k=k, result=result, adopt=bool(adopt)), indent=2))
    else:
        raise SystemExit("mode must be 'screen' or 'confirm'")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
