"""Iteration 9 - replace the UNDER-POWERED atlas transfer block with a full-stack atlas model.

MECHANISM
---------
The atlas block currently feeding the submission is a single L2 logistic (C=0.1) fitted on
*expression only* - log_cpm then z-score - over the 136,612 non-challenge parent-atlas
cells (`iteration5_features.atlas_transfer`). Standalone it calls 0.604 of the challenge
training cells correctly.

That probe throws away everything the atlas knows apart from transcript counts. The atlas
carries the SAME metadata columns as the challenge, and for the 10,000 challenge cells they
are byte-identical (verified: Section ID / Mouse ID / Gender / Datasets / EI / center_x /
center_y / volume all match at 1.000; Region matches at 0.9994 under the numeric->text map;
AP_position is exactly Axial level, 1=cervical 2=lumbar 3=sacral 4=thoracic). So the same
feature stack the challenge model uses can be built for atlas cells - minus `Segment`,
which the atlas does not have.

Why that matters biologically: at 21 transcripts/cell the expression posterior for a glial
cell is nearly flat, and the thing that breaks the tie is WHERE the cell sits. The atlas
has 964 cells/section against the challenge's 70 (a 1-in-27 subsample), so an atlas model
conditioned on section + (x, y) estimates the local composition of the tissue 27x better
than anything fitted inside the challenge file can. The main Extra Trees sees the same raw
metadata, but has only ~4,000 cells per fold to learn its interaction with expression; the
atlas model estimates that interaction on 136,612 cells and hands it over pre-mixed.

PRIOR EVIDENCE
--------------
* SCORECARD 8b: an atlas learning curve on the full feature stack reaches 0.6754 standalone
  and saturates, against 0.5992 for the expression-only logistic that is actually wired in.
* SCORECARD 11d: mouse-centred atlas transfer improved the block standalone (+0.7 pt) but
  moved the stack by +0.12 / -0.10 pt (p ~ 0.6) and was NOT adopted. So standalone gain is
  not sufficient - this script has to show a stack gain, twice.
* SCORECARD 8c: 22% of glia land in the wrong *coarse* cluster; within a coarse cluster
  glia are already at 89.2%. Hence the exploratory coarse-cluster variant below.

CHEAP DIAGNOSTICS ALREADY RUN (iteration 9 pre-work, 40k-100k atlas subsamples, ET-300)
--------------------------------------------------------------------------------------
| model fitted on atlas cells        | atlas holdout | challenge train |
|------------------------------------|---------------|-----------------|
| logistic C=0.1, expression only     | 0.5926        | 0.5892          |
| ExtraTrees, expression only         | 0.5573        | 0.5550          |
| ExtraTrees, expression + QC         | 0.5604        | 0.5622          |
| ExtraTrees, expression + QC + META  | 0.6673        | 0.6610          |
| ExtraTrees, metadata + QC only      | 0.3267        | 0.3238          |
| ExtraTrees, expr+QC+META, 100k fit  | -             | 0.6790          |

Read that carefully: ExtraTrees is a WORSE expression model than the logistic (-3.4 pt).
The entire +7.5 pt comes from conditioning on metadata and position, which is exactly the
mechanism claimed above and not "a stronger learner". Argmax agreement between the two
blocks is only 0.689, so they are far from redundant - hence the CONCAT variant.

Cheap stack screen (5-fold, 1 partition, ET-300, 1 seed - noisy, directional only):

| variant                         | partition 7        | partition 23       |
|---------------------------------|--------------------|--------------------|
| baseline (logistic atlas block) | 0.7884             | 0.7932             |
| replace with ET atlas block     | 0.7956 (p=0.036)   | 0.7962 (p=0.397)   |
| CONCAT both blocks              | 0.7972 (p=0.0016)  | 0.8006 (p=0.0073)  |
| NULL: row-shuffled ET block     | 0.7900 (p=0.585)   | 0.7910 (p=0.428)   |

Concat replicated on an independent fold partition AND an independent estimator seed, and
the row-shuffled null control did nothing - so the gain is not an artefact of handing the
forest 60 more columns to sample from.

LEGITIMACY
----------
Uses the parent atlas restricted to the 200 RELEASED genes plus the atlas's own metadata,
with all 10,000 challenge cells removed before fitting - exactly the dependency already
present in the submitted model. No withheld gene touches any cell. No challenge label of
any kind enters the atlas model, so the block is out-of-sample for every challenge cell,
train and test alike, and is identical inside and outside the CV loop.

PRE-REGISTERED DECISION RULE
----------------------------
Primary candidate: CONCAT (keep the logistic block, append the ET atlas block).
Candidate family for Holm correction: {replace, concat, concat+coarse}. The null control is
not a candidate; it is a control.

ADOPT the primary iff ALL of:
  1. screen (fold partition 7, 5x5 CV, SCREEN_SEEDS estimator seeds): gain > 0 in every one
     of the 5 repeats, and the MEDIAN per-repeat exact McNemar p (5,000 paired cells per
     repeat - never 25,000, see SCORECARD 11e) is below its Holm threshold;
  2. null control: row-shuffled block gains < +0.20 pt and p > 0.05 on the same partition;
  3. confirm (fold partition 23, 5x5 CV, CONFIRM_SEEDS seeds, ONE comparison, no
     correction): gain > 0 and median per-repeat p < 0.05.
Anything else: DO NOT ADOPT. The block is cached either way; nothing here writes a
submission and nothing here touches prediction/prediction.csv.

USAGE
-----
    python3 notebooks/lib/iteration9_atlas_model.py            # block + screen + confirm
    python3 notebooks/lib/iteration9_atlas_model.py block      # build/cache the block only
    python3 notebooks/lib/iteration9_atlas_model.py screen
    python3 notebooks/lib/iteration9_atlas_model.py confirm
Runtime (measured from a 30-tree/1-seed smoke run, scaled): block ~6 min,
screen ~15 min (625 ET-600 fits), confirm ~25 min (1,000 fits) on 8 cores.
A 2x2/1-seed smoke run of this exact file gave concat +0.46 pt, concat+coarse
+0.57 pt, null -0.13 pt - directionally consistent with the table above.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
BLOCK_CACHE = OUT / "atlas_et_block.npz"
CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")

ATLAS_TREES = 400
ATLAS_SEEDS = (0, 1, 2)
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
N_SPLITS, N_REPEATS = 5, 5
SCREEN_PARTITION, CONFIRM_PARTITION = 7, 23
ALPHA = 0.45

REGION_TEXT = {1.0: "dorsal horn", 2.0: "dorsal horn/intermediate zone",
               3.0: "intermediate zone", 4.0: "intermediate zone/ventral horn",
               5.0: "ventral horn"}
AP_TEXT = {1: "cervical", 2: "lumbar", 3: "sacral", 4: "thoracic"}
META_KEYS = ["Region", "Excitatory_vs_Inhibitory", "Datasets", "Gender", "Mouse ID",
             "Axial level", "Section ID"]


# ---------------------------------------------------------------- atlas ET block
def _categorical(handle, key):
    cats = [c.decode() if isinstance(c, bytes) else c
            for c in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.array([cats[i] if i >= 0 else "nan" for i in codes])


def load_atlas(genes, meta_train, meta_test):
    """Every atlas cell, 200 RELEASED genes only, with the 10,000 challenge cells flagged."""
    with h5py.File(F.PARENT_ATLAS, "r") as h:
        ids = np.array([x.decode() for x in h["obs/_index"][:]])
        atlas_genes = [g.decode() for g in h["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        missing = [g for g in genes if g not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} released genes absent from the atlas")
        cols = np.array([lookup[g] for g in genes])
        X = sparse.csr_matrix(
            (h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)))
        expr = np.asarray(X[:, cols].todense(), np.float32)
        obs = {k: _categorical(h, k) for k in META_KEYS}
        obs["coarse"] = _categorical(h, "1st round cluster")
        labels = np.array([F._normalise_label(s)
                           for s in _categorical(h, "MERFISH cell type annotation")])
        for k in ("center_x", "center_y", "volume"):
            obs[k] = h[f"obs/{k}"][:]

    position = {c: i for i, c in enumerate(ids)}
    challenge = np.zeros(len(ids), bool)
    for index in (meta_train.index, meta_test.index):
        challenge[[position[c] for c in index.astype(str) if c in position]] = True
    print(f"[atlas-et] challenge cells found in the atlas and removed: "
          f"{int(challenge.sum())}")
    return expr, labels, obs, challenge


def _qc(expr, volume, center_x, center_y):
    total, detected = expr.sum(1), (expr > 0).sum(1)
    volume = np.asarray(volume, float)
    safe = np.where(volume == 0, np.nan, volume)
    return np.nan_to_num(np.column_stack([
        np.log1p(total), detected, (expr == 0).mean(1),
        np.log1p(np.clip(volume, 0, None)), total / safe, detected / safe,
        volume, center_x, center_y]), nan=-1.0)


def _atlas_meta(obs, rows):
    return pd.DataFrame({k: obs[k][rows] for k in META_KEYS})


def _challenge_meta(meta):
    return pd.DataFrame({
        "Region": meta["Region"].map(REGION_TEXT).fillna("nan").to_numpy(),
        "Excitatory_vs_Inhibitory":
            meta["Excitatory_vs_Inhibitory"].fillna("nan").astype(str).to_numpy(),
        "Datasets": meta["Datasets"].astype(str).to_numpy(),
        "Gender": meta["Gender"].astype(str).to_numpy(),
        "Mouse ID": meta["Mouse_ID"].astype(str).to_numpy(),
        "Axial level": meta["AP_position"].map(AP_TEXT).astype(str).to_numpy(),
        "Section ID": meta["Section_ID"].astype(str).to_numpy()})


def build_block(genes, classes, counts_train, meta_train, counts_test, meta_test):
    """Fit the atlas model once on ALL non-challenge atlas cells, emit 60 + 14 columns.

    The fit never sees a challenge cell or a challenge label, so its probabilities are
    out-of-sample for every challenge cell and need no fold scoping.
    """
    t0 = time.time()
    expr, labels, obs, challenge = load_atlas(genes, meta_train, meta_test)
    usable = (~challenge) & np.isin(labels, list(classes)) & (expr.sum(1) > 0)
    rows = np.flatnonzero(usable)
    print(f"[atlas] {len(rows)} training cells "
          f"({int(challenge.sum())} challenge cells removed) ({time.time()-t0:.0f}s)",
          flush=True)

    encoder = OneHotEncoder(handle_unknown="ignore").fit(
        pd.concat([_atlas_meta(obs, rows), _challenge_meta(meta_train),
                   _challenge_meta(meta_test)]))

    def features(expression, volume, center_x, center_y, meta_frame):
        return np.hstack([
            F.log_cpm(expression), _qc(expression, volume, center_x, center_y),
            encoder.transform(meta_frame).toarray()]).astype(np.float32)

    X_atlas = features(expr[rows], obs["volume"][rows], obs["center_x"][rows],
                       obs["center_y"][rows], _atlas_meta(obs, rows))
    X_train = features(counts_train.to_numpy(np.float32), meta_train["volume"].to_numpy(),
                       meta_train["center_x"].to_numpy(), meta_train["center_y"].to_numpy(),
                       _challenge_meta(meta_train))
    X_test = features(counts_test.to_numpy(np.float32), meta_test["volume"].to_numpy(),
                      meta_test["center_x"].to_numpy(), meta_test["center_y"].to_numpy(),
                      _challenge_meta(meta_test))
    print(f"[atlas] feature matrices {X_atlas.shape} / {X_train.shape}", flush=True)

    def fit_and_align(targets, target_classes):
        stacked = [np.zeros((len(x), len(target_classes)), np.float32)
                   for x in (X_train, X_test)]
        for seed in ATLAS_SEEDS:
            model = ExtraTreesClassifier(ATLAS_TREES, max_features="sqrt",
                                         min_samples_leaf=1, n_jobs=-1,
                                         random_state=seed).fit(X_atlas, targets)
            for out, x in zip(stacked, (X_train, X_test)):
                out += M.align_proba(model, x, target_classes)
        return [s / len(ATLAS_SEEDS) for s in stacked]

    t0 = time.time()
    fine_tr, fine_te = fit_and_align(labels[rows], classes)
    print(f"[atlas] 60-class ET fitted, {len(ATLAS_SEEDS)} seeds ({time.time()-t0:.0f}s)",
          flush=True)

    t0 = time.time()
    coarse_classes = sorted(set(obs["coarse"][rows]))
    coarse_tr, coarse_te = fit_and_align(obs["coarse"][rows], coarse_classes)
    print(f"[atlas] {len(coarse_classes)}-cluster ET fitted ({time.time()-t0:.0f}s)",
          flush=True)

    np.savez_compressed(BLOCK_CACHE, ATL_ET_TR=fine_tr, ATL_ET_TE=fine_te,
                        COARSE_TR=coarse_tr, COARSE_TE=coarse_te,
                        classes=np.array(classes), coarse=np.array(coarse_classes),
                        n_atlas=len(rows))
    print(f"wrote {BLOCK_CACHE}", flush=True)
    return fine_tr, fine_te, coarse_tr, coarse_te


# ---------------------------------------------------------------- evaluation
def run_cv(X, y, classes, folds, seeds):
    class_array = np.array(classes)
    ok = np.zeros((N_REPEATS, len(y)), bool)
    for i, (train, valid) in enumerate(folds):
        probs = M.fit_extra_trees(X[train], pd.Series(y[train]), classes, X[valid],
                                  seeds=seeds)
        probs = M.correct_prior(probs, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        ok[i // N_SPLITS, valid] = class_array[probs.argmax(1)] == y[valid]
    return ok


def per_repeat_mcnemar(ok_candidate, ok_base):
    """One OOF prediction per cell per repeat - never flatten repeats (SCORECARD 11e)."""
    out = []
    for r in range(ok_base.shape[0]):
        p, table = M.paired_mcnemar(ok_candidate[r], ok_base[r])
        out.append({"repeat": r, "gain": ok_candidate[r].mean() - ok_base[r].mean(),
                    "p": p, "wins": table[0][1], "losses": table[1][0]})
    return out


def evaluate(configs, y, classes, glia, partition, seeds, tag):
    folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                         random_state=partition).split(y, y))
    print(f"\n=== {tag}: {N_SPLITS}x{N_REPEATS} CV, partition seed {partition}, "
          f"{len(seeds)} estimator seeds ===", flush=True)
    results = {}
    for name, X in configs.items():
        t0 = time.time()
        ok = run_cv(X, y, classes, folds, seeds)
        results[name] = ok
        accuracy = ok.mean(1)
        print(f"  {name:30s} {X.shape[1]:4d}f acc={accuracy.mean():.5f} "
              f"+/-{accuracy.std():.5f} glia={ok[:, glia].mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return results


def report(results, baseline_name, candidate_names, y, glia, family_size, label):
    base = results[baseline_name]
    rows = []
    for name in candidate_names:
        ok = results[name]
        per_repeat = per_repeat_mcnemar(ok, base)
        ps = sorted(r["p"] for r in per_repeat)
        rows.append({
            "variant": name, "accuracy": ok.mean(), "glia": ok[:, glia].mean(),
            "gain": ok.mean() - base.mean(),
            "gain_min_repeat": min(r["gain"] for r in per_repeat),
            "median_p": float(np.median(ps)),
            "all_repeats_positive": bool(all(r["gain"] > 0 for r in per_repeat)),
            "wins": sum(r["wins"] for r in per_repeat),
            "losses": sum(r["losses"] for r in per_repeat)})
    rows.sort(key=lambda r: r["median_p"])
    print(f"\n=== {label}: per-repeat exact McNemar vs baseline "
          f"(5,000 paired cells per repeat) ===", flush=True)
    for i, r in enumerate(rows):
        r["holm"] = 0.05 / max(family_size - i, 1)
        r["passes"] = bool(r["gain"] > 0 and r["all_repeats_positive"]
                           and r["median_p"] < r["holm"])
        print(f"  {r['variant']:30s} gain {r['gain']:+.5f} "
              f"(worst repeat {r['gain_min_repeat']:+.5f}) median p={r['median_p']:.3g} "
              f"Holm {r['holm']:.4f} wins/losses {r['wins']}/{r['losses']}"
              f"{'   <== PASSES' if r['passes'] else ''}", flush=True)
    return rows


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    glia = meta_train["Region"].isna().to_numpy()
    genes = list(counts_train.columns)

    if stage in ("all", "block") or not BLOCK_CACHE.exists():
        build_block(genes, classes, counts_train, meta_train, counts_test, meta_test)
    blocks = np.load(BLOCK_CACHE, allow_pickle=True)
    ATL_ET, COARSE = blocks["ATL_ET_TR"], blocks["COARSE_TR"]

    cache = np.load(CACHE, allow_pickle=True)
    core = np.hstack([cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"],
                      cache["NIC_TR"]]).astype(np.float32)
    ATL_LOGIT = cache["ATL_TR"]

    class_array = np.array(classes)
    print(f"\nstandalone on the 5,000 challenge TRAINING cells:", flush=True)
    print(f"  logistic atlas block (submitted) {(class_array[ATL_LOGIT.argmax(1)] == y).mean():.4f}",
          flush=True)
    print(f"  ExtraTrees atlas block (new)     {(class_array[ATL_ET.argmax(1)] == y).mean():.4f}",
          flush=True)
    print(f"  argmax agreement between blocks  "
          f"{(ATL_ET.argmax(1) == ATL_LOGIT.argmax(1)).mean():.4f}", flush=True)

    rng = np.random.default_rng(0)
    shuffled = ATL_ET[rng.permutation(len(ATL_ET))]
    baseline = np.hstack([core, ATL_LOGIT]).astype(np.float32)
    concat = np.hstack([core, ATL_LOGIT, ATL_ET]).astype(np.float32)
    configs = {
        "baseline (logistic block)": baseline,
        "replace with ET block": np.hstack([core, ATL_ET]).astype(np.float32),
        "CONCAT both blocks": concat,
        "CONCAT + coarse cluster": np.hstack([core, ATL_LOGIT, ATL_ET, COARSE]).astype(np.float32),
        "NULL row-shuffled ET block": np.hstack([core, ATL_LOGIT, shuffled]).astype(np.float32),
    }

    screen_rows, confirm_rows = [], []
    if stage in ("all", "screen"):
        results = evaluate(configs, y, classes, glia, SCREEN_PARTITION, SCREEN_SEEDS,
                           "SCREEN")
        screen_rows = report(results, "baseline (logistic block)",
                             ["replace with ET block", "CONCAT both blocks",
                              "CONCAT + coarse cluster"], y, glia, 3, "SCREEN")
        null_rows = report(results, "baseline (logistic block)",
                           ["NULL row-shuffled ET block"], y, glia, 1, "NULL CONTROL")
        null = null_rows[0]
        null_ok = bool(null["gain"] < 0.0020 and null["median_p"] > 0.05)
        print(f"\n  null control {'PASSES (block carries cell-wise information)' if null_ok else 'FAILS - the gain is a feature-count artefact'}",
              flush=True)
        pd.DataFrame(screen_rows + null_rows).to_csv(OUT / "atlas_model_screen.csv",
                                                    index=False)
        primary = next(r for r in screen_rows if r["variant"] == "CONCAT both blocks")
        print(f"\n  SCREEN VERDICT for the primary candidate: "
              f"{'PROCEED to confirm' if primary['passes'] and null_ok else 'STOP - do not adopt'}",
              flush=True)

    if stage in ("all", "confirm"):
        results = evaluate({"baseline (logistic block)": baseline,
                            "CONCAT both blocks": concat},
                           y, classes, glia, CONFIRM_PARTITION, CONFIRM_SEEDS, "CONFIRM")
        confirm_rows = report(results, "baseline (logistic block)",
                              ["CONCAT both blocks"], y, glia, 1, "CONFIRM")
        row = confirm_rows[0]
        adopt = bool(row["gain"] > 0 and row["median_p"] < 0.05
                     and row["all_repeats_positive"])
        pd.DataFrame(confirm_rows).to_csv(OUT / "atlas_model_confirm.csv", index=False)
        print(f"\n  VERDICT: {'ADOPT - append ATL_ET_TR/TE as a sixth block' if adopt else 'DO NOT ADOPT'}",
              flush=True)
        print("  (adoption means editing the block list in a COPY of make_submission.py; "
              "this script never writes a submission)", flush=True)


if __name__ == "__main__":
    main()
