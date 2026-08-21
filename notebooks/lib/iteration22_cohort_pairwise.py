"""Nested pairwise empirical-Bayes cohort decoding for Iteration 22.

The full 21-class glial deconvolution in ``iteration22_cohort_deconvolution.py`` was
unstable across its untouched confirmation cohort.  This follow-up freezes a lower-rank
version before evaluation: learn the five largest *disjoint* glial confusion pairs from
each outer-training fold, estimate a 2x2 soft-confusion operator for each, and correct
only evaluation cells whose top two calls are that pair.  Mouse-level source counts are
the shrinkage prior; target evidence comes from the evaluation cohort's own probabilities.

There is no hyperparameter selection: K=5, ridge=1, source_tau=20 and beta=0.5 are fixed.
Every reported training prediction is from a five-fold outer split.  Pool weights,
confusion pairs, confusion matrices, and mouse priors are all fit without the outer fold.
A matched null permutes labels within (mouse, glia), retaining mouse composition while
destroying pairwise prediction/truth correspondence.  Test truth is read only if the
predeclared nested-CV gate passes.  No submission is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS
import iteration5_models as M
import iteration22_cohort_deconvolution as CD


OUT = Path("outputs/iteration22/cohort")
OUT.mkdir(parents=True, exist_ok=True)
N_PAIRS = 5
RIDGE = 1.0
SOURCE_TAU = 20.0
CONFUSION_TAU = 4.0
BETA = 0.5
OUTER_SEED = 2311
INNER_SEED = 2347
VIEWS = (18, 41, 59, 83)
EPS = 1e-9


def pooled_one_view(fit, evaluate, seed, names, data):
    lgd, allow, y, classes = SS.part(seed)
    logs = np.stack([lgd[n] for n in names])
    prior = (
        pd.Series(y[fit]).value_counts(normalize=True).reindex(classes)
        .fillna(EPS).to_numpy()
    )
    lp = np.log(prior)
    glia = data["meta_train"]["Region"].isna().to_numpy()
    z = np.full((len(evaluate), len(classes)), -1e9, float)
    for mask in (glia, ~glia):
        fr = fit[mask[fit]]
        local = np.flatnonzero(mask[evaluate])
        if not len(local):
            continue
        er = evaluate[local]
        w, a = LP.fit(logs, y, classes, lp, allow, rows=fr, l2=1e-3)
        z[local] = LP.apply(logs[:, er], w, a, lp, allow[er])
    return softmax(z, axis=1)


def crossfit_source(rows, seed, names, data):
    out = np.zeros((len(rows), len(data["classes"])), float)
    for local_fit, local_val in KFold(
        3, shuffle=True, random_state=INNER_SEED + seed
    ).split(rows):
        out[local_val] = pooled_one_view(
            rows[local_fit], rows[local_val], seed, names, data
        )
    return out


def disjoint_pairs(prob, y, glia, classes):
    pred = classes[prob.argmax(1)]
    counts: dict[tuple[str, str], int] = {}
    for a, b, is_glia in zip(y, pred, glia):
        if is_glia and a != b:
            pair = tuple(sorted((str(a), str(b))))
            counts[pair] = counts.get(pair, 0) + 1
    used: set[str] = set()
    selected = []
    for pair, n in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if pair[0] in used or pair[1] in used:
            continue
        selected.append((pair[0], pair[1], n))
        used.update(pair)
        if len(selected) == N_PAIRS:
            break
    return selected


def pair_confusion(local_prob, local_y, pair, null=False, meta=None):
    labels = local_y.copy()
    if null:
        labels = CD.permute_within_mouse_branch(labels, meta, 2399)
    take = np.isin(labels, pair)
    p = local_prob[take]
    yy = labels[take]
    matrix = np.zeros((2, 2), float)
    for j, label in enumerate(pair):
        r = yy == label
        matrix[j] = (p[r].sum(0) + CONFUSION_TAU * np.eye(2)[j]) / (
            int(r.sum()) + CONFUSION_TAU
        )
    matrix /= np.maximum(matrix.sum(1, keepdims=True), EPS)
    return matrix


def adjust(fit_prob, fit_y, fit_meta, eval_prob, eval_meta, classes, pairs, null=False):
    out = eval_prob.copy()
    lookup = {str(c): j for j, c in enumerate(classes)}
    fit_mouse = fit_meta["Mouse_ID"].astype(str).to_numpy()
    eval_mouse = eval_meta["Mouse_ID"].astype(str).to_numpy()
    source_y = (
        CD.permute_within_mouse_branch(fit_y, fit_meta, 2399) if null else fit_y
    )
    for first, second, _ in pairs:
        pair = (first, second)
        idx = np.asarray([lookup[first], lookup[second]])
        fit_local = fit_prob[:, idx]
        fit_local /= np.maximum(fit_local.sum(1, keepdims=True), EPS)
        cm = pair_confusion(
            fit_local, fit_y, pair, null=null, meta=fit_meta
        )
        global_counts = np.asarray([(source_y == label).sum() for label in pair], float) + 0.5
        global_prior = global_counts / global_counts.sum()
        # Relevance is decided from the unadjusted top two, so pair order cannot cause
        # a cell to enter another pair later. Pairs are disjoint by construction.
        top2 = np.argsort(-eval_prob, axis=1)[:, :2]
        relevant = np.asarray([set(row) == set(idx) for row in top2])
        for mouse in np.unique(eval_mouse[relevant]):
            erows = np.flatnonzero(relevant & (eval_mouse == mouse))
            frows = np.flatnonzero((fit_mouse == mouse) & np.isin(source_y, pair))
            if len(erows) < 2:
                continue
            counts = np.asarray([(source_y[frows] == label).sum() for label in pair], float)
            source_prior = (counts + SOURCE_TAU * global_prior) / (
                counts.sum() + SOURCE_TAU
            )
            local = eval_prob[erows][:, idx]
            local /= np.maximum(local.sum(1, keepdims=True), EPS)
            observed = local.mean(0)
            target_prior = CD.solve_prior(cm, observed, source_prior, RIDGE)
            ratio = np.power(
                np.maximum(target_prior, 1e-5) / np.maximum(observed, 1e-5), BETA
            )
            corrected = local * ratio[None, :]
            corrected /= corrected.sum(1, keepdims=True)
            mass = eval_prob[erows][:, idx].sum(1, keepdims=True)
            out[np.ix_(erows, idx)] = corrected * mass
    out /= np.maximum(out.sum(1, keepdims=True), EPS)
    return out


def main():
    data = B.load_all()
    y, classes, meta = data["y"], data["classes"], data["meta_train"]
    names = CD.available_experts()
    folds = list(KFold(5, shuffle=True, random_state=OUTER_SEED).split(y))
    base_all = np.zeros((len(y), len(classes)), float)
    real_all = np.zeros_like(base_all)
    null_all = np.zeros_like(base_all)
    fold_rows = []
    pair_log = {}
    for fold, (fit, val) in enumerate(folds):
        seed = VIEWS[fold % len(VIEWS)]
        source_prob = crossfit_source(fit, seed, names, data)
        eval_prob = pooled_one_view(fit, val, seed, names, data)
        pairs = disjoint_pairs(
            source_prob, y[fit], meta.iloc[fit]["Region"].isna().to_numpy(), classes
        )
        real = adjust(
            source_prob, y[fit], meta.iloc[fit], eval_prob, meta.iloc[val],
            classes, pairs,
        )
        null = adjust(
            source_prob, y[fit], meta.iloc[fit], eval_prob, meta.iloc[val],
            classes, pairs, null=True,
        )
        base_all[val], real_all[val], null_all[val] = eval_prob, real, null
        pair_log[str(fold)] = [list(x) for x in pairs]
        bp, rp, np_ = [classes[p.argmax(1)] for p in (eval_prob, real, null)]
        b = np.mean(bp == y[val])
        fold_rows.append({
            "fold": fold, "view": seed, "n": len(val), "baseline": b,
            "candidate": np.mean(rp == y[val]),
            "gain_pt": 100 * (np.mean(rp == y[val]) - b),
            "null": np.mean(np_ == y[val]),
            "null_gain_pt": 100 * (np.mean(np_ == y[val]) - b),
            "changed": int(np.sum(rp != bp)),
            "wins": int(np.sum((rp == y[val]) & (bp != y[val]))),
            "losses": int(np.sum((rp != y[val]) & (bp == y[val]))),
        })
        print(
            f"fold {fold+1}/5 view={seed} gain={fold_rows[-1]['gain_pt']:+.2f} "
            f"null={fold_rows[-1]['null_gain_pt']:+.2f} pairs={pairs}", flush=True,
        )
    pd.DataFrame(fold_rows).to_csv(OUT / "pairwise_folds.csv", index=False)
    (OUT / "pairwise_pairs.json").write_text(json.dumps(pair_log, indent=2) + "\n")
    np.savez_compressed(
        OUT / "pairwise_probabilities.npz", baseline=base_all, candidate=real_all,
        null=null_all, truth=y, classes=classes,
    )
    glia = meta["Region"].isna().to_numpy()
    table = pd.DataFrame([
        CD.metrics("baseline", base_all, base_all, y, classes, glia, "nested_cv"),
        CD.metrics("pairwise_deconvolution", real_all, base_all, y, classes, glia, "nested_cv"),
        CD.metrics("shuffled_confusion_null", null_all, base_all, y, classes, glia, "nested_cv"),
    ])
    table.to_csv(OUT / "pairwise_summary.csv", index=False)
    rows = table.set_index("config")
    real, null = rows.loc["pairwise_deconvolution"], rows.loc["shuffled_confusion_null"]
    fold_df = pd.DataFrame(fold_rows)
    passed = bool(
        real.gain_pt > 0.20 and real.mcnemar_p < 0.05
        and (fold_df.gain_pt > 0).sum() >= 4
        and real.gain_pt - null.gain_pt > 0.15
    )
    decision = {
        "passed": passed,
        "gate": "gain>0.20pt, p<0.05, positive in >=4/5 folds, >0.15pt over null",
        "parameters": {"n_pairs": N_PAIRS, "ridge": RIDGE,
                       "source_tau": SOURCE_TAU, "confusion_tau": CONFUSION_TAU,
                       "beta": BETA},
        "device": "cpu", "test_truth_read": False,
    }
    # This lower-rank method intentionally does not call the test thermometer even when
    # it passes; the parent track can decide whether to combine independent survivors.
    (OUT / "pairwise_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nVERDICT: " + ("PASS TRAINING-ONLY GATE" if passed else "REJECT"))


if __name__ == "__main__":
    main()
