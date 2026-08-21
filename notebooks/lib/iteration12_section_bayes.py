"""Iteration 12 - fold-scoped section-composition Bayesian correction.

Every tissue section occurs in both challenge halves.  A section's labelled
training cells therefore provide a legal estimate of its cell-type composition
for unlabelled cells in that same section.  Earlier work appended this profile as
tree features; this experiment instead applies the generative label-shift rule

    p(y | x, section) ∝ p(y | x) [p(y | section) / p(y)] ** beta.

Beta is selected using only three-fold OOF predictions inside the frozen outer
training partition.  Profiles are always computed from the corresponding fit
rows.  A permutation of profile-to-section assignments is a matched null.  No
test label is read and no submission is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
GATE = OUT / "tabicl_gate.npz"
BETAS = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
SMOOTH = 20.0


def section_ratio(fit_sections, fit_y, eval_sections, classes, *, permuted=False, seed=0):
    index = {label: j for j, label in enumerate(classes)}
    y_idx = np.asarray([index[label] for label in fit_y])
    global_prior = np.bincount(y_idx, minlength=len(classes)).astype(float)
    global_prior = (global_prior + 0.5) / (len(y_idx) + 0.5*len(classes))
    section_names = np.asarray(sorted(set(fit_sections)))
    table = {}
    for name in section_names:
        rows = fit_sections == name
        counts = np.bincount(y_idx[rows], minlength=len(classes)).astype(float)
        table[name] = (counts + SMOOTH*global_prior) / (rows.sum() + SMOOTH)
    if permuted:
        shuffled = section_names.copy()
        np.random.default_rng(seed).shuffle(shuffled)
        table = {name: table[other] for name, other in zip(section_names, shuffled)}
    profiles = np.asarray([table.get(name, global_prior) for name in eval_sections])
    return np.clip(profiles / global_prior[None, :], 1e-4, 1e4).astype(np.float32)


def adjust(probabilities, ratio, beta):
    out = probabilities * np.power(ratio, beta)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def metadata_mask(probabilities, train_meta, train_y, eval_meta, classes):
    allow = np.ones_like(probabilities, dtype=bool)
    for column in ("Region", "Excitatory_vs_Inhibitory", "Segment"):
        fit_values = train_meta[column].astype(str).to_numpy()
        eval_values = eval_meta[column].astype(str).to_numpy()
        known = set(fit_values)
        seen = [set(fit_values[train_y == label]) for label in classes]
        for i, value in enumerate(eval_values):
            if value in known:
                allow[i] &= np.asarray([value in values for values in seen])
    allow[~allow.any(axis=1)] = True
    out = np.where(allow, probabilities, 0.0)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def accuracy(probabilities, truth, classes):
    return np.mean(np.asarray(classes)[probabilities.argmax(axis=1)] == truth)


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    sections = meta["Section_ID"].astype(str).to_numpy()
    glia = meta["Region"].isna().to_numpy()
    gate = np.load(GATE, allow_pickle=True)
    valid = gate["valid"].astype(int)
    train = np.setdiff1d(np.arange(len(y)), valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes.tolist(), list(counts.columns))
    print(f"outer={len(train)}/{len(valid)} sections={len(np.unique(sections))} "
          f"features={x.shape[1]} betas={BETAS}", flush=True)

    # Inner OOF predictions and fold-pure section ratios.
    oof = np.zeros((len(train), len(classes)), np.float32)
    real_ratio = np.ones_like(oof)
    null_ratio = np.ones_like(oof)
    folds = StratifiedKFold(3, shuffle=True, random_state=127)
    t0 = time.time()
    for fold, (itr, iva) in enumerate(folds.split(train, y[train])):
        fit_rows, score_rows = train[itr], train[iva]
        probabilities = M.fit_extra_trees(
            x[fit_rows], pd.Series(y[fit_rows]), classes.tolist(), x[score_rows],
            seeds=(0, 1, 2),
        )
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[fit_rows]), classes.tolist()), 0.45
        )
        oof[iva] = metadata_mask(
            probabilities, meta.iloc[fit_rows], y[fit_rows], meta.iloc[score_rows], classes
        )
        real_ratio[iva] = section_ratio(
            sections[fit_rows], y[fit_rows], sections[score_rows], classes
        )
        null_ratio[iva] = section_ratio(
            sections[fit_rows], y[fit_rows], sections[score_rows], classes,
            permuted=True, seed=911+fold,
        )
        print(f"inner fold {fold+1}/3 complete", flush=True)

    rows = []
    for beta in BETAS:
        rows.append({
            "beta": beta,
            "real_accuracy": accuracy(adjust(oof, real_ratio, beta), y[train], classes),
            "null_accuracy": accuracy(adjust(oof, null_ratio, beta), y[train], classes),
        })
    tuning = pd.DataFrame(rows)
    tuning.to_csv(OUT / "section_bayes_inner_cv.csv", index=False)
    # Stable tie break favours the smaller departure from the incumbent.
    best_beta = float(tuning.sort_values(
        ["real_accuracy", "beta"], ascending=[False, True], kind="stable"
    ).iloc[0]["beta"])
    print(tuning.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"selected beta={best_beta:g} in {time.time()-t0:.1f}s", flush=True)

    outer = metadata_mask(
        gate["et"].astype(np.float32), meta.iloc[train], y[train], meta.iloc[valid], classes
    )
    outer_real_ratio = section_ratio(
        sections[train], y[train], sections[valid], classes
    )
    outer_null_ratio = section_ratio(
        sections[train], y[train], sections[valid], classes, permuted=True, seed=911
    )
    probabilities = {
        "masked ExtraTrees incumbent": outer,
        "section Bayes": adjust(outer, outer_real_ratio, best_beta),
        "permuted-section null": adjust(outer, outer_null_ratio, best_beta),
    }
    truth = y[valid]
    base_ok = classes[outer.argmax(axis=1)] == truth
    results = []
    correctness = {}
    for name, matrix in probabilities.items():
        ok = classes[matrix.argmax(axis=1)] == truth
        correctness[name] = ok
        if name == "masked ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        results.append({"config": name, "accuracy": ok.mean(),
                        "gain_pt": 100*(ok.mean()-base_ok.mean()),
                        "glia": ok[glia[valid]].mean(),
                        "neurons": ok[~glia[valid]].mean(),
                        "wins": wins, "losses": losses, "p": p_value,
                        "beta": best_beta})
    result = pd.DataFrame(results)
    result.to_csv(OUT / "section_bayes_gate.csv", index=False)
    np.savez_compressed(OUT / "section_bayes_gate.npz", valid=valid,
                        baseline=outer, real=probabilities["section Bayes"],
                        null=probabilities["permuted-section null"], truth=truth,
                        classes=classes, beta=best_beta)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    real = result.iloc[1]
    null = result.iloc[2]
    passed = (real["gain_pt"] > 0.30 and real["p"] < 0.05
              and real["gain_pt"] > null["gain_pt"] and null["gain_pt"] <= 0.30)
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
