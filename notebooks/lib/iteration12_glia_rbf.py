"""Iteration 12 - nonlinear RBF-kernel specialist for non-neuronal phenotypes.

Earlier glia sweeps covered linear/logistic, tree, boosting, kNN and neural
models, but not a nonlinear margin classifier.  The released ``Region`` field
identifies non-neuronal cells without a target label, so this gate leaves every
neuronal call untouched and reclassifies only those cells with an RBF SVM.

Hyperparameters are selected by three-fold CV entirely inside the 4,000-row
outer training partition.  The frozen seed-557 outer 1,000 rows are used once.
Advance only for >0.30-point overall gain, positive glia gain, and p < 0.05.
No test label is read and no submission is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
GATE = OUT / "tabicl_gate.npz"

# Compact, predeclared kernel family.  Scaling makes gamma = multiplier / p.
CONFIGS = [
    (c, gamma, weight)
    for c in (1.0, 4.0, 16.0)
    for gamma in (0.25, 1.0)
    for weight in (None, "balanced")
]


def make_svc(n_features: int, c: float, gamma: float, weight: str | None):
    return make_pipeline(
        StandardScaler(),
        SVC(C=c, gamma=gamma/n_features, class_weight=weight,
            kernel="rbf", cache_size=4096, decision_function_shape="ovr"),
    )


def metadata_mask(probabilities, train_meta, train_y, eval_meta, classes):
    allow = np.ones((len(eval_meta), len(classes)), bool)
    for column in ("Region", "Excitatory_vs_Inhibitory", "Segment"):
        tr_values = train_meta[column].astype(str).to_numpy()
        ev_values = eval_meta[column].astype(str).to_numpy()
        known = set(tr_values)
        seen = [set(tr_values[train_y == label]) for label in classes]
        for i, value in enumerate(ev_values):
            if value in known:
                allow[i] &= np.asarray([value in values for values in seen])
    allow[~allow.any(axis=1)] = True
    return np.where(allow, probabilities, -1.0)


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    gate = np.load(GATE, allow_pickle=True)
    valid = gate["valid"].astype(int)
    train = np.setdiff1d(np.arange(len(y)), valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes.tolist(), list(counts.columns))
    glia = meta["Region"].isna().to_numpy()
    glia_train = train[glia[train]]
    glia_valid = valid[glia[valid]]
    print(f"outer={len(train)}/{len(valid)} glia={len(glia_train)}/{len(glia_valid)} "
          f"features={x.shape[1]} configs={len(CONFIGS)}", flush=True)

    # Select the kernel only within the outer training partition.
    folds = StratifiedKFold(3, shuffle=True, random_state=119)
    cv_rows = []
    t0 = time.time()
    for c, gamma, weight in CONFIGS:
        predictions = np.empty(len(glia_train), dtype=object)
        for inner_train, inner_valid in folds.split(glia_train, y[glia_train]):
            fit_rows = glia_train[inner_train]
            score_rows = glia_train[inner_valid]
            model = make_svc(x.shape[1], c, gamma, weight)
            model.fit(x[fit_rows], y[fit_rows])
            predictions[inner_valid] = model.predict(x[score_rows])
        cv_rows.append({
            "C": c, "gamma_multiplier": gamma,
            "class_weight": "none" if weight is None else weight,
            "accuracy": accuracy_score(y[glia_train], predictions),
            "balanced_accuracy": balanced_accuracy_score(y[glia_train], predictions),
        })
        print(f"C={c:>4g} gamma={gamma:>4g}/p weight={str(weight):>8s} "
              f"acc={cv_rows[-1]['accuracy']:.4f} bal={cv_rows[-1]['balanced_accuracy']:.4f}",
              flush=True)
    cv = pd.DataFrame(cv_rows).sort_values(
        ["accuracy", "balanced_accuracy"], ascending=False, kind="stable"
    )
    cv.to_csv(OUT / "glia_rbf_inner_cv.csv", index=False)
    best = cv.iloc[0]
    best_weight = None if best["class_weight"] == "none" else "balanced"
    print(f"selected C={best['C']:g} gamma={best['gamma_multiplier']:g}/p "
          f"weight={best_weight} in {time.time()-t0:.1f}s", flush=True)

    specialist = make_svc(
        x.shape[1], float(best["C"]), float(best["gamma_multiplier"]), best_weight
    )
    specialist.fit(x[glia_train], y[glia_train])

    et = gate["et"].astype(np.float32)
    masked = metadata_mask(et, meta.iloc[train], y[train], meta.iloc[valid], classes)
    baseline = classes[masked.argmax(axis=1)]
    candidate = baseline.copy()
    candidate[np.flatnonzero(glia[valid])] = specialist.predict(x[glia_valid])
    truth = y[valid]
    base_ok = baseline == truth
    cand_ok = candidate == truth
    p_value, _ = M.paired_mcnemar(cand_ok, base_ok)
    wins = int((cand_ok & ~base_ok).sum())
    losses = int((base_ok & ~cand_ok).sum())
    result = pd.DataFrame([{
        "baseline_accuracy": base_ok.mean(),
        "candidate_accuracy": cand_ok.mean(),
        "gain_pt": 100*(cand_ok.mean()-base_ok.mean()),
        "baseline_glia": base_ok[glia[valid]].mean(),
        "candidate_glia": cand_ok[glia[valid]].mean(),
        "baseline_neuron": base_ok[~glia[valid]].mean(),
        "candidate_neuron": cand_ok[~glia[valid]].mean(),
        "wins": wins, "losses": losses, "p": p_value,
        "C": best["C"], "gamma_multiplier": best["gamma_multiplier"],
        "class_weight": best["class_weight"],
    }])
    result.to_csv(OUT / "glia_rbf_gate.csv", index=False)
    np.savez_compressed(OUT / "glia_rbf_gate.npz", valid=valid,
                        baseline=baseline, candidate=candidate, truth=truth,
                        classes=classes)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    passed = (result.loc[0, "gain_pt"] > 0.30
              and result.loc[0, "candidate_glia"] > result.loc[0, "baseline_glia"]
              and p_value < 0.05)
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
