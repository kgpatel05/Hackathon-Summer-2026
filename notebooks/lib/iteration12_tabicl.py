"""Iteration 12 - TabICLv2 many-class foundation-model gate on Apple MPS.

TabICLv2 is a permissively licensed in-context tabular transformer.  Its v2 checkpoint
was pretrained mainly on tables with <=100 columns, so feature selection is deliberately
fold-scoped: the 100 highest ANOVA-F columns from the adopted 694-feature stack are chosen
using only the fitting rows.  This avoids both the poor inductive match of all 694 columns
and any validation-label leakage.

The first gate is one frozen 80/20 stratified split (seed 557), four TabICL ensemble
members, and the incumbent five-seed ExtraTrees.  Advance only if TabICL is >=78% alone
and a fixed 80/20 ET/TabICL blend gains >0.30 percentage points.  No hidden label is read
and this script cannot write the submission.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
# Load the repository numerical stack before exposing the locally installed package,
# whose wheel directory may also contain unrelated dependency versions.
sys.path.insert(0, str(ROOT / ".deps"))
from tabicl import TabICLClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
PARTITION = 557
N_FEATURES = 100
N_ESTIMATORS = 4
ALPHA = 0.45
BLEND_WEIGHT = 0.20


def selected_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores, _ = f_classif(x, y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max)
    return np.argsort(scores, kind="stable")[-N_FEATURES:]


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    labels = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))

    train, valid = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=PARTITION, stratify=y
    )
    columns = selected_columns(x[train], y[train])
    print(f"device=mps split={len(train)}/{len(valid)} selected={len(columns)}/{x.shape[1]} "
          f"estimators={N_ESTIMATORS}", flush=True)

    t0 = time.time()
    et = M.fit_extra_trees(x[train], pd.Series(y[train]), classes, x[valid],
                           seeds=tuple(range(5)))
    et = M.correct_prior(et, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
    print(f"ExtraTrees finished in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model = TabICLClassifier(
        n_estimators=N_ESTIMATORS,
        support_many_classes=True,
        batch_size=1,
        device="mps",
        use_amp=False,
        offload_mode=False,
        random_state=557,
        verbose=True,
    )
    model.fit(x[train][:, columns], y[train])
    raw = model.predict_proba(x[valid][:, columns])
    tabicl = np.zeros_like(et)
    class_index = {name: j for j, name in enumerate(classes)}
    for j, name in enumerate(model.classes_.astype(str)):
        tabicl[:, class_index[name]] = raw[:, j]
    print(f"TabICLv2 finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et + BLEND_WEIGHT * tabicl
    truth = y[valid]
    base_ok = labels[et.argmax(1)] == truth
    rows = []
    for name, probabilities in {
        "ExtraTrees incumbent": et,
        "TabICLv2": tabicl,
        "0.80 ET + 0.20 TabICLv2": blend,
    }.items():
        ok = labels[probabilities.argmax(1)] == truth
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        gain = ok.mean() - base_ok.mean()
        rows.append({"config": name, "accuracy": ok.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:28s} acc={ok.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "tabicl_gate.csv", index=False)
    np.savez_compressed(OUT / "tabicl_gate.npz", valid=valid, et=et, tabicl=tabicl,
                        truth=truth, classes=labels, selected=columns)
    passed = rows[1]["accuracy"] >= 0.78 and rows[2]["gain_pt"] > 0.30
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
