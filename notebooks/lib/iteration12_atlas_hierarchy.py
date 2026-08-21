"""Iteration 12 - atlas-defined coarse-to-fine hierarchy on the 694-feature stack.

The old iteration-3 hierarchy predated every parent-atlas block and lost through data
starvation.  The current stack has a 14-way atlas coarse-cluster posterior, local atlas
composition, and a metadata-conditioned atlas model.  This gate tests whether those new
signals are better decoded explicitly:

  P(cell type | x) = P(atlas coarse cluster | x)
                       * P(cell type | x, atlas coarse cluster).

The fine->coarse map is read from non-challenge atlas annotations and is effectively
deterministic.  Every fitted model sees only the training side of a frozen 80/20 split
(seed 613).  Advance only if a fixed 80/20 global/hierarchical blend gains >0.30 points.
No test label or withheld gene is read, and this script never writes a submission.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
PARTITION = 613
ALPHA = 0.45
BLEND_WEIGHT = 0.20
SEEDS = tuple(range(5))


def atlas_coarse_map(classes: list[str]) -> dict[str, str]:
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        fine_categories = [x.decode() for x in
                           handle["obs/MERFISH cell type annotation/categories"][:]]
        fine_codes = handle["obs/MERFISH cell type annotation/codes"][:]
        coarse_categories = [x.decode() for x in
                             handle["obs/1st round cluster/categories"][:]]
        coarse_codes = handle["obs/1st round cluster/codes"][:]

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for fine_code, coarse_code in zip(fine_codes, coarse_codes):
        if fine_code < 0 or coarse_code < 0:
            continue
        fine = F._normalise_label(fine_categories[fine_code])
        counts[fine][coarse_categories[coarse_code]] += 1
    mapping = {fine: max(groups, key=groups.get) for fine, groups in counts.items()}
    missing = sorted(set(classes) - set(mapping))
    if missing:
        raise ValueError(f"classes absent from atlas hierarchy: {missing}")
    purity = {
        fine: max(groups.values()) / sum(groups.values()) for fine, groups in counts.items()
        if fine in classes
    }
    print(f"atlas map: {len(set(mapping[c] for c in classes))} coarse groups; "
          f"minimum fine->coarse purity={min(purity.values()):.5f}", flush=True)
    return {c: mapping[c] for c in classes}


def fit_probabilities(x_train: np.ndarray, y_train: np.ndarray,
                      x_eval: np.ndarray, classes: list[str],
                      seeds: tuple[int, ...]) -> np.ndarray:
    out = np.zeros((len(x_eval), len(classes)), np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        ).fit(x_train, y_train)
        out += M.align_proba(model, x_eval, classes)
    return out / len(seeds)


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    mapping = atlas_coarse_map(classes)
    coarse_classes = sorted(set(mapping.values()))
    coarse_y = np.asarray([mapping[label] for label in y])
    coarse_index = {name: j for j, name in enumerate(coarse_classes)}

    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    train, valid = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=PARTITION, stratify=y
    )
    print(f"split={len(train)}/{len(valid)} features={x.shape[1]} seeds={len(SEEDS)}",
          flush=True)

    t0 = time.time()
    global_p = fit_probabilities(x[train], y[train], x[valid], classes, SEEDS)
    global_p = M.correct_prior(global_p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
    print(f"global model {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    coarse_p = fit_probabilities(
        x[train], coarse_y[train], x[valid], coarse_classes, SEEDS
    )
    coarse_p = M.correct_prior(
        coarse_p, M.prior_vector(pd.Series(coarse_y[train]), coarse_classes), ALPHA
    )
    hierarchical = np.zeros_like(global_p)
    for coarse in coarse_classes:
        member_classes = [c for c in classes if mapping[c] == coarse]
        target_columns = [classes.index(c) for c in member_classes]
        if len(member_classes) == 1:
            fine_p = np.ones((len(valid), 1), np.float32)
        else:
            rows = train[coarse_y[train] == coarse]
            fine_p = fit_probabilities(x[rows], y[rows], x[valid], member_classes,
                                       tuple(range(3)))
            fine_p = M.correct_prior(
                fine_p, M.prior_vector(pd.Series(y[rows]), member_classes), ALPHA
            )
        hierarchical[:, target_columns] = (
            coarse_p[:, coarse_index[coarse], None] * fine_p
        )
    hierarchical /= np.maximum(hierarchical.sum(1, keepdims=True), 1e-12)
    print(f"coarse + {len(coarse_classes)} specialists {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * global_p + BLEND_WEIGHT * hierarchical
    truth = y[valid]
    base_ok = class_array[global_p.argmax(1)] == truth
    rows = []
    for name, probabilities in {
        "global incumbent": global_p,
        "atlas hierarchy": hierarchical,
        "0.80 global + 0.20 hierarchy": blend,
    }.items():
        ok = class_array[probabilities.argmax(1)] == truth
        if name == "global incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        gain = ok.mean() - base_ok.mean()
        rows.append({"config": name, "accuracy": ok.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:32s} acc={ok.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "atlas_hierarchy_gate.csv", index=False)
    np.savez_compressed(OUT / "atlas_hierarchy_gate.npz", valid=valid,
                        global_p=global_p, hierarchy=hierarchical, truth=truth,
                        classes=class_array)
    passed = rows[-1]["gain_pt"] > 0.30
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
