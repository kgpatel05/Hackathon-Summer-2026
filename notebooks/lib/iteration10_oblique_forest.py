"""Iteration 10 - random-projection oblique forest complement.

ExtraTrees is axis-aligned.  A sparse random projection of the standardized 694-feature
stack turns random gene/probability combinations into axes, so an ordinary tree split in
projected space is an oblique split in the original space.  This tests one frozen 384-D
projection forest and a fixed blend:

    0.80 * adopted ExtraTrees + 0.20 * projected ExtraTrees

Projection and scaling are fitted inside each fold.  Screen uses partition seed 443 and
five seeds for both forests.  Advance only if the blend gains >0.30 points with paired
exact McNemar p<0.05.  Confirm on untouched partition seed 461 with 20 seeds; adopt only
for >0.20 points and p<0.05.  No test label is read.

Usage:
    python3 notebooks/lib/iteration10_oblique_forest.py screen
    python3 notebooks/lib/iteration10_oblique_forest.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.45
BLEND_WEIGHT = 0.20
N_COMPONENTS = 384
SCREEN_PARTITION = 443
CONFIRM_PARTITION = 461


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = tuple(range(5)) if mode == "screen" else tuple(range(20))
    print(f"mode={mode} partition={partition} x={x.shape} projection={N_COMPONENTS} "
          f"seeds={len(seeds)}", flush=True)

    base_oof = np.zeros((len(y), len(classes)), np.float32)
    oblique_oof = np.zeros_like(base_oof)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        t0 = time.time()
        base = M.fit_extra_trees(
            x[train], pd.Series(y[train]), classes, x[valid], seeds=seeds
        )
        base_oof[valid] = M.correct_prior(
            base, M.prior_vector(pd.Series(y[train]), classes), ALPHA
        )
        scaler = StandardScaler().fit(x[train])
        projector = SparseRandomProjection(
            n_components=N_COMPONENTS, density="auto", random_state=20260819
        )
        projected_train = projector.fit_transform(scaler.transform(x[train])).astype(np.float32)
        projected_valid = projector.transform(scaler.transform(x[valid])).astype(np.float32)
        oblique = M.fit_extra_trees(
            projected_train, pd.Series(y[train]), classes, projected_valid, seeds=seeds
        )
        oblique_oof[valid] = M.correct_prior(
            oblique, M.prior_vector(pd.Series(y[train]), classes), ALPHA
        )
        print(f"fold {fold}/5 finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * base_oof + BLEND_WEIGHT * oblique_oof
    configurations = {
        "ExtraTrees incumbent": base_oof,
        "oblique forest standalone": oblique_oof,
        "0.80 axis + 0.20 oblique": blend,
    }
    base_ok = class_array[base_oof.argmax(1)] == y
    rows = []
    for name, probabilities in configurations.items():
        correct = class_array[probabilities.argmax(1)] == y
        gain = correct.mean() - base_ok.mean()
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_ok)
            wins = int((correct & ~base_ok).sum())
            losses = int((base_ok & ~correct).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": correct.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:29s} acc={correct.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    candidate = rows[2]
    if mode == "screen":
        passed = candidate["gain_pt"] > 0.30 and candidate["p"] < 0.05
        verdict = "ADVANCE TO CONFIRM" if passed else "REJECT"
    else:
        passed = candidate["gain_pt"] > 0.20 and candidate["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
    path = OUT / f"oblique_forest_{mode}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    np.savez_compressed(OUT / f"oblique_forest_{mode}_oof.npz", base=base_oof,
                        oblique=oblique_oof, y=y, classes=class_array)
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
