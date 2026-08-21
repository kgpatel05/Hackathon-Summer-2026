"""Iteration 10 - LightGBM complementarity on the adopted 694-feature stack.

The repository tested XGBoost and sklearn HistGradientBoosting on an older stack but
explicitly did not test LightGBM.  This is one frozen, strongly regularised leaf-wise
configuration plus one fixed probability blend:

    0.80 * adopted ExtraTrees + 0.20 * LightGBM

Screen uses partition seed 373 and five ExtraTrees seeds.  Advance only if the blend
gains >0.30 points with paired exact McNemar p<0.05.  Confirmation uses untouched seed
389 and 20 ExtraTrees seeds; adopt only for >0.20 points and p<0.05.  No test label is
read.

Usage:
    python3 notebooks/lib/iteration10_lightgbm.py screen
    python3 notebooks/lib/iteration10_lightgbm.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.45
BLEND_WEIGHT = 0.20
SCREEN_PARTITION = 373
CONFIRM_PARTITION = 389


def main(mode: str) -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y_text = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y_text))
    class_array = np.asarray(classes)
    class_index = {name: i for i, name in enumerate(classes)}
    y = np.asarray([class_index[name] for name in y_text], np.int64)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    et_seeds = tuple(range(5)) if mode == "screen" else tuple(range(20))
    print(f"mode={mode} partition={partition} x={x.shape} ET={len(et_seeds)}", flush=True)

    et_oof = np.zeros((len(y), len(classes)), np.float32)
    lgb_oof = np.zeros_like(et_oof)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for fold, (train, valid) in enumerate(folds.split(y, y), 1):
        t0 = time.time()
        et = M.fit_extra_trees(
            x[train], pd.Series(y_text[train]), classes, x[valid], seeds=et_seeds
        )
        et_oof[valid] = M.correct_prior(
            et, M.prior_vector(pd.Series(y_text[train]), classes), ALPHA
        )
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=6.0,
            random_state=0,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x[train], y[train])
        raw = model.predict_proba(x[valid])
        for j, code in enumerate(model.classes_.astype(int)):
            lgb_oof[valid, code] = raw[:, j]
        print(f"fold {fold}/5 finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et_oof + BLEND_WEIGHT * lgb_oof
    configurations = {
        "ExtraTrees incumbent": et_oof,
        "LightGBM standalone": lgb_oof,
        "0.80 ET + 0.20 LightGBM": blend,
    }
    base_ok = class_array[et_oof.argmax(1)] == y_text
    rows = []
    for name, probabilities in configurations.items():
        correct = class_array[probabilities.argmax(1)] == y_text
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
    path = OUT / f"lightgbm_{mode}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    np.savez_compressed(OUT / f"lightgbm_{mode}_oof.npz", et=et_oof, lightgbm=lgb_oof,
                        y=y_text, classes=class_array)
    print(f"VERDICT: {verdict}; wrote {path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
