"""Fresh full-OOF confirmation of the fixed Iteration-13 scANVI blend.

The gate at split seed 733 selected exactly one candidate: 80% adopted ExtraTrees
probabilities plus 20% scANVI probabilities.  This script makes one prediction for
each training cell under a fresh five-fold partition (seed 997).  Each fold rebuilds
the AnnData object so that its 1,000 validation labels and all 5,000 real-test labels
are literal ``Unknown`` values before scVI/scANVI setup.  Atlas sampling, network,
epochs, balanced sampler, MPS backend, metadata mask, and blend weight are frozen from
the gate.

Confirm only if the fixed blend gains more than 0.30 point and exact paired McNemar
p < 0.05.  Otherwise reject.  This script never reads recovered labels and never
writes ``prediction/prediction.csv``.

Usage:
    MPLCONFIGDIR=/tmp/merfish-mpl NUMBA_CACHE_DIR=/tmp/merfish-numba \
      python3 notebooks/lib/iteration13_scanvi_confirm.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps_scvi"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scvi.model import SCANVI, SCVI

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration13_scanvi as I13
from iteration10_spatial_fields import current_stack


OUT = ROOT / "outputs/iteration13"
OUT.mkdir(parents=True, exist_ok=True)
FOLD_SEED = 997
BLEND_WEIGHT = 0.20


def fit_fold(
    fold: int,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    counts_train: pd.DataFrame,
    meta_train: pd.DataFrame,
    counts_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    classes: list[str],
) -> np.ndarray:
    """Fit one leakage-isolated scANVI model and predict its outer-valid rows."""
    cache = OUT / f"scanvi_confirm_fold{fold}.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=True)
        if np.array_equal(saved["valid"], valid_rows) and list(
            saved["classes"].astype(str)
        ) == classes:
            print(f"fold {fold}: loaded {cache}", flush=True)
            return saved["probabilities"].astype(np.float32)
        raise ValueError(f"stale confirmation cache: {cache}")

    model_seed = FOLD_SEED * 10 + fold
    I13.seed_all(model_seed)
    adata, _, valid_indices, _ = I13.build_anndata(
        counts_train, meta_train, counts_test, meta_test, train_rows, valid_rows
    )
    SCVI.setup_anndata(adata, layer="counts", labels_key="cell_type")
    vae = SCVI(
        adata,
        n_hidden=128,
        n_latent=I13.LATENT_DIM,
        n_layers=2,
        dropout_rate=0.10,
        gene_likelihood="nb",
    )
    t0 = time.time()
    vae.train(
        max_epochs=I13.SCVI_EPOCHS,
        accelerator="mps",
        devices=1,
        batch_size=I13.BATCH_SIZE,
        train_size=0.95,
        check_val_every_n_epoch=5,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    scanvi = SCANVI.from_scvi_model(
        vae,
        unlabeled_category=I13.UNKNOWN,
        labels_key="cell_type",
        linear_classifier=False,
    )
    scanvi.train(
        max_epochs=I13.SCANVI_EPOCHS,
        n_samples_per_label=64,
        accelerator="mps",
        devices=1,
        batch_size=I13.BATCH_SIZE,
        train_size=0.95,
        check_val_every_n_epoch=5,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    probabilities = I13.align_soft_predictions(
        scanvi.predict(indices=valid_indices, soft=True, batch_size=I13.BATCH_SIZE),
        classes,
    )
    np.savez_compressed(
        cache,
        valid=valid_rows,
        probabilities=probabilities,
        classes=np.asarray(classes),
        model_seed=model_seed,
    )
    print(
        f"fold {fold}: scVI+scANVI complete in {time.time()-t0:.1f}s; wrote {cache}",
        flush=True,
    )
    del scanvi, vae, adata
    gc.collect()
    torch.mps.empty_cache()
    return probabilities


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run with normal macOS GPU access")
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    stack = current_stack(meta_all, classes, list(counts_train.columns))
    folds = list(StratifiedKFold(
        5, shuffle=True, random_state=FOLD_SEED
    ).split(stack, y))
    et_oof = np.zeros((len(y), len(classes)), np.float32)
    scanvi_oof = np.zeros_like(et_oof)
    print(
        f"fresh confirmation partition={FOLD_SEED} folds=5 blend={BLEND_WEIGHT:.2f} "
        f"features={stack.shape[1]} device=mps",
        flush=True,
    )
    t0 = time.time()

    for fold, (train_rows, valid_rows) in enumerate(folds, start=1):
        neural = fit_fold(
            fold,
            train_rows,
            valid_rows,
            counts_train,
            meta_train,
            counts_test,
            meta_test,
            classes,
        )
        scanvi_oof[valid_rows] = I13.masked_probabilities(
            neural,
            meta_train.iloc[train_rows],
            y[train_rows],
            meta_train.iloc[valid_rows],
            classes,
        )
        et = M.fit_extra_trees(
            stack[train_rows],
            pd.Series(y[train_rows]),
            classes,
            stack[valid_rows],
            seeds=I13.ET_SEEDS,
        )
        et = M.correct_prior(
            et, M.prior_vector(pd.Series(y[train_rows]), classes), I13.ALPHA
        )
        et_oof[valid_rows] = I13.masked_probabilities(
            et,
            meta_train.iloc[train_rows],
            y[train_rows],
            meta_train.iloc[valid_rows],
            classes,
        )
        print(f"fold {fold}/5 complete; elapsed={time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et_oof + BLEND_WEIGHT * scanvi_oof
    glia = meta_train["Region"].isna().to_numpy()
    base_correct = class_array[et_oof.argmax(axis=1)] == y
    rows = []
    for name, probabilities in {
        "ExtraTrees incumbent": et_oof,
        "scANVI standalone": scanvi_oof,
        "0.80 ET + 0.20 scANVI": blend,
    }.items():
        correct = class_array[probabilities.argmax(axis=1)] == y
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_correct)
            wins = int((correct & ~base_correct).sum())
            losses = int((base_correct & ~correct).sum())
        rows.append({
            "config": name,
            "accuracy": correct.mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "glia": correct[glia].mean(),
            "neurons": correct[~glia].mean(),
            "wins": wins,
            "losses": losses,
            "p": p_value,
            "fold_seed": FOLD_SEED,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "scanvi_confirm_oof.csv", index=False)
    np.savez_compressed(
        OUT / "scanvi_confirm_oof.npz",
        et=et_oof,
        scanvi=scanvi_oof,
        blend=blend,
        truth=y,
        classes=class_array,
        fold_seed=FOLD_SEED,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    candidate = rows[-1]
    confirmed = candidate["gain_pt"] > 0.30 and candidate["p"] < 0.05
    print(
        "VERDICT: " + ("CONFIRMED" if confirmed else "REJECT"),
        flush=True,
    )


if __name__ == "__main__":
    main()
