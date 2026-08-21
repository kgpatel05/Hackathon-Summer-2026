"""Iteration 13 - scANVI atlas-to-challenge semi-supervised count model.

This is not the custom ZINB CVAE from Iteration 10.  It uses the maintained scANVI
model from scvi-tools, initializes its classifier from an unsupervised scVI count
model, trains on a much larger and class-balanced parent-atlas sample, and treats the
challenge validation and test cells as explicitly unlabeled query cells.

Competition boundary
--------------------
* expression is restricted to the 200 released challenge genes;
* every one of the 10,000 challenge cells is removed from the labelled atlas pool;
* only the outer-training challenge labels are copied into the AnnData object;
* validation and test labels are the literal string ``Unknown`` before setup/training;
* no recovered-label artifact is imported or opened;
* this gate writes evidence under ``outputs/iteration13`` and never writes a submission.

The gate tests two different ways for scANVI to complement the adopted 694-feature
ExtraTrees model: a fixed probability blend and its latent representation as an
additional feature block.  A within-section shuffled latent block is the width-matched
control.  The frozen 80/20 split and all candidates are declared here before results:

* split seed 733; model seed 733; 60,000 non-challenge atlas cells;
* 20 scVI epochs followed by 20 scANVI epochs, 20 latent dimensions;
* blend weights {0.10, 0.20, 0.30};
* advance only for a gain above 0.30 point, and for the latent arm additionally a
  gain at least 0.20 point above its shuffled control.

Any survivor requires fresh full-OOF confirmation before it can be considered for the
submission.  The production prediction is deliberately out of reach of this script.

Usage (Apple Silicon; scvi-tools lives in the ignored ``.deps_scvi`` directory):
    MPLCONFIGDIR=/tmp/merfish-mpl NUMBA_CACHE_DIR=/tmp/merfish-numba \
      python3 notebooks/lib/iteration13_scanvi.py gate
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Import the workspace PyTorch build before prepending the isolated dependency tree.
# This preserves the matching macOS torchvision build and MPS support.
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps_scvi"))

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from scvi.model import SCANVI, SCVI

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_atlas_model as I9A
import iteration9_quota as Q
from iteration10_spatial_fields import current_stack


OUT = ROOT / "outputs/iteration13"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_SEED = 733
ATLAS_SAMPLE = 60_000
SCVI_EPOCHS = 20
SCANVI_EPOCHS = 20
LATENT_DIM = 20
BATCH_SIZE = 1024
ET_SEEDS = tuple(range(5))
ALPHA = 0.45
BLEND_WEIGHTS = (0.10, 0.20, 0.30)
UNKNOWN = "Unknown"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_anndata(
    counts_train: pd.DataFrame,
    meta_train: pd.DataFrame,
    counts_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    outer_train: np.ndarray,
    outer_valid: np.ndarray,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    """Build the labelled-atlas + partially labelled challenge AnnData object."""
    classes = sorted(meta_train[F.TARGET].astype(str).unique())
    expression, atlas_labels, _, challenge = I9A.load_atlas(
        list(counts_train.columns), meta_train, meta_test
    )
    usable = (~challenge) & np.isin(atlas_labels, classes) & (expression.sum(1) > 0)
    atlas_rows = np.flatnonzero(usable)
    if len(atlas_rows) > ATLAS_SAMPLE:
        atlas_rows, _ = train_test_split(
            atlas_rows,
            train_size=ATLAS_SAMPLE,
            random_state=SPLIT_SEED,
            stratify=atlas_labels[atlas_rows],
        )

    challenge_counts = np.vstack([
        counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32)
    ])
    matrix = sparse.csr_matrix(np.vstack([
        expression[atlas_rows].astype(np.float32), challenge_counts
    ]))
    labels = np.full(matrix.shape[0], UNKNOWN, dtype=object)
    labels[: len(atlas_rows)] = atlas_labels[atlas_rows]
    challenge_offset = len(atlas_rows)
    train_labels = meta_train[F.TARGET].astype(str).to_numpy()
    labels[challenge_offset + outer_train] = train_labels[outer_train]

    # Validation and real test rows remain Unknown by construction.  These assertions
    # are the central leakage guard and intentionally precede scvi-tools setup.
    valid_indices = challenge_offset + outer_valid
    test_indices = challenge_offset + len(meta_train) + np.arange(len(meta_test))
    if not np.all(labels[valid_indices] == UNKNOWN):
        raise AssertionError("outer-validation label entered the scANVI object")
    if not np.all(labels[test_indices] == UNKNOWN):
        raise AssertionError("test label entered the scANVI object")

    obs = pd.DataFrame({
        "cell_type": pd.Categorical(labels, categories=[*classes, UNKNOWN]),
        "domain": np.concatenate([
            np.full(len(atlas_rows), "atlas", dtype=object),
            np.full(len(challenge_counts), "challenge", dtype=object),
        ]),
    })
    adata = ad.AnnData(X=matrix, obs=obs)
    adata.var_names = counts_train.columns.astype(str)
    adata.layers["counts"] = adata.X.copy()
    train_indices = challenge_offset + outer_train
    print(
        f"AnnData={adata.shape} atlas={len(atlas_rows)} "
        f"challenge_labeled={len(train_indices)} valid_unknown={len(valid_indices)} "
        f"test_unknown={len(test_indices)}",
        flush=True,
    )
    return adata, train_indices, valid_indices, test_indices


def align_soft_predictions(soft, classes: list[str]) -> np.ndarray:
    """Return scANVI probabilities in the challenge class order."""
    if isinstance(soft, tuple):
        soft = soft[0]
    if isinstance(soft, pd.DataFrame):
        missing = sorted(set(classes) - set(soft.columns.astype(str)))
        if missing:
            raise ValueError(f"scANVI output misses classes: {missing}")
        return soft.loc[:, classes].to_numpy(np.float32)
    array = np.asarray(soft, np.float32)
    if array.shape[1] != len(classes):
        raise ValueError(
            f"unlabelled scANVI probability array has {array.shape[1]} columns; "
            f"expected {len(classes)}"
        )
    return array


def section_shuffled(block: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Width-matched null retaining the section-wise latent distribution."""
    rng = np.random.default_rng(SPLIT_SEED)
    output = block.copy()
    section = meta["Section_ID"].astype(str).to_numpy()
    for value in np.unique(section):
        rows = np.flatnonzero(section == value)
        output[rows] = block[rng.permutation(rows)]
    return output


def masked_probabilities(
    probabilities: np.ndarray,
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    meta_eval: pd.DataFrame,
    classes: list[str],
) -> np.ndarray:
    allow = Q.compatibility_mask(meta_train, y_train, meta_eval, classes)
    output = np.where(allow, probabilities, 0.0)
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-12)
    return output.astype(np.float32)


def evaluate(
    name: str,
    probabilities: np.ndarray,
    truth: np.ndarray,
    classes: list[str],
    baseline_correct: np.ndarray,
    glia: np.ndarray,
) -> dict[str, object]:
    prediction = np.asarray(classes)[probabilities.argmax(axis=1)]
    correct = prediction == truth
    if name == "ExtraTrees incumbent":
        p_value, wins, losses = 1.0, 0, 0
    else:
        p_value, _ = M.paired_mcnemar(correct, baseline_correct)
        wins = int((correct & ~baseline_correct).sum())
        losses = int((baseline_correct & ~correct).sum())
    row = {
        "config": name,
        "accuracy": float(correct.mean()),
        "gain_pt": float(100 * (correct.mean() - baseline_correct.mean())),
        "glia": float(correct[glia].mean()),
        "neurons": float(correct[~glia].mean()),
        "wins": wins,
        "losses": losses,
        "p": float(p_value),
    }
    print(
        f"{name:34s} acc={row['accuracy']:.4f} gain={row['gain_pt']:+.2f}pt "
        f"glia={row['glia']:.4f} {wins}w/{losses}l p={p_value:.5g}",
        flush=True,
    )
    return row


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode != "gate":
        raise SystemExit("usage: iteration13_scanvi.py gate")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process; run with normal macOS GPU access"
        )
    seed_all(SPLIT_SEED)
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    outer_train, outer_valid = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SPLIT_SEED
    ).split(counts_train, y))
    adata, train_indices, valid_indices, _ = build_anndata(
        counts_train, meta_train, counts_test, meta_test, outer_train, outer_valid
    )

    SCVI.setup_anndata(adata, layer="counts", labels_key="cell_type")
    vae = SCVI(
        adata,
        n_hidden=128,
        n_latent=LATENT_DIM,
        n_layers=2,
        dropout_rate=0.10,
        gene_likelihood="nb",
    )
    print(
        f"device=mps scVI_epochs={SCVI_EPOCHS} scANVI_epochs={SCANVI_EPOCHS} "
        f"latent={LATENT_DIM} batch={BATCH_SIZE}",
        flush=True,
    )
    t0 = time.time()
    vae.train(
        max_epochs=SCVI_EPOCHS,
        accelerator="mps",
        devices=1,
        batch_size=BATCH_SIZE,
        train_size=0.95,
        check_val_every_n_epoch=5,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    print(f"scVI complete in {time.time()-t0:.1f}s", flush=True)

    scanvi = SCANVI.from_scvi_model(
        vae,
        unlabeled_category=UNKNOWN,
        labels_key="cell_type",
        linear_classifier=False,
    )
    scanvi.train(
        max_epochs=SCANVI_EPOCHS,
        n_samples_per_label=64,
        accelerator="mps",
        devices=1,
        batch_size=BATCH_SIZE,
        train_size=0.95,
        check_val_every_n_epoch=5,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    print(f"scANVI complete in {time.time()-t0:.1f}s", flush=True)

    scanvi_prob = align_soft_predictions(
        scanvi.predict(indices=valid_indices, soft=True, batch_size=BATCH_SIZE), classes
    )
    latent_train = np.asarray(
        scanvi.get_latent_representation(indices=train_indices, batch_size=BATCH_SIZE),
        np.float32,
    )
    latent_valid = np.asarray(
        scanvi.get_latent_representation(indices=valid_indices, batch_size=BATCH_SIZE),
        np.float32,
    )

    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    stack = current_stack(meta_all, classes, list(counts_train.columns))
    incumbent = M.fit_extra_trees(
        stack[outer_train], pd.Series(y[outer_train]), classes, stack[outer_valid],
        seeds=ET_SEEDS,
    )
    incumbent = M.correct_prior(
        incumbent, M.prior_vector(pd.Series(y[outer_train]), classes), ALPHA
    )
    incumbent = masked_probabilities(
        incumbent, meta_train.iloc[outer_train], y[outer_train],
        meta_train.iloc[outer_valid], classes,
    )
    scanvi_prob = masked_probabilities(
        scanvi_prob, meta_train.iloc[outer_train], y[outer_train],
        meta_train.iloc[outer_valid], classes,
    )

    # The latent candidate and its section-wise shuffled width control use exactly the
    # same ExtraTrees hyperparameters and seeds as the incumbent.
    latent_all = np.zeros((len(meta_train), LATENT_DIM), np.float32)
    latent_all[outer_train] = latent_train
    latent_all[outer_valid] = latent_valid
    shuffled = section_shuffled(latent_all, meta_train)
    latent_prob = M.fit_extra_trees(
        np.hstack([stack[outer_train], latent_train]),
        pd.Series(y[outer_train]), classes,
        np.hstack([stack[outer_valid], latent_valid]), seeds=ET_SEEDS,
    )
    latent_prob = M.correct_prior(
        latent_prob, M.prior_vector(pd.Series(y[outer_train]), classes), ALPHA
    )
    latent_prob = masked_probabilities(
        latent_prob, meta_train.iloc[outer_train], y[outer_train],
        meta_train.iloc[outer_valid], classes,
    )
    null_prob = M.fit_extra_trees(
        np.hstack([stack[outer_train], shuffled[outer_train]]),
        pd.Series(y[outer_train]), classes,
        np.hstack([stack[outer_valid], shuffled[outer_valid]]), seeds=ET_SEEDS,
    )
    null_prob = M.correct_prior(
        null_prob, M.prior_vector(pd.Series(y[outer_train]), classes), ALPHA
    )
    null_prob = masked_probabilities(
        null_prob, meta_train.iloc[outer_train], y[outer_train],
        meta_train.iloc[outer_valid], classes,
    )

    truth = y[outer_valid]
    glia = meta_train.iloc[outer_valid]["Region"].isna().to_numpy()
    baseline_correct = class_array[incumbent.argmax(axis=1)] == truth
    rows = [evaluate(
        "ExtraTrees incumbent", incumbent, truth, classes, baseline_correct, glia
    )]
    rows.append(evaluate(
        "scANVI standalone", scanvi_prob, truth, classes, baseline_correct, glia
    ))
    for weight in BLEND_WEIGHTS:
        blend = (1.0 - weight) * incumbent + weight * scanvi_prob
        rows.append(evaluate(
            f"ET + scANVI blend w={weight:.2f}", blend, truth, classes,
            baseline_correct, glia,
        ))
    rows.append(evaluate(
        "ET + scANVI latent", latent_prob, truth, classes, baseline_correct, glia
    ))
    rows.append(evaluate(
        "ET + shuffled latent (null)", null_prob, truth, classes,
        baseline_correct, glia,
    ))

    frame = pd.DataFrame(rows)
    frame["split_seed"] = SPLIT_SEED
    frame.to_csv(OUT / "scanvi_gate.csv", index=False)
    np.savez_compressed(
        OUT / "scanvi_gate.npz",
        valid=outer_valid,
        truth=truth,
        classes=class_array,
        incumbent=incumbent,
        scanvi=scanvi_prob,
        latent=latent_prob,
        null=null_prob,
    )
    real_latent = next(row for row in rows if row["config"] == "ET + scANVI latent")
    null_latent = next(
        row for row in rows if row["config"] == "ET + shuffled latent (null)"
    )
    blend_best = max(
        (row for row in rows if "blend" in str(row["config"])),
        key=lambda row: float(row["gain_pt"]),
    )
    latent_pass = (
        float(real_latent["gain_pt"]) > 0.30
        and float(real_latent["gain_pt"]) - float(null_latent["gain_pt"]) > 0.20
    )
    blend_pass = float(blend_best["gain_pt"]) > 0.30
    print(
        "VERDICT: "
        + ("ADVANCE TO FRESH FULL-OOF CONFIRMATION" if latent_pass or blend_pass
           else "REJECT"),
        flush=True,
    )
    print(f"wrote {OUT/'scanvi_gate.csv'} and compact probability evidence", flush=True)


if __name__ == "__main__":
    main()
