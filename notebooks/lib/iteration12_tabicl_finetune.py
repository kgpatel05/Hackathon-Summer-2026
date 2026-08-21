"""Iteration 12 - dataset-specific TabICLv2 fine-tuning on Apple MPS.

Zero-shot TabICLv2 reached 75.0% on the frozen seed-557 gate.  The official v2 package
supports single-dataset fine-tuning with an internal validation split and a safety net
that keeps the pretrained checkpoint if adaptation degrades validation performance.
This script reuses the exact outer 80/20 split and fold-scoped 100-feature selection from
the zero-shot gate; the outer 1,000 rows remain untouched until final scoring.

Fixed configuration: 12 epochs, lr=1e-5, internal 10% early-stopping validation,
one training/validation ensemble member and four inference members.  Advance only if the
fine-tuned model reaches >=78% alone and a fixed 80/20 ET/TabICL blend gains >0.30 point.
No test label is read and no submission is written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
from tabicl import FinetunedTabICLClassifier
import tabicl._finetune.data as tabicl_finetune_data
import tabicl._finetune.base as tabicl_finetune_base
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack
import pandas as pd

OUT = Path("outputs/iteration12")
MODEL_OUT = OUT / "tabicl_finetune_model"
OUT.mkdir(parents=True, exist_ok=True)
GATE = OUT / "tabicl_gate.npz"
WEIGHT = 0.20


# TabICL 2.1.1 assumes every random meta-chunk has >=2 rows from every class.
# With 60 imbalanced labels, a chunk can legitimately contain one rare example.  Keep
# those singletons in the in-context side, then stratify the remaining rows normally.
_ORIGINAL_SPLIT = tabicl_finetune_data._split_ctx_query
_ORIGINAL_BUILD_META_BATCH = tabicl_finetune_data._build_meta_batch
_ORIGINAL_ITER_META_BATCHES = tabicl_finetune_base.iter_epoch_meta_batches


def _rare_safe_split(y_chunk, query_size: int, seed: int, stratify: bool):
    if not stratify:
        return _ORIGINAL_SPLIT(
            y_chunk, query_size=query_size, seed=seed, stratify=stratify
        )
    _, inverse, counts = np.unique(y_chunk, return_inverse=True, return_counts=True)
    forced_context = np.flatnonzero(counts[inverse] < 2)
    remaining = np.flatnonzero(counts[inverse] >= 2)
    if len(forced_context) == 0:
        return _ORIGINAL_SPLIT(
            y_chunk, query_size=query_size, seed=seed, stratify=stratify
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=query_size, random_state=seed)
    context_local, query_local = next(
        splitter.split(np.zeros((len(remaining), 1)), y_chunk[remaining])
    )
    return np.r_[forced_context, remaining[context_local]], remaining[query_local]


tabicl_finetune_data._split_ctx_query = _rare_safe_split


def _contiguous_meta_batch(X_chunk, y_chunk, **kwargs):
    """Relabel each random meta-chunk to the 0..K-1 IDs TabICL expects.

    The upstream builder sizes its class permutation from the number of labels
    present in the chunk, but then indexes it with the dataset-wide encoded IDs.
    A random chunk that omits any class therefore fails even though the labels
    are otherwise valid.  Meta-learning labels are exchangeable, so a local,
    one-to-one encoding is the intended representation and changes no target.
    """
    _, y_local = np.unique(y_chunk, return_inverse=True)
    return _ORIGINAL_BUILD_META_BATCH(
        X_chunk, y_local.astype(np.int64, copy=False), **kwargs
    )


tabicl_finetune_data._build_meta_batch = _contiguous_meta_batch


def _ten_way_meta_batches(
    X,
    y,
    *,
    classification,
    n_estimators,
    max_chunk_size,
    query_ratio,
    epoch_seed,
    preprocessing_seed,
    norm_methods,
    feat_shuffle_method,
    class_shuffle_method,
    outlier_threshold,
    min_chunk_size=50,
    rank=0,
    world_size=1,
):
    """Build balanced 10-way tasks for the checkpoint's native output head.

    Many-class TabICL inference is hierarchical, but the package's gradient
    training path is a single native-head forward and therefore cannot train
    directly on more than 10 labels.  Partitioning all 60 labels into six
    freshly shuffled 10-way tasks per epoch exposes every class exactly once
    per epoch and adapts the shared representation without inventing targets.
    """
    labels = np.unique(y)
    if not classification or len(labels) <= 10:
        yield from _ORIGINAL_ITER_META_BATCHES(
            X, y, classification=classification, n_estimators=n_estimators,
            max_chunk_size=max_chunk_size, query_ratio=query_ratio,
            epoch_seed=epoch_seed, preprocessing_seed=preprocessing_seed,
            norm_methods=norm_methods, feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold, min_chunk_size=min_chunk_size,
            rank=rank, world_size=world_size,
        )
        return

    rng = np.random.default_rng(epoch_seed)
    labels = rng.permutation(labels)
    for task_idx, task_labels in enumerate(np.array_split(labels, int(np.ceil(len(labels) / 10)))):
        per_class = max(2, max_chunk_size // len(task_labels))
        picked = []
        for label in task_labels:
            available = np.flatnonzero(y == label)
            if len(available) > per_class:
                available = rng.choice(available, size=per_class, replace=False)
            picked.append(available)
        indices = np.concatenate(picked)
        rng.shuffle(indices)
        yield _contiguous_meta_batch(
            X[indices], y[indices], classification=True,
            n_estimators=n_estimators,
            query_size=max(len(task_labels), int(len(indices) * query_ratio)),
            epoch_seed=epoch_seed, chunk_idx=task_idx,
            norm_methods=norm_methods, feat_shuffle_method=feat_shuffle_method,
            class_shuffle_method=class_shuffle_method,
            outlier_threshold=outlier_threshold,
            preprocessing_seed=preprocessing_seed,
        )


tabicl_finetune_base.iter_epoch_meta_batches = _ten_way_meta_batches
tabicl_finetune_base.count_chunks = lambda *args, **kwargs: 6


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y_text = meta[F.TARGET].astype(str).to_numpy(); classes = sorted(set(y_text))
    class_array = np.asarray(classes); class_index = {c: j for j, c in enumerate(classes)}
    y = np.asarray([class_index[c] for c in y_text], np.int64)
    gate = np.load(GATE, allow_pickle=True)
    valid = gate["valid"].astype(int); selected = gate["selected"].astype(int)
    train = np.setdiff1d(np.arange(len(y)), valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts.columns))[:, selected]
    print(f"device=mps split={len(train)}/{len(valid)} features={x.shape[1]} epochs=12",
          flush=True)

    model = FinetunedTabICLClassifier(
        epochs=12,
        learning_rate=1e-5,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=4,
        validation_split_ratio=0.10,
        max_data_size=1000,
        early_stopping=True,
        patience=4,
        time_limit=600,
        save_interval=100,
        device="mps",
        amp=False,
        random_state=557,
        verbose=True,
        support_many_classes=True,
        eval_metric="accuracy",
    )
    t0 = time.time()
    model.fit(x[train], y[train], output_dir=MODEL_OUT)
    raw = model.predict_proba(x[valid])
    tab = np.zeros((len(valid), len(classes)), np.float32)
    for j, code in enumerate(model.classes_.astype(int)):
        tab[:, code] = raw[:, j]
    print(f"fine-tune + inference finished in {time.time()-t0:.1f}s", flush=True)

    et = gate["et"].astype(np.float32); truth = y_text[valid]
    blend = (1-WEIGHT)*et + WEIGHT*tab
    base_ok = class_array[et.argmax(1)] == truth
    rows = []
    for name, pmat in {"ExtraTrees incumbent": et, "fine-tuned TabICLv2": tab,
                       "0.80 ET + 0.20 fine-tuned TabICL": blend}.items():
        ok = class_array[pmat.argmax(1)] == truth
        if name == "ExtraTrees incumbent": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum()); losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "tabicl_finetune_gate.csv", index=False)
    np.savez_compressed(OUT / "tabicl_finetune_gate.npz", valid=valid, et=et,
                        tabicl=tab, truth=truth, classes=class_array, selected=selected)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    passed = rows[1]["accuracy"] >= 0.78 and rows[2]["gain_pt"] > 0.30
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__": main()
