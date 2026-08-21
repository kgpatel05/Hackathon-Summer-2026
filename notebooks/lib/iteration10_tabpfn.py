"""Iteration 10 - TabPFN-3 MPS gate on the adopted 694-feature stack.

TabPFN-3 supports up to 160 classes natively, so this 60-class problem no longer needs
the experimental many-class decomposition required by TabPFN-2.x.  The first run is a
compute gate, not an adoption test: one frozen stratified 80/20 split (seed 367), two
TabPFN estimators on Apple MPS, and the same five-seed ExtraTrees baseline.

Advance to full paired CV only if TabPFN standalone reaches at least 0.78 and the fixed
80/20 ExtraTrees/TabPFN blend improves ExtraTrees by >0.30 points on this gate.  No hidden
test label is read.  The model checkpoint is downloaded locally by the official package;
feature data never leaves the machine.

Usage:
    PYTHONPATH=.deps python3 notebooks/lib/iteration10_tabpfn.py gate
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.45
SPLIT_SEED = 367
BLEND_WEIGHT = 0.20


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; run this script outside the sandbox")
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y_text = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y_text))
    class_array = np.asarray(classes)
    class_index = {name: i for i, name in enumerate(classes)}
    y = np.asarray([class_index[name] for name in y_text], np.int64)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts_train.columns))
    train, valid = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SPLIT_SEED
    ).split(x, y))
    print(f"MPS={torch.backends.mps.is_available()} x={x.shape} split="
          f"{len(train)}/{len(valid)} classes={len(classes)}", flush=True)

    t0 = time.time()
    et = M.fit_extra_trees(
        x[train], pd.Series(y_text[train]), classes, x[valid], seeds=tuple(range(5))
    )
    et = M.correct_prior(et, M.prior_vector(pd.Series(y_text[train]), classes), ALPHA)
    print(f"ExtraTrees finished in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model = TabPFNClassifier(
        n_estimators=2,
        auto_scale_n_estimators=False,
        device="mps",
        fit_mode="low_memory",
        random_state=0,
        show_progress_bar=True,
    )
    model.fit(x[train], y[train])
    raw = model.predict_proba(x[valid])
    tab = np.zeros((len(valid), len(classes)), np.float32)
    for j, code in enumerate(model.classes_.astype(int)):
        tab[:, code] = raw[:, j]
    print(f"TabPFN finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1.0 - BLEND_WEIGHT) * et + BLEND_WEIGHT * tab
    truth = y_text[valid]
    et_ok = class_array[et.argmax(1)] == truth
    tab_ok = class_array[tab.argmax(1)] == truth
    blend_ok = class_array[blend.argmax(1)] == truth
    p_value, _ = M.paired_mcnemar(blend_ok, et_ok)
    wins = int((blend_ok & ~et_ok).sum())
    losses = int((et_ok & ~blend_ok).sum())
    print(f"ExtraTrees={et_ok.mean():.4f}", flush=True)
    print(f"TabPFN={tab_ok.mean():.4f}", flush=True)
    print(f"0.80 ET + 0.20 TabPFN={blend_ok.mean():.4f} "
          f"gain={100*(blend_ok.mean()-et_ok.mean()):+.2f}pt "
          f"{wins}w/{losses}l p={p_value:.5g}", flush=True)
    passed = tab_ok.mean() >= 0.78 and blend_ok.mean() - et_ok.mean() > 0.0030
    print("VERDICT: " + ("ADVANCE TO FULL CV" if passed else "REJECT"), flush=True)
    np.savez_compressed(OUT / "tabpfn_gate.npz", valid=valid, et=et, tabpfn=tab,
                        y=truth, classes=class_array)


if __name__ == "__main__":
    main()
