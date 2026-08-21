"""Iteration 12 - RealMLP tuned-default gate on Apple MPS.

RealMLP (NeurIPS 2024, official PyTabKit implementation) combines robust smooth
clipping, modern parameterization, tuned optimization schedules, and internal
best-epoch selection.  It is distinct from both the older plain MLP experiments
and TabM's weight-sharing ensemble.

This fixed gate uses one internal validation split, one full-data refit, class
error for early stopping, and the adopted 694 features.  Advance only if RealMLP
is >=78% standalone and a fixed 80/20 ET/RealMLP blend gains >0.30 point on the
frozen seed-557 outer split.  No test label is read and no submission is written.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # Load the system MPS-compatible build before repo-local packages.

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hackathon-realmlp-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hackathon-realmlp-cache")
sys.path.insert(0, str(ROOT / ".deps"))
from pytabkit import RealMLP_TD_Classifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack
from iteration12_tabm import metadata_mask

OUT = ROOT / "outputs/iteration12"
OUT.mkdir(parents=True, exist_ok=True)
GATE = OUT / "tabicl_gate.npz"
WEIGHT = 0.20


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    gate = np.load(GATE, allow_pickle=True)
    valid = gate["valid"].astype(int)
    train = np.setdiff1d(np.arange(len(y)), valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes.tolist(), list(counts.columns))
    print(f"device=mps split={len(train)}/{len(valid)} features={x.shape[1]} "
          "model=RealMLP-TD epochs<=128 n_cv=1 n_refit=1", flush=True)

    model = RealMLP_TD_Classifier(
        device="mps",
        random_state=557,
        n_cv=1,
        n_refit=1,
        val_fraction=0.15,
        n_epochs=128,
        batch_size=256,
        val_metric_name="class_error",
        verbosity=2,
    )
    t0 = time.time()
    model.fit(x[train], y[train])
    raw = model.predict_proba(x[valid])
    realmlp = np.zeros((len(valid), len(classes)), np.float32)
    index = {label: j for j, label in enumerate(classes)}
    for j, label in enumerate(model.classes_.astype(str)):
        realmlp[:, index[label]] = raw[:, j]
    realmlp = metadata_mask(
        realmlp, meta.iloc[train], y[train], meta.iloc[valid], classes.tolist()
    )
    et = metadata_mask(
        gate["et"].astype(np.float32), meta.iloc[train], y[train],
        meta.iloc[valid], classes.tolist()
    )
    print(f"fit + inference finished in {time.time()-t0:.1f}s", flush=True)

    blend = (1-WEIGHT)*et + WEIGHT*realmlp
    truth = y[valid]
    base_ok = classes[et.argmax(axis=1)] == truth
    rows = []
    for name, probabilities in {
        "masked ExtraTrees incumbent": et,
        "RealMLP-TD": realmlp,
        "0.80 ET + 0.20 RealMLP": blend,
    }.items():
        ok = classes[probabilities.argmax(axis=1)] == truth
        if name == "masked ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p_value})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "realmlp_gate.csv", index=False)
    np.savez_compressed(OUT / "realmlp_gate.npz", valid=valid, et=et,
                        realmlp=realmlp, truth=truth, classes=classes)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    passed = rows[1]["accuracy"] >= 0.78 and rows[2]["gain_pt"] > 0.30
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
