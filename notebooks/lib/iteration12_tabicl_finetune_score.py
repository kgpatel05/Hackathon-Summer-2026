"""Checkpoint-only screen for the Iteration 12 TabICLv2 adaptation.

The four-member hierarchical inference call is expensive on Apple MPS.  This
script reuses the saved best fine-tuned checkpoint and the exact frozen outer
gate with one deterministic ensemble member.  It is a rejection screen only:
failure to reach 78% rejects adaptation, while a pass still requires the
predeclared four-member confirmation in ``iteration12_tabicl_finetune.py``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
from tabicl import TabICLClassifier

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = ROOT / "outputs/iteration12"
GATE = OUT / "tabicl_gate.npz"
CHECKPOINT = OUT / "tabicl_finetune_model/best.ckpt"
WEIGHT = 0.20


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge()
    y = meta[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    gate = np.load(GATE, allow_pickle=True)
    valid = gate["valid"].astype(int)
    selected = gate["selected"].astype(int)
    train = np.setdiff1d(np.arange(len(y)), valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes.tolist(), list(counts.columns))[:, selected]

    model = TabICLClassifier(
        n_estimators=1,
        support_many_classes=True,
        batch_size=1,
        model_path=CHECKPOINT,
        allow_auto_download=False,
        device="mps",
        use_amp=False,
        offload_mode=False,
        random_state=557,
        verbose=True,
    )
    t0 = time.time()
    model.fit(x[train], y[train])
    raw = model.predict_proba(x[valid])
    tab = np.zeros((len(valid), len(classes)), np.float32)
    index = {name: j for j, name in enumerate(classes)}
    for j, name in enumerate(model.classes_.astype(str)):
        tab[:, index[name]] = raw[:, j]
    print(f"checkpoint inference finished in {time.time()-t0:.1f}s", flush=True)

    et = gate["et"].astype(np.float32)
    truth = y[valid]
    base_ok = classes[et.argmax(1)] == truth
    rows = []
    for name, probabilities in {
        "ExtraTrees incumbent": et,
        "fine-tuned TabICLv2 (1 member)": tab,
        "0.80 ET + 0.20 fine-tuned TabICL": (1-WEIGHT)*et + WEIGHT*tab,
    }.items():
        ok = classes[probabilities.argmax(1)] == truth
        if name == "ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p_value})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tabicl_finetune_screen_gate.csv", index=False)
    np.savez_compressed(OUT / "tabicl_finetune_screen_gate.npz", valid=valid,
                        et=et, tabicl=tab, truth=truth, classes=classes,
                        selected=selected)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    print("VERDICT: " + ("RUN FOUR-MEMBER CONFIRMATION"
                         if rows[1]["accuracy"] >= 0.78 else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
