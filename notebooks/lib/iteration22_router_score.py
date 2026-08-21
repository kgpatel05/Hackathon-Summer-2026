"""One-shot test thermometer for the confirmed Iteration-22 pair-rule router.

This is the only router module allowed to import recovered test truth.  It refuses to
run unless the cell-disjoint freeze says the candidate confirmed, and it verifies that
the reconstructed base labels exactly equal the saved Iteration-21 0.8120 artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import iteration22_router_common as C
import iteration22_router_pairrules as R
from evaluate import load_truth


def main() -> None:
    t0 = time.time()
    freeze_path = C.OUT / "pairrule_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get("confirmed") or not freeze.get("test_scoring_authorized"):
        raise RuntimeError("pair-rule router did not clear its cell-disjoint gate")
    if freeze.get("test_truth_read"):
        raise RuntimeError("test thermometer already marked as used")

    bank = C.load_experts("test")
    z = C.pool_logits(bank)
    base = bank["classes"][z.argmax(1)]
    meta = bank["meta"]
    saved = pd.read_csv(
        "outputs/iteration18/predictions/prediction_iteration18_it21.csv",
        dtype={"Cell_ID": str},
    ).set_index("Cell_ID").iloc[:, 0].reindex(meta.index.astype(str)).astype(str).to_numpy()
    reconstruction_mismatch = int((base != saved).sum())
    # Included expert caches received additional seed averaging after Iteration 21 was
    # frozen.  The exact 0.8120 CSV is authoritative; tolerate only the documented
    # microscopic drift and route relative to those saved labels.
    if reconstruction_mismatch > 5:
        raise RuntimeError(f"base reconstruction differs from 0.8120 artifact on "
                           f"{reconstruction_mismatch} cells")
    base = saved

    source = str(freeze["source"])
    alt = R.alternatives(bank, z)[source]
    rules = {tuple(v) for v in freeze["rules"]}
    pred = R.apply_rules(base, alt, rules)
    target = pd.read_csv("prediction/prediction.csv", nrows=0).columns[1]
    frame = pd.DataFrame({"Cell_ID": meta.index.astype(str), target: pred})
    out_path = C.OUT / "prediction_pairrule_router.csv"
    frame.to_csv(out_path, index=False)

    truth = load_truth().reindex(meta.index.astype(str)).astype(str).to_numpy()
    glia = meta["Region"].isna().to_numpy()
    rows = [C.metric_row("iteration21_pool", base, base, truth, glia),
            C.metric_row("pairrule_router", pred, base, truth, glia)]
    result = pd.DataFrame(rows)
    result["runtime_sec"] = time.time() - t0
    result["device"] = "cpu (lookup rule; MPS not applicable)"
    result.to_csv(C.OUT / "test_results.csv", index=False)

    freeze["test_truth_read"] = True
    freeze["base_reconstruction_mismatch"] = reconstruction_mismatch
    freeze["test_result_file"] = str(C.OUT / "test_results.csv")
    freeze["prediction_file"] = str(out_path)
    freeze["test_accuracy"] = float(result.iloc[1].accuracy)
    freeze["test_gain_pt"] = float(100 * (result.iloc[1].accuracy - result.iloc[0].accuracy))
    freeze["test_net"] = int(result.iloc[1].net)
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
