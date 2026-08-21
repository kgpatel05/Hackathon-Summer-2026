"""Post-freeze diagnostic report for Iteration 16.

This script may read recovered test truth, but it cannot fit, tune, promote, or write a
prediction.  Its outputs explain where frozen candidates changed the incumbent and are
not a license to build a test-targeted router.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import iteration16_common as C
from evaluate import load_truth


def main() -> None:
    screen = pd.read_csv(C.OUT / "screen_results.csv")
    test = pd.read_csv(C.OUT / "test_results.csv")
    comparison = screen[["candidate", "accuracy", "cohen_kappa", "gain_pt"]].merge(
        test[["candidate", "accuracy", "cohen_kappa", "gain_pt_vs_incumbent"]],
        on="candidate", suffixes=("_screen", "_test"),
    )
    comparison["screen_to_test_delta_pt"] = 100 * (
        comparison.accuracy_test - comparison.accuracy_screen
    )
    comparison.to_csv(C.OUT / "screen_test_comparison.csv", index=False)

    truth_series = load_truth()
    meta = pd.read_csv("data/meta_test.csv", index_col=0)
    meta.index = meta.index.astype(str)
    truth = truth_series.reindex(meta.index).astype(str).to_numpy()
    production = pd.read_csv(C.PRODUCTION, dtype={"Cell_ID": str}).set_index("Cell_ID")
    incumbent = production.iloc[:, 0].reindex(meta.index).astype(str).to_numpy()
    incumbent_correct = incumbent == truth
    manifest = pd.read_csv(C.OUT / "test_manifest.csv")

    targeted_rows = []
    class_rows = []
    mouse_rows = []
    for row in manifest.itertuples():
        frame = pd.read_csv(row.file, dtype={"Cell_ID": str}).set_index("Cell_ID")
        pred = frame.iloc[:, 0].reindex(meta.index).astype(str).to_numpy()
        correct = pred == truth
        changed = pred != incumbent
        targeted_rows.append({
            "candidate": row.candidate,
            "changed": int(changed.sum()),
            "incumbent_accuracy_on_changed": float(incumbent_correct[changed].mean()),
            "candidate_accuracy_on_changed": float(correct[changed].mean()),
            "net_correct_on_changed": int(correct[changed].sum() - incumbent_correct[changed].sum()),
        })
        for label in np.unique(truth):
            mask = truth == label
            class_rows.append({
                "candidate": row.candidate,
                "class": label,
                "n": int(mask.sum()),
                "incumbent_recall": float(incumbent_correct[mask].mean()),
                "candidate_recall": float(correct[mask].mean()),
                "delta": float(correct[mask].mean() - incumbent_correct[mask].mean()),
            })
        for mouse in meta["Mouse_ID"].astype(str).unique():
            mask = meta["Mouse_ID"].astype(str).to_numpy() == mouse
            mouse_rows.append({
                "candidate": row.candidate,
                "mouse": mouse,
                "n": int(mask.sum()),
                "incumbent_accuracy": float(incumbent_correct[mask].mean()),
                "candidate_accuracy": float(correct[mask].mean()),
                "delta": float(correct[mask].mean() - incumbent_correct[mask].mean()),
            })
    targeted = pd.DataFrame(targeted_rows)
    targeted.to_csv(C.OUT / "changed_cell_diagnostics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(C.OUT / "per_class_diagnostics.csv", index=False)
    pd.DataFrame(mouse_rows).to_csv(C.OUT / "per_mouse_diagnostics.csv", index=False)
    print("SCREEN / TEST\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nCHANGED-CELL YIELD\n" + targeted.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
