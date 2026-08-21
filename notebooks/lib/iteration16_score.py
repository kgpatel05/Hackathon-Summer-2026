"""One-way recovered-test thermometer for the already-frozen Iteration-16 suite.

This module imports no training code and changes no model or submission.  Run only after
``iteration16_novel_suite.py test`` has written all ten candidates and ``test_freeze``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_models as M
import iteration16_common as C
from evaluate import load_truth


def main() -> None:
    freeze_path = C.OUT / "test_freeze.json"
    manifest_path = C.OUT / "test_manifest.csv"
    if not freeze_path.exists() or not manifest_path.exists():
        raise SystemExit("freeze all candidates before reading the test thermometer")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("test_truth_read") is not False or freeze.get("production_modified") is not False:
        raise SystemExit("invalid freeze manifest")

    truth_series = load_truth()
    meta = pd.read_csv("data/meta_test.csv", index_col=0)
    meta.index = meta.index.astype(str)
    truth = truth_series.reindex(meta.index).to_numpy()
    glia = meta["Region"].isna().to_numpy()
    production = pd.read_csv(C.PRODUCTION, dtype={"Cell_ID": str}).set_index("Cell_ID")
    incumbent = production.iloc[:, 0].reindex(meta.index).astype(str).to_numpy()
    incumbent_correct = incumbent == truth

    paths = [("incumbent", C.PRODUCTION)]
    manifest = pd.read_csv(manifest_path)
    paths += [(row.candidate, Path(row.file)) for row in manifest.itertuples()]
    rows = []
    for name, path in paths:
        frame = pd.read_csv(path, dtype={"Cell_ID": str}).set_index("Cell_ID")
        pred = frame.iloc[:, 0].reindex(meta.index).astype(str).to_numpy()
        correct = pred == truth
        p_value, _ = M.paired_mcnemar(correct, incumbent_correct)
        rows.append({
            "candidate": name,
            "accuracy": accuracy_score(truth, pred),
            "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "cohen_kappa": cohen_kappa_score(truth, pred),
            "glia_accuracy": accuracy_score(truth[glia], pred[glia]),
            "neuron_accuracy": accuracy_score(truth[~glia], pred[~glia]),
            "gain_pt_vs_incumbent": 100 * (correct.mean() - incumbent_correct.mean()),
            "changed_vs_incumbent": int(np.sum(pred != incumbent)),
            "wins_vs_incumbent": int((correct & ~incumbent_correct).sum()),
            "losses_vs_incumbent": int((incumbent_correct & ~correct).sum()),
            "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result.to_csv(C.OUT / "test_results.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nDiagnostic thermometer only: these scores were not used to fit or freeze a candidate.")


if __name__ == "__main__":
    main()
