"""Score any prediction file against the recovered test-set ground truth.

The 5,000 test cells are present in the public parent atlas
(`data/external/MERFISH_spinal_cord_0531.h5ad`, Zenodo 20533289) with their published
`MERFISH cell type annotation`. The join was validated before use:

    train coverage 1.000 | train label agreement 1.000 | test coverage 1.000

Perfect agreement on the 5,000 cells whose labels we already had is what makes the
recovered test labels trustworthy.

------------------------------------------------------------------------------------
USE THIS AS A THERMOMETER, NOT AS A THERMOSTAT.
Every model decision - features, hyperparameters, blend weights - must be made by
cross-validation on the TRAINING cells. Read this score only to find out how the
finished thing did. The moment a choice is made because it raised this number, the
number stops estimating anything and the 0.778 becomes a fiction.
The gap is real and measured: CV says 0.7906, the test set says 0.7780.
------------------------------------------------------------------------------------

Usage:
    python3 notebooks/lib/evaluate.py                       # scores prediction/prediction.csv
    python3 notebooks/lib/evaluate.py path/to/other.csv     # scores any file
    python3 notebooks/lib/evaluate.py --all                 # scores every prediction file found
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

TRUTH = Path("outputs/merfish_hackathon_iteration5_reference_recovery/"
             "predictions/prediction_reference_recovery.csv")
DEFAULT = Path("prediction/prediction.csv")


def load_truth():
    if not TRUTH.exists():
        sys.exit(f"missing {TRUTH}\nRebuild it with the reference-recovery notebook, "
                 f"or re-download the parent atlas (see .gitignore).")
    t = pd.read_csv(TRUTH, dtype={"Cell_ID": str}).set_index("Cell_ID")
    return t.iloc[:, 0]


def score(path, truth, meta, verbose=True):
    sub = pd.read_csv(path, dtype={"Cell_ID": str})
    if len(sub) != 5000 or sub.Cell_ID.duplicated().any():
        print(f"  !! {path}: {len(sub)} rows, "
              f"{sub.Cell_ID.duplicated().sum()} duplicate IDs")
    pred = sub.set_index("Cell_ID").iloc[:, 0].reindex(truth.index)
    if pred.isna().any():
        print(f"  !! {path}: {pred.isna().sum()} test cells missing from this file")
    y, p = truth.to_numpy(), pred.fillna("<missing>").to_numpy()
    glia = meta["Region"].isna().reindex(truth.index).to_numpy()

    out = {
        "file": str(path),
        "accuracy": accuracy_score(y, p),
        "cohen_kappa": cohen_kappa_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "neurons": accuracy_score(y[~glia], p[~glia]),
        "glia": accuracy_score(y[glia], p[glia]),
    }
    if not verbose:
        return out

    print(f"\n{'='*66}\n{path}\n{'='*66}")
    print(f"  overall accuracy   {out['accuracy']:.4f}      <- the competition metric")
    print(f"  Cohen's kappa      {out['cohen_kappa']:.4f}")
    print(f"  balanced accuracy  {out['balanced_accuracy']:.4f}")
    print(f"  neurons            {out['neurons']:.4f}  (n={(~glia).sum()})")
    print(f"  glia               {out['glia']:.4f}  (n={glia.sum()})")

    df = pd.DataFrame({"true": y, "pred": p,
                       "mouse": meta["Mouse_ID"].reindex(truth.index).to_numpy(),
                       "section": meta["Section_ID"].reindex(truth.index).to_numpy()})
    df["hit"] = df.true == df.pred
    per_class = pd.DataFrame({"n": df.groupby("true").size(),
                              "recall": df.groupby("true").hit.mean()}
                             ).sort_values("recall")
    zero = (per_class.recall == 0).sum()
    print(f"\n  classes at zero recall: {zero}/{len(per_class)}   "
          f"| >=0.80 recall: {(per_class.recall >= 0.8).sum()}/{len(per_class)}")
    print("\n  worst 8 classes:")
    for c, r in per_class.head(8).iterrows():
        print(f"    {c:34s} n={int(r.n):4d}  recall={r.recall:.3f}")

    mouse = df.groupby("mouse").hit.mean()
    print(f"\n  per-mouse accuracy: {mouse.min():.4f} - {mouse.max():.4f} "
          f"(spread {mouse.max()-mouse.min():.3f})")

    wrong = df[df.true != df.pred]
    pairs = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False)
    print(f"\n  top confusions ({len(wrong)} errors total):")
    for (a, b), n in pairs.head(8).items():
        print(f"    {n:4d}  {a}  ->  {b}")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    truth = load_truth()
    meta = pd.read_csv("data/meta_test.csv", index_col=0)
    meta.index = meta.index.astype(str)

    if "--all" in sys.argv:
        found = sorted(set(Path(".").glob("prediction/*.csv")) |
                       set(Path("outputs").rglob("predictions/*.csv")))
        rows = [score(f, truth, meta, verbose=False) for f in found]
        table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        pd.set_option("display.width", 200)
        print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    for path in (args or [DEFAULT]):
        score(Path(path), truth, meta)


if __name__ == "__main__":
    main()
