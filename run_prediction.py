"""END-TO-END ENTRY POINT: released data in, prediction/prediction.csv out.

Run this after `data/meta_test.csv` and `data/counts_test.csv` are replaced by the
validation cohort:

    python3 run_prediction.py

It fingerprints the input data, discards every derived cache if the data changed (a stale
cache would otherwise produce predictions for the *previous* cohort while appearing to
succeed), rebuilds every expert, refits the pool on the released training cells, and
writes the submission.  Stages are resumable.

    python3 run_prediction.py --dry-run    list the stages and what is already built
    python3 run_prediction.py --force      rebuild everything regardless of fingerprint

SOURCE DATA IS NOT USED.  Per the 22 August clarification, nothing here trains on the
published dataset the challenge was carved from.  The model sees the released 200 genes,
the released metadata, and the challenge cells' own spatial neighbourhoods - nothing
else.  The companion SNI dataset was dropped as well: its labels were produced by
transferring the source atlas's annotations, so training on them would have been
training on the source labels indirectly.  Everything needed to reproduce the
submission is in this repository.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "notebooks" / "lib"
OUT = ROOT / "outputs"
FINGERPRINT = OUT / ".data_fingerprint"

DERIVED_GLOBS = ["outputs/iteration28/*.npz", "outputs/iteration28/predictions/*.csv"]
PARTITIONS = ("18", "41", "59", "83")


# Only what the model actually reads: the five encoded columns, Section_ID for spatial
# grouping, and the three QC columns.  Cohort identifiers are not required.
REQUIRED_META = ["Region", "Excitatory_vs_Inhibitory", "Segment", "Gender",
                 "AP_position", "Section_ID", "center_x", "center_y", "volume"]


def preflight():
    """Check the cohort before spending an hour on it.

    The validation cohort replaces meta_test.csv/counts_test.csv wholesale, so it may
    differ in size, in section and mouse identifiers, and in which metadata values
    appear.  Size and unseen categorical values are handled (nothing downstream assumes
    a cell count, and the one-hot encoder ignores unknowns).  A changed *gene panel* or
    a missing metadata column is not recoverable, so fail here with a clear message
    rather than deep inside a learner.
    """
    import pandas as pd
    d = ROOT / "data"
    tr_c = pd.read_csv(d / "counts_train.csv", index_col=0, nrows=1)
    te_c = pd.read_csv(d / "counts_test.csv", index_col=0, nrows=1)
    miss = sorted(set(tr_c.columns) - set(te_c.columns))
    extra = sorted(set(te_c.columns) - set(tr_c.columns))
    if miss or extra:
        raise SystemExit(
            "counts_test.csv gene panel does not match counts_train.csv.\n"
            f"  train genes {len(tr_c.columns)}, test genes {len(te_c.columns)}\n"
            + (f"  missing from test: {miss[:8]}\n" if miss else "")
            + (f"  unexpected in test: {extra[:8]}\n" if extra else "")
            + "  the model is fitted on the training panel; it cannot score a different one")
    if list(tr_c.columns) != list(te_c.columns):
        # harmless: load_challenge() reindexes the test frame onto the training panel
        print("[preflight] test genes are in a different order; realigned on load")
    te_m = pd.read_csv(d / "meta_test.csv", index_col=0, nrows=5)
    absent = [c for c in REQUIRED_META if c not in te_m.columns]
    if absent:
        raise SystemExit(f"meta_test.csv is missing required column(s): {absent}\n"
                         f"  present: {list(te_m.columns)}")
    n_te = sum(1 for _ in (d / "meta_test.csv").open()) - 1
    print(f"[preflight] {n_te} cells, {len(te_c.columns)} genes, metadata complete")


def stages():
    s = [("source-free feature stack",
          [sys.executable, str(LIB / "iteration28_clean.py"), "features"],
          ["outputs/iteration28/features.npz"])]
    for p in PARTITIONS:
        s.append((f"experts, partition {p}",
                  [sys.executable, str(LIB / "iteration28_clean.py"), "experts", p],
                  [f"outputs/iteration28/experts_oof_seed{p}.npz"]))
    s.append(("experts, test cohort",
              [sys.executable, str(LIB / "iteration28_clean.py"), "experts", "test"],
              ["outputs/iteration28/experts_test.npz"]))
    # no artifact list: prediction/prediction.csv already exists from the previous
    # cohort, so an existence check here would skip the one stage that must always run
    s.append(("fit the pool and write the submission",
              [sys.executable, str(LIB / "iteration28_clean.py"), "submit", "final"], []))
    return s


def fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("meta_train.csv", "counts_train.csv", "meta_test.csv", "counts_test.csv"):
        p = ROOT / "data" / name
        if not p.exists():
            raise SystemExit(f"missing {p}")
        h.update(name.encode())
        with p.open("rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def wipe():
    n = 0
    for pattern in DERIVED_GLOBS:
        for p in ROOT.glob(pattern):
            p.unlink(); n += 1
    print(f"[cache] discarded {n} derived artifacts built from the previous data")


def main():
    force, dry = "--force" in sys.argv, "--dry-run" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    fp = fingerprint()
    old = FINGERPRINT.read_text().strip() if FINGERPRINT.exists() else ""
    changed = force or (old and old != fp)
    if changed:
        print(f"[data] fingerprint changed\n       was {old[:16] or '(none)'}\n"
              f"       now {fp[:16]}")
        if not dry:
            wipe()
            stale = ROOT / "prediction" / "prediction.csv"
            if stale.exists():
                shutil.copy(stale, stale.with_suffix(".csv.previous"))
                print("[cache] previous submission kept as prediction.csv.previous")
    else:
        print(f"[data] fingerprint {fp[:16]}{'' if old else ' (first run)'}")

    todo = stages()
    if dry:
        for name, cmd, arts in todo:
            done = arts and all((ROOT / a).exists() for a in arts)
            print(f"  [{'done' if done else '    '}] {name}")
        return

    t_all = time.time()
    for i, (name, cmd, arts) in enumerate(todo, 1):
        if arts and all((ROOT / a).exists() for a in arts):
            print(f"[{i}/{len(todo)}] {name}: already built", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {name} ...", flush=True)
        t0 = time.time()
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit(f"stage failed: {name}\n  {' '.join(map(str, cmd))}")
        print(f"[{i}/{len(todo)}] {name}: {time.time()-t0:.0f}s", flush=True)
    shutil.copy(ROOT / "outputs/iteration28/predictions/prediction_final.csv",
                ROOT / "prediction/prediction.csv")
    FINGERPRINT.write_text(fp)
    print(f"\ntotal {time.time()-t_all:.0f}s")
    verify()


def verify():
    import numpy as np
    import pandas as pd
    sub = pd.read_csv(ROOT / "prediction/prediction.csv", dtype={"Cell_ID": str})
    meta = pd.read_csv(ROOT / "data/meta_test.csv", index_col=0)
    tr = pd.read_csv(ROOT / "data/meta_train.csv", index_col=0)
    classes = set(tr["MERFISH_cell_type_annotation"].astype(str))
    assert list(sub.columns) == ["Cell_ID", "MERFISH_cell_type_annotation.y"], sub.columns
    assert len(sub) == len(meta), f"{len(sub)} rows for {len(meta)} test cells"
    assert not sub.Cell_ID.duplicated().any(), "duplicate Cell_IDs"
    assert np.array_equal(sub.Cell_ID.to_numpy(), meta.index.astype(str).to_numpy()), \
        "row order does not match meta_test.csv"
    assert set(sub.iloc[:, 1]) <= classes, "predicted a label outside the training taxonomy"
    assert sub.iloc[:, 1].notna().all(), "null prediction"
    print(f"VERIFIED prediction/prediction.csv: {len(sub)} rows, "
          f"{sub.iloc[:, 1].nunique()} distinct labels, order matches meta_test.csv")


if __name__ == "__main__":
    main()
