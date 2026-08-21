"""END-TO-END ENTRY POINT: released data in, prediction/prediction.csv out.

Run this after `data/meta_test.csv` and `data/counts_test.csv` are replaced by the
validation cohort:

    python3 run_prediction.py

It fingerprints the input data, discards every derived cache if the data changed (a stale
cache would otherwise produce predictions for the *previous* cohort while appearing to
succeed), rebuilds all forty experts, refits the pool exponents on the released training
cells, and writes the submission.  Stages are resumable: re-running skips work whose
output already exists for the current fingerprint, so an interrupted run can be continued.

The model itself is fixed - a 40-expert calibration-aware log-linear pool, exponents fitted
by out-of-fold likelihood on the released training cells, separate exponents for the glia
and neuron branches.  Nothing here reads a test or validation label.

    python3 run_prediction.py --dry-run    list the stages and what is already built
    python3 run_prediction.py --force      rebuild everything regardless of fingerprint
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

DERIVED = [
    "outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
    "outputs/iteration8/atlas_niche.npz",
    "outputs/iteration9/atlas_composition_cache.npz",
    "outputs/iteration9/atlas_et_block.npz",
    "outputs/iteration18/atlas_cache.npz",
    "outputs/iteration18/hierarchy_maps.npz",
    "outputs/iteration18/class_features.npz",
    "outputs/iteration18/incumbent_probs.npz",
    "outputs/iteration18/experts_test.npz",
    "outputs/iteration18/atlasft_pretrained.pt",
    "outputs/iteration18/atlasft_test.npz",
]
DERIVED_GLOBS = [
    "outputs/iteration18/atlas_nn*_block.npz",
    "outputs/iteration18/refnn_*.npz",
    "outputs/iteration18/atlasextra_*.npz",
    "outputs/iteration18/experts_oof_seed*.npz",
    "outputs/iteration18/atlasft_oof_seed*.npz",
    "outputs/iteration19/*.npz",
    "outputs/iteration19/*.pt",
]

PARTITIONS = ("18", "41", "59", "83")
REF = ("atlaslam_lin", "atlaslam_nn", "atlaslam_md", "atlaslam_mdlin",
       "atlaslam_lin2", "atlaslam_nn3")
E1 = "et,xgb,logit,mlp,rf"
E2 = ("etnog,etgene,etnn,nb,knnp,meta,meta2,sni,atlaslr,atlaset,atlasnn,atlasnn2,"
      "atlasnn3,atlasnn4,atlasnn5,atlasnn_md,sninn,atlaslin,atlaslin_g,gliann,"
      "atlaslam_lin,atlaslam_nn,atlaslam_md,atlaslam_mdlin,atlaslam_lin2,atlaslam_nn3,"
      "atlaslam_et,atlaslam_et2,atlaslam_rf_0.1,atlasft,atlasftlam,etaug,etaug3,"
      "etaug4_0.25_3,xgbaug")


def stages():
    s = [("base feature caches", [sys.executable, "build_features.py"],
          ["outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
           "outputs/iteration9/atlas_et_block.npz"]),
         ("atlas cache (challenge cells removed)",
          [sys.executable, str(LIB / "iteration18_atlas.py")],
          ["outputs/iteration18/atlas_cache.npz"]),
         ("published clustering hierarchy",
          [sys.executable, str(LIB / "iteration18_hierarchy.py")],
          ["outputs/iteration18/hierarchy_maps.npz"]),
         ("per-(cell, class) channels",
          [sys.executable, str(LIB / "iteration18_classfeat.py")],
          ["outputs/iteration18/class_features.npz"]),
         ("incumbent probabilities",
          [sys.executable, str(LIB / "iteration18_base.py")],
          ["outputs/iteration18/incumbent_probs.npz"]),
         ("atlas networks",
          [sys.executable, str(LIB / "iteration18_atlasnn.py")],
          ["outputs/iteration18/atlas_nn_block.npz"]),
         ("atlas network + section context",
          [sys.executable, str(LIB / "iteration18_atlasnn3.py")],
          ["outputs/iteration18/atlas_nn3_block.npz"]),
         ("atlas network, multi-scale",
          [sys.executable, str(LIB / "iteration18_atlasnn3.py"), "multi"],
          ["outputs/iteration18/atlas_nn4_block.npz"]),
         ("atlas network, multi-task",
          [sys.executable, str(LIB / "iteration18_atlasnn5.py")],
          ["outputs/iteration18/atlas_nn5_block.npz"]),
         ("reference variants and SNI network",
          [sys.executable, str(LIB / "iteration18_refnn.py")],
          ["outputs/iteration18/refnn_atlasnn2.npz",
           "outputs/iteration18/refnn_atlasnn_md.npz",
           "outputs/iteration18/refnn_sninn.npz"]),
         ("linear and glia-specialist transfers",
          [sys.executable, str(LIB / "iteration18_atlasextra.py")],
          ["outputs/iteration18/atlasextra_atlaslin.npz",
           "outputs/iteration18/atlasextra_atlaslin_g.npz",
           "outputs/iteration18/atlasextra_gliann.npz"]),
         ("Segment-aware reference models",
          [sys.executable, str(LIB / "iteration19_laminae.py"), *REF],
          [f"outputs/iteration19/{n}.npz" for n in REF]),
         ("reference ExtraTrees (base)",
          [sys.executable, str(LIB / "iteration19_atlasbank.py"), "atlaslam_et"],
          ["outputs/iteration19/atlaslam_et.npz"]),
         ("reference ExtraTrees (tuned) and RandomForest",
          [sys.executable, str(LIB / "iteration19_atlasbank.py"),
           "atlaslam_et_0.1_1_600_10", "atlaslam_rf_0.1"],
          ["outputs/iteration19/atlaslam_et_0.1_1_600_10.npz",
           "outputs/iteration19/atlaslam_rf_0.1.npz"]),
         ]
    for tag, extra, dest in (("", [], "atlasft"), (" (Segment-aware)", ["laminae"], "atlasftlam")):
        s.append((f"atlas-pretrained fine-tuned expert{tag}",
                  [sys.executable, str(LIB / "iteration18_atlasft.py"), *extra, *PARTITIONS],
                  [(f"outputs/iteration19/{dest}_oof_seed{p}.npz" if extra
                    else f"outputs/iteration18/atlasft_oof_seed{p}.npz")
                   for p in PARTITIONS]))
        s.append((f"atlas-pretrained fine-tuned expert{tag}, test",
                  [sys.executable, str(LIB / "iteration18_atlasft.py"), *extra, "test"],
                  [f"outputs/iteration19/{dest}.npz" if extra
                   else "outputs/iteration18/atlasft_test.npz"]))
    for p in PARTITIONS:
        s.append((f"experts, partition {p}",
                  [sys.executable, str(LIB / "iteration18_experts.py"), p, E1], []))
        s.append((f"experts (extended), partition {p}",
                  [sys.executable, str(LIB / "iteration18_experts2.py"), p, E2], []))
    s.append(("test-side experts",
              [sys.executable, str(LIB / "iteration18_experts_test.py")],
              ["outputs/iteration18/experts_test.npz"]))
    # No artifact list: prediction/prediction.csv already exists from the previous
    # cohort, so an existence check here would skip the one stage that must always run.
    s.append(("fit the pool and write the submission",
              [sys.executable, str(LIB / "iteration18_submit.py"), "final"], []))
    return s


def fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("meta_train.csv", "counts_train.csv", "meta_test.csv", "counts_test.csv"):
        p = ROOT / "data" / name
        if not p.exists():
            raise SystemExit(f"missing {p}")
        h.update(name.encode())
        h.update(str(p.stat().st_size).encode())
        with p.open("rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def wipe():
    n = 0
    for rel in DERIVED:
        p = ROOT / rel
        if p.exists():
            p.unlink(); n += 1
    for pattern in DERIVED_GLOBS:
        for p in ROOT.glob(pattern):
            p.unlink(); n += 1
    print(f"[cache] discarded {n} derived artifacts built from the previous data")


def main():
    force = "--force" in sys.argv
    dry = "--dry-run" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    fp = fingerprint()
    old = FINGERPRINT.read_text().strip() if FINGERPRINT.exists() else ""
    if force or (old and old != fp):
        print(f"[data] fingerprint changed\n       was {old[:16] or '(none)'}\n"
              f"       now {fp[:16]}")
        if not dry:
            wipe()
    elif not old:
        print(f"[data] fingerprint {fp[:16]} (first run)")
    else:
        print(f"[data] fingerprint unchanged ({fp[:16]})")

    todo = stages()
    if dry:
        for name, cmd, arts in todo:
            done = arts and all((ROOT / a).exists() for a in arts)
            print(f"  [{'done' if done else '    '}] {name}")
        return

    # the submission from the previous cohort must not survive into a new run
    stale = ROOT / "prediction" / "prediction.csv"
    if (force or (old and old != fp)) and stale.exists():
        shutil.copy(stale, stale.with_suffix(".csv.previous"))
        print("[cache] previous submission moved to prediction/prediction.csv.previous")

    t_all = time.time()
    for i, (name, cmd, arts) in enumerate(todo, 1):
        if arts and all((ROOT / a).exists() for a in arts):
            print(f"[{i}/{len(todo)}] {name}: already built", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {name} ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"stage failed: {name}\n  command: {' '.join(map(str, cmd))}")
        print(f"[{i}/{len(todo)}] {name}: {time.time()-t0:.0f}s", flush=True)
        if name.startswith("reference ExtraTrees (tuned)"):
            shutil.copy(ROOT / "outputs/iteration19/atlaslam_et_0.1_1_600_10.npz",
                        ROOT / "outputs/iteration19/atlaslam_et2.npz")
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
