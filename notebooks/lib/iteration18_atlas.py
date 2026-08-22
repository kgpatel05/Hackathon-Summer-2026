"""Cache the public parent atlas restricted to the 200 released genes, challenge removed.

Everything downstream in Iteration 18 reads this cache instead of re-parsing the h5ad.
No challenge cell is present in the cache; no withheld gene column is stored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F

CACHE = Path("outputs/iteration18/atlas_cache.npz")
CACHE.parent.mkdir(parents=True, exist_ok=True)
OBS_COLS = ["Region", "Excitatory_vs_Inhibitory", "1st round cluster",
            "2nd round subcluster", "Laminae", "Markers", "Neurotransmitter",
            "Axial level", "Mouse ID", "Section ID", "Datasets", "Gender"]


def build() -> None:
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    gene_order = list(counts_train.columns)
    with h5py.File(F.PARENT_ATLAS, "r") as h:
        ids = np.array([x.decode() for x in h["obs/_index"][:]])
        atlas_genes = [g.decode() for g in h["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        columns = np.array([lookup[g] for g in gene_order])
        matrix = sparse.csr_matrix(
            (h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)))
        cats = [c.decode() for c in h["obs/MERFISH cell type annotation/categories"][:]]
        codes = h["obs/MERFISH cell type annotation/codes"][:]
        obs = {}
        for col in OBS_COLS:
            cc = [c.decode() for c in h[f"obs/{col}/categories"][:]]
            kk = h[f"obs/{col}/codes"][:]
            obs[col] = np.array([cc[k] if k >= 0 else "NA" for k in kk], dtype=object)
        for col in ["center_x", "center_y", "volume"]:
            obs[col] = h[f"obs/{col}"][:]

    labels = np.array([F._normalise_label(cats[c]) if c >= 0 else "NA" for c in codes])
    position = {c: i for i, c in enumerate(ids)}
    challenge = np.zeros(len(ids), bool)
    for index in (meta_train.index, meta_test.index):
        rows = [position[c] for c in index.astype(str) if c in position]
        challenge[rows] = True
    # The validation dataset that replaces meta_test.csv after 3pm 8/22 need not be a
    # subset of the public atlas.  Remove whatever IS present and report the rest; a
    # hard assertion here would break the required re-run.
    print(f"[atlas] challenge cells found in the atlas and removed: "
          f"{int(challenge.sum())} of {len(meta_train) + len(meta_test)}")
    outside = np.flatnonzero(~challenge)

    counts = np.asarray(matrix[outside][:, columns].todense(), np.float32)
    keep = counts.sum(1) > 0
    outside = outside[keep]
    counts = counts[keep]
    payload = {"counts": counts.astype(np.int16), "labels": labels[outside],
               "genes": np.array(gene_order, dtype=object), "ids": ids[outside]}
    for col, arr in obs.items():
        payload["obs_" + col.replace(" ", "_")] = np.asarray(arr)[outside]
    np.savez_compressed(CACHE, **payload)
    print(f"wrote {CACHE}: {counts.shape}, {len(set(labels[outside]))} labels")


def load() -> dict:
    if not CACHE.exists():
        build()
    d = np.load(CACHE, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["counts"] = out["counts"].astype(np.float32)
    return out


if __name__ == "__main__":
    build()
