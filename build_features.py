"""Rebuild the four base feature caches the 694-column stack is assembled from.

Self-contained and size-agnostic: it reads only the four released CSVs and the two public
reference files, and works for any number of test/validation cells.  Written so that the
re-run required after `data/meta_test.csv` and `data/counts_test.csv` are replaced by the
validation cohort needs no edits.

Outputs
  outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz  BASE/EXT/SPA/NIC/ATL
  outputs/iteration9/atlas_composition_cache.npz                     k10 composition
  outputs/iteration8/atlas_niche.npz                                 k50 atlas niche
  outputs/iteration9/atlas_et_block.npz                              atlas ExtraTrees
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent / "notebooks" / "lib"))
import iteration5_features as F
import iteration9_atlas_model as I9A

BASE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
COMP_CACHE = Path("outputs/iteration9/atlas_composition_cache.npz")
NICHE_CACHE = Path("outputs/iteration8/atlas_niche.npz")


def main():
    for p in (BASE_CACHE, COMP_CACHE, NICHE_CACHE, I9A.BLOCK_CACHE):
        p.parent.mkdir(parents=True, exist_ok=True)
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str)
    classes = sorted(y.unique())
    genes = list(counts_train.columns)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    print(f"train={len(meta_train)} test={len(meta_test)} genes={len(genes)} "
          f"classes={len(classes)}", flush=True)

    t0 = time.time()
    encoder = OneHotEncoder(handle_unknown="ignore").fit(
        pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str))
    BASE_TR = F.base_block(counts_train, meta_train, encoder)
    BASE_TE = F.base_block(counts_test, meta_test, encoder)
    print(f"[base]    {BASE_TR.shape} / {BASE_TE.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    (EXT_TR, EXT_TE), REF_X, REF_Y = F.reference_transfer(
        genes, classes, [counts_train, counts_test], label_column="voting")
    print(f"[ext]     SNI reference cells={len(REF_X)} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    neuron_all = (~meta_all["Region"].isna()).to_numpy() & (meta_all["Region"] == 1).to_numpy()
    SPATIAL = F.registered_spatial(meta_all, neuron_all)
    SPA_TR, SPA_TE = SPATIAL[: len(meta_train)], SPATIAL[len(meta_train):]
    print(f"[spatial] {SPATIAL.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    EXPR_ALL = F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]))
    NICHE = F.niche_expression(EXPR_ALL, meta_all, k=15, n_components=30)
    NIC_TR, NIC_TE = NICHE[: len(meta_train)], NICHE[len(meta_train):]
    print(f"[niche]   {NICHE.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    (ATL_TR, ATL_TE), n_ref = F.atlas_transfer(genes, classes, [counts_train, counts_test])
    print(f"[atlas]   reference cells={n_ref} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(BASE_CACHE, BASE_TR=BASE_TR, BASE_TE=BASE_TE, EXT_TR=EXT_TR,
                        EXT_TE=EXT_TE, SPA_TR=SPA_TR, SPA_TE=SPA_TE, NIC_TR=NIC_TR,
                        NIC_TE=NIC_TE, REF_EXPR=F.log_cpm(REF_X), REF_Y=REF_Y,
                        EXPR_ALL=EXPR_ALL, ATL_TR=ATL_TR, ATL_TE=ATL_TE)
    print(f"[cache]   {BASE_CACHE}", flush=True)

    t0 = time.time()
    COMP = F.atlas_composition(meta_all, classes, k=10)
    np.savez_compressed(COMP_CACHE, k10=COMP)
    print(f"[comp]    {COMP.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    ANIC = F.atlas_niche(meta_all, genes, k=50, n_components=30)
    np.savez_compressed(NICHE_CACHE, k50=ANIC)
    print(f"[aniche]  {ANIC.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    I9A.build_block(genes, classes, counts_train, meta_train, counts_test, meta_test)
    print(f"[atlas-et] {I9A.BLOCK_CACHE} ({time.time()-t0:.0f}s)", flush=True)
    print("\nall base feature caches rebuilt")


if __name__ == "__main__":
    main()
