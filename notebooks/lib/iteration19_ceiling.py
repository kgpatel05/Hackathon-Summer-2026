"""How much headroom is left, and where exactly does it live?

Within-atlas cross-validation on the 136,612 non-challenge cells, using the same context
design the Iteration-18 atlas experts use, under three gene panels:

  200  the released panel                    -> the honest ceiling
  500  the full published panel              -> the ceiling the organisers withheld
  300  the withheld panel alone              -> how much of the signal was removed

Non-challenge cells only, so this is a measurement of the *problem*, never a model that
touches a challenge cell.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration18_atlasnn3 as A3
import iteration18_refnn as RN
import iteration5_features as F

FULL_CACHE = B.OUT.parent / "iteration19" / "atlas_full_panel.npz"


def build_full_panel():
    """All 500 published genes for the 136,612 NON-challenge atlas cells."""
    FULL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    with h5py.File(F.PARENT_ATLAS, "r") as h:
        ids = np.array([x.decode() for x in h["obs/_index"][:]])
        genes = np.array([g.decode() for g in h["var/_index"][:]])
        m = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                              shape=(len(ids), len(genes)))
    pos = {c: i for i, c in enumerate(ids)}
    challenge = np.zeros(len(ids), bool)
    for index in (meta_train.index, meta_test.index):
        challenge[[pos[c] for c in index.astype(str)]] = True
    assert challenge.sum() == 10000
    outside = np.flatnonzero(~challenge)
    counts = np.asarray(m[outside].todense(), np.float32)
    keep = counts.sum(1) > 0
    np.savez_compressed(FULL_CACHE, counts=counts[keep].astype(np.int16),
                        genes=genes, ids=ids[outside][keep])
    print(f"wrote {FULL_CACHE}: {counts[keep].shape}")


def context_blocks():
    """The non-expression half of the Iteration-18 atlas design."""
    data = B.load_all()
    atlas = A.load()
    classes = data["classes"]
    al = atlas["labels"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code = np.array([ci.get(l, -1) for l in al])
    xy = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec = atlas["obs_Section_ID"].astype(str)
    comp, dist = A3._neighbour_block(xy, sec, xy, sec, code, len(classes), True)
    qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                          (atlas["counts"] > 0).sum(1),
                          (atlas["counts"] == 0).mean(1),
                          np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                          atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": ANN._region_to_challenge(atlas["obs_Region"].astype(str)),
        "EI": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse": atlas["obs_Mouse_ID"].astype(str), "Section": sec,
        # Iteration 19: `Laminae` IS the challenge's `Segment`, so the honest ceiling
        # must include it - the earlier run understated the released panel.
        "Segment": atlas["obs_Laminae"].astype(str)})
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(cat)
    ctx = np.hstack([qc, ANN._section_pos(xy, sec), comp, dist, onehot]).astype(np.float32)
    glia = atlas["obs_Region"].astype(str) == "nan"
    return ctx, code, al, glia, classes, atlas


def main(n_sub=70000, folds=4, seeds=(0, 1)):
    if not FULL_CACHE.exists():
        build_full_panel()
    full = np.load(FULL_CACHE, allow_pickle=True)
    ctx, code, al, glia, classes, atlas = context_blocks()
    genes500 = full["genes"].astype(str)
    released = set(np.asarray(atlas["genes"]).astype(str))
    idx200 = np.array([i for i, g in enumerate(genes500) if g in released])
    idx300 = np.array([i for i, g in enumerate(genes500) if g not in released])
    # align on the Iteration-18 cache, which drops cells with no released-gene counts
    order = {c: i for i, c in enumerate(full["ids"].astype(str))}
    take = np.array([order[c] for c in np.asarray(atlas["ids"]).astype(str)])
    counts500 = full["counts"][take].astype(np.float32)
    assert len(counts500) == len(ctx), (counts500.shape, ctx.shape)
    print(f"atlas {counts500.shape}, released {len(idx200)}, withheld {len(idx300)}")

    expr = {"200": F.log_cpm(counts500[:, idx200]),
            "300": F.log_cpm(counts500[:, idx300]),
            "500": F.log_cpm(counts500)}
    rng = np.random.default_rng(0)
    rows = {}
    for tag, mask in (("all", np.ones(len(ctx), bool)), ("glia", glia)):
        sub = rng.permutation(np.flatnonzero(mask & (code >= 0)))[:n_sub]
        y = code[sub]
        ok = pd.Series(y).groupby(y).transform("size").to_numpy() >= 20
        sub, y = sub[ok], y[ok]
        lab = np.unique(y)
        remap = {v: i for i, v in enumerate(lab)}
        yy = np.array([remap[v] for v in y])
        for panel in ("200", "500", "300"):
            for use_ctx in (True, False):
                X = np.hstack([expr[panel][sub], ctx[sub]]) if use_ctx else expr[panel][sub]
                X = StandardScaler().fit_transform(X).astype(np.float32)
                t0 = time.time()
                oof = np.zeros(len(yy), int)
                for fit, val in StratifiedKFold(folds, shuffle=True,
                                                random_state=0).split(X, yy):
                    p = RN._train_predict(X[fit], yy[fit], len(lab), X[val], seeds,
                                          (1024, 512), 45, 0.25)
                    oof[val] = p.argmax(1)
                acc = float(np.mean(oof == yy))
                key = f"{tag}/{panel}{'+ctx' if use_ctx else ''}"
                rows[key] = acc
                print(f"  {key:16s} n={len(yy):6d} cls={len(lab):3d} "
                      f"acc {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    out = pd.Series(rows, name="within_atlas_cv_accuracy")
    out.to_csv(B.OUT.parent / "iteration19" / "ceiling_with_segment.csv")
    print("\n" + out.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
