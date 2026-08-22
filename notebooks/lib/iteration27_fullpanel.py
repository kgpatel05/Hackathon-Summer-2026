"""Full published 500-gene panel as model input, with a lookup fallback.

Authorised by the organisers on 22 August: teams may use any online resource, including
the full-length published dataset.  The challenge released 200 of the study's 500 genes;
the other 300 are public for the same cells in `MERFISH_spinal_cord_0531.h5ad` and
`SNI_merged_0531.h5ad`, and within-atlas cross-validation puts the full panel at 0.9402
against 0.7863 for the released panel.

The cohort scored on Sunday is replaced after the freeze, so the 300 extra genes may or may
not be available for it.  This module therefore looks each cell up by ID in both public
files and reports coverage:

  * cells found  -> the 500-gene reference model gives its posterior;
  * cells missing -> the posterior is UNIFORM, which in a log-linear pool is exactly "no
    opinion", so the 40 released-panel experts decide those cells unchanged.

One frozen model therefore handles both cases without being modified between Saturday and
Sunday.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration18_atlasnn3 as A3
import iteration18_refnn as RN
import iteration5_features as F

OUT = Path("outputs/iteration27")
OUT.mkdir(parents=True, exist_ok=True)
LOOKUP = OUT / "challenge_full_panel.npz"


def build_lookup():
    """500-gene profile for every challenge/validation cell that appears in a public file."""
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    ids = np.concatenate([meta_train.index.astype(str), meta_test.index.astype(str)])
    genes = None
    found = np.zeros(len(ids), bool)
    out = None
    want = {c: i for i, c in enumerate(ids)}

    for path, dense in ((F.PARENT_ATLAS, False), (F.EXTERNAL, True)):
        with h5py.File(path, "r") as h:
            g = np.array([x.decode() for x in h["var/_index"][:]])
            src_ids = np.array([x.decode() for x in h["obs/_index"][:]])
            hit = np.array([i for i, c in enumerate(src_ids) if c in want])
            if len(hit) == 0:
                continue
            if genes is None:
                genes = g
                out = np.zeros((len(ids), len(genes)), np.float32)
            cols = np.array([list(genes).index(x) for x in g])
            if dense:
                block = h["X"][:, :][hit]
            else:
                m = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:],
                                       h["X/indptr"][:]),
                                      shape=(len(src_ids), len(g)))
                block = np.asarray(m[hit].todense(), np.float32)
            rows = np.array([want[c] for c in src_ids[hit]])
            fresh = ~found[rows]
            out[rows[fresh][:, None], cols[None, :]] = block[fresh]
            found[rows[fresh]] = True
        print(f"  {Path(path).name}: matched {int(found.sum())} of {len(ids)} cells",
              flush=True)

    if genes is None:
        raise SystemExit("no public file contained any challenge cell")
    np.savez_compressed(LOOKUP, counts=out.astype(np.int16), genes=genes,
                        ids=ids, found=found)
    n_tr = len(meta_train)
    print(f"coverage: training {found[:n_tr].mean():.3f}, "
          f"test/validation {found[n_tr:].mean():.3f}")


def load():
    if not LOOKUP.exists():
        build_lookup()
    d = np.load(LOOKUP, allow_pickle=True)
    return d["counts"].astype(np.float32), d["genes"].astype(str), d["found"]


def design():
    """Reference cells and challenge cells over the full panel plus the shared context."""
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code = np.array([ci.get(l, -1) for l in al])

    import iteration19_ceiling as C19
    if not C19.FULL_CACHE.exists():
        # the 500-gene profiles of the non-challenge atlas cells the reference model
        # trains on; built on demand so a fresh clone needs no manual step
        C19.build_full_panel()
    full = np.load(C19.FULL_CACHE, allow_pickle=True)
    order = {c: i for i, c in enumerate(full["ids"].astype(str))}
    take = np.array([order[c] for c in np.asarray(atlas["ids"]).astype(str)])
    a_counts = full["counts"][take].astype(np.float32)
    a_genes = full["genes"].astype(str)

    c_counts, c_genes, found = load()
    cols = np.array([list(c_genes).index(g) for g in a_genes])
    c_counts = c_counts[:, cols]

    xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec_a = atlas["obs_Section_ID"].astype(str)
    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    xy_c = meta_all[["center_x", "center_y"]].to_numpy()
    sec_c = meta_all["Section_ID"].astype(str).to_numpy()
    comp_a, dist_a = A3._neighbour_block(xy_a, sec_a, xy_a, sec_a, code, len(classes), True)
    comp_c, dist_c = A3._neighbour_block(xy_c, sec_c, xy_a, sec_a, code, len(classes), False)

    a_qc = np.column_stack([np.log1p(a_counts.sum(1)), (a_counts > 0).sum(1),
                            (a_counts == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            a_counts.sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": ANN._region_to_challenge(atlas["obs_Region"].astype(str)),
        "Excitatory_vs_Inhibitory": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP_position": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str), "Section_ID": sec_a,
        "Segment": atlas["obs_Laminae"].astype(str)})
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(c_counts.sum(1)), (c_counts > 0).sum(1),
                            (c_counts == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            c_counts.sum(1) / np.maximum(vol, 1)])
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy() for k in a_cat.columns})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([F.log_cpm(a_counts), a_qc, ANN._section_pos(xy_a, sec_a),
                    comp_a, dist_a, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(c_counts), c_qc, ANN._section_pos(xy_c, sec_c),
                    comp_c, dist_c, enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    keep = code >= 0
    return (sc.transform(Xa)[keep].astype(np.float32), code[keep],
            sc.transform(Xc).astype(np.float32), found, data)


VARIANTS = {
    "full500_nn": dict(hidden=(1024, 512), epochs=55, dropout=0.25, lr=2e-3,
                       seeds=tuple(range(6))),
    "full500_lin": dict(hidden=(), epochs=40, dropout=0.0, lr=3e-3,
                        seeds=tuple(range(10))),
}


def build(name):
    dest = OUT / f"{name}.npz"
    if dest.exists():
        print(f"{name}: cached"); return
    cfg = VARIANTS[name]
    Xa, ya, Xc, found, data = design()
    classes, y = data["classes"], data["y"]
    print(f"{name}: reference {Xa.shape}, challenge {Xc.shape}", flush=True)
    t0 = time.time()
    probs = RN._train_predict(Xa, ya, len(classes), Xc, cfg["seeds"], cfg["hidden"],
                              cfg["epochs"], cfg["dropout"], cfg["lr"])
    # cells with no public full-panel record get a uniform posterior: no opinion
    probs[~found] = 1.0 / len(classes)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    np.savez_compressed(dest, probs=probs.astype(np.float32), classes=classes,
                        found=found)
    n = len(y)
    m = found[:n]
    pred = classes[probs[:n].argmax(1)]
    print(f"{name}: on challenge training cells WITH a full-panel record "
          f"({m.sum()}/{n}): {np.mean(pred[m] == y[m]):.4f}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(VARIANTS)):
        build(nm)
