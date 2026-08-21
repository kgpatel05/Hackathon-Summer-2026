"""The parent atlas does have `Segment` - it is published as `Laminae`.

`Segment` is the single most informative released column (Region + EI + Segment alone
determine 79.8% of neurons), and every atlas transfer built in this project so far has
been blind to it, which is why SCORECARD 8a concluded that an atlas-trained model
"collapses to ~0.67 on neurons no matter how much data it gets".

Restricted to the 44 cell types that carry a Segment, the class-to-Segment map and the
class-to-Laminae map compose into a bijection: every Laminae level corresponds to exactly
one Segment level and vice versa.  Supplying atlas cells with the Segment implied by their
published Laminae lets a 136,612-cell reference model learn the within-Segment
discrimination that the challenge's own 1,858 neurons can only estimate coarsely.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_atlasnn as ANN
import iteration18_atlasnn3 as A3
import iteration18_refnn as RN
import iteration5_features as F

OUT = Path("outputs/iteration19")
OUT.mkdir(parents=True, exist_ok=True)


def laminae_to_segment(y_fit, meta_fit, al, atlas_lam, verbose=True):
    """Compose class->Segment (challenge labels) with class->Laminae (public atlas)."""
    seg = meta_fit["Segment"].astype(str).to_numpy()
    m = seg != "nan"
    c2s = pd.DataFrame({"y": np.asarray(y_fit)[m], "s": seg[m]}).groupby("y").s.agg(
        lambda v: v.value_counts().index[0])
    c2l = pd.DataFrame({"y": al, "l": atlas_lam}).groupby("y").l.agg(
        lambda v: v.value_counts().index[0])
    shared = [c for c in c2s.index if c in c2l.index]
    pairs = pd.DataFrame({"lam": c2l.reindex(shared).to_numpy(),
                          "seg": c2s.reindex(shared).to_numpy()})
    lam2seg = pairs.groupby("lam").seg.agg(lambda v: v.value_counts().index[0]).to_dict()
    if verbose:
        inj = pairs.groupby("seg").lam.nunique().max()
        sur = pairs.groupby("lam").seg.nunique().max()
        print(f"  laminae->segment map: {len(lam2seg)} levels, "
              f"max segments per laminae {sur}, max laminae per segment {inj}")
    return lam2seg


def design(y_fit=None, meta_fit=None, drop_label_meta=False):
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    lam = atlas["obs_Laminae"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code = np.array([ci.get(l, -1) for l in al])
    if y_fit is None:
        y_fit, meta_fit = data["y"], data["meta_train"]
    lam2seg = laminae_to_segment(y_fit, meta_fit, al, lam)
    atlas_segment = np.array([lam2seg.get(x, "nan") for x in lam])
    cover = float(np.mean(atlas_segment != "nan"))
    print(f"  atlas cells given a Segment: {cover:.3f} "
          f"(challenge training: {np.mean(meta_fit['Segment'].astype(str) != 'nan'):.3f})")

    xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec_a = atlas["obs_Section_ID"].astype(str)
    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    xy_c = meta_all[["center_x", "center_y"]].to_numpy()
    sec_c = meta_all["Section_ID"].astype(str).to_numpy()
    comp_a, dist_a = A3._neighbour_block(xy_a, sec_a, xy_a, sec_a, code, len(classes), True)
    comp_c, dist_c = A3._neighbour_block(xy_c, sec_c, xy_a, sec_a, code, len(classes), False)
    a_qc = np.column_stack([np.log1p(atlas["counts"].sum(1)),
                            (atlas["counts"] > 0).sum(1),
                            (atlas["counts"] == 0).mean(1),
                            np.log1p(np.clip(atlas["obs_volume"], 0, None)),
                            atlas["counts"].sum(1) / np.maximum(atlas["obs_volume"], 1)])
    a_cat = pd.DataFrame({
        "Datasets": atlas["obs_Datasets"].astype(str),
        "Gender": atlas["obs_Gender"].astype(str),
        "Region": ANN._region_to_challenge(atlas["obs_Region"].astype(str)),
        "Excitatory_vs_Inhibitory": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP_position": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str), "Section_ID": sec_a,
        "Segment": atlas_segment})
    if drop_label_meta:
        # a deliberately different view: no Region / EI, so the model must earn the
        # neuron-vs-glia split from expression and tissue context
        a_cat = a_cat.drop(columns=["Region", "Excitatory_vs_Inhibitory"])
    counts_all = np.vstack([data["counts_train"].to_numpy(),
                            data["counts_test"].to_numpy()]).astype(np.float32)
    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(counts_all.sum(1)), (counts_all > 0).sum(1),
                            (counts_all == 0).mean(1),
                            np.log1p(np.clip(vol, 0, None)),
                            counts_all.sum(1) / np.maximum(vol, 1)])
    c_cat = pd.DataFrame({k: meta_all[k].astype(str).to_numpy() for k in a_cat.columns})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([F.log_cpm(atlas["counts"]), a_qc, ANN._section_pos(xy_a, sec_a),
                    comp_a, dist_a, enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(counts_all), c_qc, ANN._section_pos(xy_c, sec_c),
                    comp_c, dist_c, enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    keep = code >= 0
    return sc.transform(Xa)[keep].astype(np.float32), code[keep], \
        sc.transform(Xc).astype(np.float32), data


VARIANTS = {
    # seed counts raised in Iteration 21: averaging more independently initialised fits
    # of the same estimator is strictly variance-reducing and cannot change the estimand
    "atlaslam_lin": dict(hidden=(), epochs=40, dropout=0.0, lr=3e-3, seeds=tuple(range(24))),
    "atlaslam_nn": dict(hidden=(1024, 512), epochs=55, dropout=0.25, lr=2e-3,
                        seeds=tuple(range(12))),
    "atlaslam_md": dict(hidden=(1024, 512), epochs=55, dropout=0.25, lr=2e-3,
                        seeds=tuple(range(12)), drop_label_meta=True),
    "atlaslam_mdlin": dict(hidden=(), epochs=40, dropout=0.0, lr=3e-3,
                           seeds=tuple(range(24)), drop_label_meta=True),
    # extra architectures, purely for ensemble diversity inside the reference family
    "atlaslam_nn3": dict(hidden=(1536, 768, 384), epochs=65, dropout=0.3, lr=1.5e-3,
                         seeds=tuple(range(6))),
    "atlaslam_lin2": dict(hidden=(), epochs=80, dropout=0.0, lr=1e-3,
                          seeds=tuple(range(12))),
}


def build(name):
    out = OUT / f"{name}.npz"
    if out.exists():
        print(f"{name}: cached"); return
    cfg = VARIANTS[name]
    Xa, ya, Xc, data = design(drop_label_meta=cfg.get("drop_label_meta", False))
    classes, y = data["classes"], data["y"]
    print(f"{name}: reference {Xa.shape}", flush=True)
    t0 = time.time()
    probs = RN._train_predict(Xa, ya, len(classes), Xc, cfg["seeds"], cfg["hidden"],
                              cfg["epochs"], cfg["dropout"], cfg["lr"])
    np.savez_compressed(out, probs=probs.astype(np.float32), classes=classes)
    pred = classes[probs[:len(y)].argmax(1)]
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    print(f"{name}: standalone on challenge training cells {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f}) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(VARIANTS)):
        build(nm)
