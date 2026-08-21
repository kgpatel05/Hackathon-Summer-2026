"""More parent-atlas experts, aimed at the glia branch.

The fitted glia exponents are dominated by atlas-derived members - the plain logistic
transfer carries the single largest weight - so the cheapest remaining gains are more
atlas models of different families:

  atlaslin    multinomial logistic on the full context design (not just 200 genes)
  atlaslin_g  multinomial logistic on the 200 genes alone, weaker regularisation
  gliann      network trained only on the 86,356 atlas glia over the 21 glial classes
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

FLOOR = 1e-5


def full_design():
    data = B.load_all()
    classes = data["classes"]
    atlas = A.load()
    al = atlas["labels"].astype(str)
    ci = {c: i for i, c in enumerate(classes)}
    code_a = np.array([ci.get(l, -1) for l in al])
    xy_a = np.vstack([atlas["obs_center_x"], atlas["obs_center_y"]]).T
    sec_a = atlas["obs_Section_ID"].astype(str)
    meta_all = pd.concat([data["meta_train"].drop(columns=[F.TARGET]), data["meta_test"]])
    xy_c = meta_all[["center_x", "center_y"]].to_numpy()
    sec_c = meta_all["Section_ID"].astype(str).to_numpy()
    comp_a, dist_a = A3._neighbour_block(xy_a, sec_a, xy_a, sec_a, code_a, len(classes), True)
    comp_c, dist_c = A3._neighbour_block(xy_c, sec_c, xy_a, sec_a, code_a, len(classes), False)
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
        "Mouse_ID": atlas["obs_Mouse_ID"].astype(str), "Section_ID": sec_a})
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
    return sc.transform(Xa).astype(np.float32), sc.transform(Xc).astype(np.float32), \
        code_a, al, data


def build(name):
    out = B.OUT / f"atlasextra_{name}.npz"
    if out.exists():
        print(f"{name}: cached"); return
    Xa, Xc, code_a, al, data = full_design()
    classes = data["classes"]
    y = data["y"]
    keep = code_a >= 0
    t0 = time.time()
    if name == "atlaslin":
        probs = RN._train_predict(Xa[keep], code_a[keep], len(classes), Xc,
                                  seeds=tuple(range(10)), hidden=(), epochs=40, lr=3e-3)
    elif name == "atlaslin_g":
        g = slice(0, 200)
        probs = RN._train_predict(Xa[keep][:, g], code_a[keep], len(classes),
                                  Xc[:, g], seeds=(0, 1), hidden=(), epochs=40, lr=3e-3)
    elif name == "gliann":
        neuron_types = set(data["meta_train"].loc[
            ~data["meta_train"]["Region"].isna(), F.TARGET].astype(str))
        glia_classes = np.array([c for c in classes if c not in neuron_types])
        gi = {c: i for i, c in enumerate(glia_classes)}
        m = keep & np.array([l in gi for l in al])
        yg = np.array([gi[l] for l in al[m]])
        print(f"gliann: {m.sum()} atlas glia, {len(glia_classes)} classes", flush=True)
        sub = RN._train_predict(Xa[m], yg, len(glia_classes), Xc, seeds=tuple(range(10)),
                                hidden=(1024, 512), epochs=55, dropout=0.25)
        probs = np.full((len(Xc), len(classes)), FLOOR, np.float32)
        idx = [list(classes).index(c) for c in glia_classes]
        probs[:, idx] = np.maximum(sub, FLOOR)
        probs /= probs.sum(1, keepdims=True)
    else:
        raise SystemExit(name)
    np.savez_compressed(out, probs=probs.astype(np.float32), classes=classes)
    print(f"{name}: standalone on challenge training cells "
          f"{np.mean(classes[probs[:len(y)].argmax(1)] == y):.4f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["atlaslin", "atlaslin_g", "gliann"]):
        build(nm)
