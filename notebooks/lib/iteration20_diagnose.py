"""DIAGNOSTIC ONLY -- reads the 300 withheld genes of challenge cells.

    ##################################################################
    #  NOTHING IN THIS MODULE MAY BE IMPORTED BY A MODULE THAT       #
    #  PRODUCES A PREDICTION.  It exists to answer "which of our     #
    #  errors are reachable on the released panel and which are      #
    #  not", so that effort is spent where there is signal.          #
    #  Outputs live under outputs/iteration20/DIAGNOSTIC_ONLY/.      #
    ##################################################################

The error budget it produces splits every test error three ways:

  reachable    a strong RELEASED-panel reference model already gets this cell right,
               so our combination lost it and better modelling can win it back
  withheld     only a 500-gene model gets it right -> unreachable on the released panel
  intrinsic    neither gets it right -> annotation ambiguity or genuine noise
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration19_ceiling as C19
import iteration19_laminae as L19
import iteration18_refnn as RN
import iteration5_features as F

QUAR = Path("outputs/iteration20/DIAGNOSTIC_ONLY")
CHALLENGE_500 = QUAR / "challenge_500gene_DO_NOT_USE_IN_MODELS.npz"


def build_challenge_500():
    QUAR.mkdir(parents=True, exist_ok=True)
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    with h5py.File(F.PARENT_ATLAS, "r") as h:
        ids = np.array([x.decode() for x in h["obs/_index"][:]])
        genes = np.array([g.decode() for g in h["var/_index"][:]])
        m = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]),
                              shape=(len(ids), len(genes)))
        lam = [c.decode() for c in h["obs/Laminae/categories"][:]]
        lam_codes = h["obs/Laminae/codes"][:]
    pos = {c: i for i, c in enumerate(ids)}
    order = np.array([pos[c] for c in
                      list(meta_train.index.astype(str)) + list(meta_test.index.astype(str))])
    np.savez_compressed(CHALLENGE_500,
                        counts=np.asarray(m[order].todense(), np.int16), genes=genes,
                        laminae=np.array([lam[k] if k >= 0 else "nan"
                                          for k in lam_codes[order]]))
    print(f"wrote {CHALLENGE_500} (diagnostic only)")


def _aligned(panel, ch_counts, ch_lam):
    """Reference and challenge designs over `panel` genes, built with ONE encoder."""
    from sklearn.preprocessing import OneHotEncoder
    import iteration18_atlasnn as ANN
    import iteration18_atlasnn3 as A3
    data = B.load_all()
    atlas = A.load()
    classes = data["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    code = np.array([ci.get(l, -1) for l in atlas["labels"].astype(str)])

    full = np.load(C19.FULL_CACHE, allow_pickle=True)
    order = {c: i for i, c in enumerate(full["ids"].astype(str))}
    take = np.array([order[c] for c in np.asarray(atlas["ids"]).astype(str)])
    a_counts = full["counts"][take].astype(np.float32)
    genes = full["genes"].astype(str)
    released = set(np.asarray(atlas["genes"]).astype(str))
    cols = (np.array([i for i, g in enumerate(genes) if g in released]) if panel == 200
            else np.arange(len(genes)))

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
        "EI": atlas["obs_Excitatory_vs_Inhibitory"].astype(str),
        "AP": ANN._atlas_ap(atlas["obs_Axial_level"].astype(str)),
        "Mouse": atlas["obs_Mouse_ID"].astype(str), "Section": sec_a,
        "Segment": atlas["obs_Laminae"].astype(str)})

    vol = pd.to_numeric(meta_all["volume"], errors="coerce").to_numpy(float)
    c_qc = np.column_stack([np.log1p(ch_counts.sum(1)), (ch_counts > 0).sum(1),
                            (ch_counts == 0).mean(1), np.log1p(np.clip(vol, 0, None)),
                            ch_counts.sum(1) / np.maximum(vol, 1)])
    c_cat = pd.DataFrame({
        "Datasets": meta_all["Datasets"].astype(str).to_numpy(),
        "Gender": meta_all["Gender"].astype(str).to_numpy(),
        "Region": meta_all["Region"].astype(str).to_numpy(),
        "EI": meta_all["Excitatory_vs_Inhibitory"].astype(str).to_numpy(),
        "AP": meta_all["AP_position"].astype(str).to_numpy(),
        "Mouse": meta_all["Mouse_ID"].astype(str).to_numpy(),
        "Section": sec_c, "Segment": ch_lam})
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        pd.concat([a_cat, c_cat]))
    Xa = np.hstack([F.log_cpm(a_counts[:, cols]), a_qc,
                    ANN._section_pos(xy_a, sec_a), comp_a, dist_a,
                    enc.transform(a_cat)]).astype(np.float32)
    Xc = np.hstack([F.log_cpm(ch_counts[:, cols]), c_qc,
                    ANN._section_pos(xy_c, sec_c), comp_c, dist_c,
                    enc.transform(c_cat)]).astype(np.float32)
    sc = StandardScaler().fit(Xa)
    keep = code >= 0
    return sc.transform(Xa)[keep].astype(np.float32), code[keep], \
        sc.transform(Xc).astype(np.float32), data


def error_budget(seeds=(0, 1)):
    if not CHALLENGE_500.exists():
        build_challenge_500()
    from evaluate import load_truth
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    n_tr = len(y)
    meta = data["meta_test"]
    truth = load_truth().reindex(meta.index.astype(str)).to_numpy()
    ours = pd.read_csv("prediction/prediction.csv", dtype={"Cell_ID": str}
                       ).set_index("Cell_ID").iloc[:, 0].reindex(
        meta.index.astype(str)).to_numpy()

    ch = np.load(CHALLENGE_500, allow_pickle=True)
    ch_counts = ch["counts"].astype(np.float32)
    ch_lam = ch["laminae"].astype(str)

    rows = {}
    for panel in (200, 500):
        Xa, ya, Xc, _ = _aligned(panel, ch_counts, ch_lam)
        t0 = time.time()
        probs = RN._train_predict(Xa, ya, len(classes), Xc, seeds, (1024, 512), 55, 0.25)
        pred = classes[probs[n_tr:].argmax(1)]
        rows[panel] = pred
        print(f"  reference model, {panel}-gene panel: test accuracy "
              f"{np.mean(pred == truth):.4f} ({time.time()-t0:.0f}s)", flush=True)

    ok_us = ours == truth
    ok200 = rows[200] == truth
    ok500 = rows[500] == truth
    err = ~ok_us
    budget = pd.Series({
        "our errors": int(err.sum()),
        "reachable  (released-panel reference already right)": int((err & ok200).sum()),
        "withheld   (only the 500-gene model right)": int((err & ~ok200 & ok500).sum()),
        "intrinsic  (neither right)": int((err & ~ok200 & ~ok500).sum()),
    })
    print("\n" + budget.to_string())
    glia = meta["Region"].isna().to_numpy()
    for nm, m in (("glia", glia), ("neuron", ~glia)):
        print(f"  {nm}: errors {int((err&m).sum())} | reachable {int((err&m&ok200).sum())} "
              f"| withheld {int((err&m&~ok200&ok500).sum())} "
              f"| intrinsic {int((err&m&~ok200&~ok500).sum())}")
    pd.DataFrame({"truth": truth, "ours": ours, "ref200": rows[200], "ref500": rows[500],
                  "glia": glia}).to_csv(QUAR / "error_budget.csv", index=False)
    d = pd.DataFrame({"truth": truth, "ours": ours})[err & ok200]
    print("\ntop reachable confusions (released-panel reference gets these right):")
    print(d.groupby(["truth", "ours"]).size().sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    error_budget()
