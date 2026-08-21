"""Fit every expert on all 5,000 released training cells and score the 5,000 test cells.

Produces the test-side counterpart of `iteration18_experts*`.  Recovered test truth is
never read here.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_classfeat as CF
import iteration18_experts as E
import iteration18_experts2 as E2
import iteration5_models as M

CACHE = B.OUT / "experts_test.npz"


def main(names=None):
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    Xtr, Xte = data["x_train"], data["x_test"]
    store = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    store["classes"] = classes
    store["allow"] = B.compat_mask(data["meta_train"], y, data["meta_test"], classes)

    cf = CF.load()
    n_tr = len(y)

    def _norm(z):
        z = np.maximum(np.asarray(z, np.float32), 1e-6)
        return z / z.sum(1, keepdims=True)

    fixed = {"sni": _norm(Xte[:, 371:431]),
             "atlaslr": _norm(Xte[:, 560:620]),
             "atlaset": _norm(Xte[:, 620:680])}
    for nm, path in [("atlasnn", "atlas_nn_block.npz"),
                     ("atlasnn3", "atlas_nn3_block.npz"),
                     ("atlasnn4", "atlas_nn4_block.npz"),
                     ("atlasnn5", "atlas_nn5_block.npz"),
                     ("rank", "rankexpert_test.npz"),
                     ("atlaslin", "atlasextra_atlaslin.npz"),
                     ("atlaslin_g", "atlasextra_atlaslin_g.npz"),
                     ("gliann", "atlasextra_gliann.npz"),
                     ("atlaslam_lin", "../iteration19/atlaslam_lin.npz"),
                     ("atlaslam_nn", "../iteration19/atlaslam_nn.npz"),
                     ("atlaslam_et", "../iteration19/atlaslam_et.npz"),
                     ("atlaslam_et2", "../iteration19/atlaslam_et2.npz"),
                     ("atlaslam_md", "../iteration19/atlaslam_md.npz"),
                     ("atlaslam_mdlin", "../iteration19/atlaslam_mdlin.npz"),
                     ("atlaslam_rf_0.1", "../iteration19/atlaslam_rf_0.1.npz"),
                     ("atlaslam_proto", "../iteration19/atlaslam_proto.npz"),
                     ("atlaslam_lin2", "../iteration19/atlaslam_lin2.npz"),
                     ("atlaslam_nn3", "../iteration19/atlaslam_nn3.npz"),
                     ("atlasknn", "../iteration19/atlasknn.npz"),
                     ("atlascons", "../iteration19/atlascons.npz"),
                     ("atlascons2", "../iteration19/atlascons2.npz"),
                     ("atlascons_md", "../iteration19/atlascons_md.npz"),
                     ("atlasftlam", "../iteration19/atlasftlam.npz"),
                     ("atlasnn2", "refnn_atlasnn2.npz"),
                     ("atlasnn_md", "refnn_atlasnn_md.npz"),
                     ("sninn", "refnn_sninn.npz")]:
        f = B.OUT / path
        if f.exists():
            d = np.load(f, allow_pickle=True)
            key = "fine" if "fine" in d.files else "probs"
            arr = d[key]
            # some blocks are stored for all 10,000 challenge cells, others test-only
            fixed[nm] = _norm(arr[n_tr:] if len(arr) > len(Xte) else arr)
    ft = B.OUT / "atlasft_test.npz"
    if ft.exists():
        fixed["atlasft"] = _norm(np.load(ft)["probs"])
    ll = cf["loglik"][n_tr:]
    depth = data["counts_test"].to_numpy().sum(1, keepdims=True)
    fixed["nb"] = softmax(ll / np.maximum(np.sqrt(depth), 1.0) * 4.0, axis=1).astype(np.float32)
    knn = cf["knn"][n_tr:, :, 2]
    knn_tr = cf["knn"][:n_tr, :, 2]
    fixed["knnp"] = softmax(-knn / max(knn_tr.std(), 1e-6) * 1.5, axis=1).astype(np.float32)

    todo = names or ["et", "xgb", "logit", "mlp", "rf", "etnog", "etgene", "etnn",
                     "nb", "knnp", "meta", "sni", "atlaslr", "atlaset", "atlasnn",
                     "atlasnn2", "atlasnn3", "atlasnn4", "atlasnn5",
                     "atlasnn_md", "sninn", "atlasft", "rank", "etaug", "xgbaug",
                     "atlaslin", "atlaslin_g", "gliann", "meta2", "etaug3",
                     "atlaslam_lin", "atlaslam_nn", "atlaslam_et", "atlasftlam",
                     "etaug4_0.08", "etaug4_0.25_3", "xgbaug4"]
    for name in todo:
        if name in store:
            print(f"  {name}: cached"); continue
        t0 = time.time()
        if name in fixed:
            out = fixed[name]
        elif name == "meta":
            key_tr = (data["meta_train"]["Region"].astype(str) + "|"
                      + data["meta_train"]["Excitatory_vs_Inhibitory"].astype(str) + "|"
                      + data["meta_train"]["Segment"].astype(str)).to_numpy()
            key_te = (data["meta_test"]["Region"].astype(str) + "|"
                      + data["meta_test"]["Excitatory_vs_Inhibitory"].astype(str) + "|"
                      + data["meta_test"]["Segment"].astype(str)).to_numpy()
            ci = {c: i for i, c in enumerate(classes)}
            table, glob = {}, np.full(len(classes), 1.0)
            for k, v in zip(key_tr, y):
                table.setdefault(k, np.full(len(classes), 1.0))[ci[v]] += 1.0
                glob[ci[v]] += 1.0
            glob /= glob.sum()
            out = np.zeros((len(key_te), len(classes)), np.float32)
            for r, k in enumerate(key_te):
                row = table.get(k)
                out[r] = (row / row.sum()) if row is not None else glob
        elif name == "meta2":
            splits = [(np.arange(len(y)), np.arange(len(y)))]
            tr = E2._meta_expert2(data, classes, splits)
            key_tr = (data["meta_train"]["Region"].astype(str) + "|"
                      + data["meta_train"]["Excitatory_vs_Inhibitory"].astype(str) + "|"
                      + data["meta_train"]["Segment"].astype(str)).to_numpy()
            key_te = (data["meta_test"]["Region"].astype(str) + "|"
                      + data["meta_test"]["Excitatory_vs_Inhibitory"].astype(str) + "|"
                      + data["meta_test"]["Segment"].astype(str)).to_numpy()
            lut = {}
            for k, row in zip(key_tr, tr):
                lut.setdefault(k, row)
            fallback = tr.mean(0)
            out = np.stack([lut.get(k, fallback) for k in key_te]).astype(np.float32)
        elif name.startswith("etaug4"):
            Xa, Xb = E2.augmented4(data, 18)
            bits = name.split("_")
            mf = float(bits[1]) if len(bits) > 1 else "sqrt"
            leaf = int(bits[2]) if len(bits) > 2 else 2
            from sklearn.ensemble import ExtraTreesClassifier
            out = np.zeros((len(Xb), len(classes)), np.float32)
            nsd = 20
            for sd in range(nsd):
                m_ = ExtraTreesClassifier(n_estimators=600, max_features=mf,
                                          min_samples_leaf=leaf, n_jobs=-1,
                                          random_state=sd).fit(Xa, y)
                idx = {c: i for i, c in enumerate(classes)}
                raw = m_.predict_proba(Xb)
                for j, lab in enumerate(m_.classes_):
                    out[:, idx[str(lab)]] += raw[:, j]
            out /= nsd
        elif name == "xgbaug4":
            Xa, Xb = E2.augmented4(data, 18)
            out = E.expert_xgb(Xa, y, Xb, classes, seeds=(0, 1, 2))
        elif name in ("etaug3", "xgbaug3"):
            Xa, Xb = E2.augmented3(data, 18)
            if name == "etaug3":
                out = M.fit_extra_trees(Xa, pd.Series(y), list(classes), Xb,
                                        seeds=tuple(range(10)))
            else:
                out = E.expert_xgb(Xa, y, Xb, classes, seeds=(0, 1, 2))
        elif name in ("etaug", "xgbaug"):
            Xa, Xb = E2.augmented2(data, 18)
            if name == "etaug":
                out = M.fit_extra_trees(Xa, pd.Series(y), list(classes), Xb,
                                        seeds=tuple(range(10)))
            else:
                out = E.expert_xgb(Xa, y, Xb, classes, seeds=(0, 1, 2))
        elif name == "etnn":
            Xa, Xb = E2.augmented(data)
            out = M.fit_extra_trees(Xa, pd.Series(y), list(classes), Xb,
                                    seeds=tuple(range(10)))
        elif name == "etnog":
            out = M.fit_extra_trees(Xtr[:, E2.CTX_SLICE], pd.Series(y), list(classes),
                                    Xte[:, E2.CTX_SLICE], seeds=tuple(range(10)))
        elif name == "etgene":
            out = M.fit_extra_trees(Xtr[:, E2.GENE_SLICE], pd.Series(y), list(classes),
                                    Xte[:, E2.GENE_SLICE], seeds=tuple(range(10)))
        elif name == "et":
            out = E.expert_et(Xtr, y, Xte, classes, seeds=tuple(range(20)))
        elif name == "rf":
            out = E.expert_rf(Xtr, y, Xte, classes, seeds=(0, 1, 2, 3))
        elif name == "xgb":
            out = E.expert_xgb(Xtr, y, Xte, classes, seeds=(0, 1, 2))
        elif name == "mlp":
            out = E.expert_mlp(Xtr, y, Xte, classes, seeds=(0, 1, 2, 3, 4))
        elif name == "logit":
            out = E.expert_logit(Xtr, y, Xte, classes)
        else:
            raise SystemExit(f"unknown expert {name}")
        out = np.asarray(out, np.float32)
        store[name] = out / np.maximum(out.sum(1, keepdims=True), 1e-12)
        print(f"  {name:6s} test probs done, mean max-p "
              f"{store[name].max(1).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(CACHE, **store)
    np.savez_compressed(CACHE, **store)
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
