"""Additional experts with genuinely different inductive biases.

  etnog  ExtraTrees on the 494 context/transfer columns, no raw gene column
  etgene ExtraTrees on the 209 gene + QC columns only
  nb     count-native multinomial generative model with atlas-estimated profiles
  knnp   per-class atlas k-nearest-neighbour distance turned into a posterior

`nb` and `knnp` are fold-independent: they are estimated entirely from the 136,612
non-challenge public atlas cells, so they carry no fold-specific label information.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration18_classfeat as CF
import iteration18_atlasnn as ANN
import iteration5_features as F
import iteration5_models as M

import iteration15_optimal_transport as _I15

GENE_SLICE = slice(0, 209)          # 200 genes + 9 QC columns, always at the front


def _blocks():
    return _I15.block_offsets()


def ctx_slice():
    """Everything except the raw gene columns, whatever the stack width is."""
    return slice(200, _blocks()["TOTAL"][1])


def _fixed_experts(data, n_train):
    cf = CF.load()
    ll = cf["loglik"][:n_train]                      # multinomial log-likelihood
    depth = data["counts_train"].to_numpy().sum(1, keepdims=True)
    nb = softmax(ll / np.maximum(np.sqrt(depth), 1.0) * 4.0, axis=1)
    knn = cf["knn"][:n_train, :, 2]                  # k = 15
    knnp = softmax(-knn / max(knn.std(), 1e-6) * 1.5, axis=1)
    fine, _ = ANN.load()
    x = data["x_train"]
    def _norm(z):
        z = np.maximum(np.asarray(z, np.float32), 1e-6)
        return z / z.sum(1, keepdims=True)
    return {"nb": nb.astype(np.float32), "knnp": knnp.astype(np.float32),
            "atlasnn": fine[:n_train].astype(np.float32),
            # the adopted feature stack already contains three external posteriors;
            # they are fold-independent, so they are free additional pool members
            "sni": _norm(x[:, slice(*_blocks()["EXT"])]),
            "atlaslr": _norm(x[:, slice(*_blocks()["ATL"])]),
            "atlaset": _norm(x[:, slice(*_blocks()["ATL_ET"])]),
            **{k: np.load(B.OUT / f"refnn_{k}.npz")["probs"][:n_train].astype(np.float32)
               for k in ("atlasnn2", "atlasnn_md", "sninn")
               if (B.OUT / f"refnn_{k}.npz").exists()},
            **{k: np.load(B.OUT / f"atlas_nn{v}_block.npz")["probs"][:n_train]
                 .astype(np.float32)
               for k, v in (("atlasnn3", 3), ("atlasnn4", 4), ("atlasnn5", 5))
               if (B.OUT / f"atlas_nn{v}_block.npz").exists()}}


def augmented(data):
    """694 adopted columns plus the 74 atlas-neural transfer columns."""
    fine, coarse = ANN.load()
    n = len(data["y"])
    tr = np.hstack([data["x_train"], fine[:n], coarse[:n]]).astype(np.float32)
    te = np.hstack([data["x_test"], fine[n:], coarse[n:]]).astype(np.float32)
    return tr, te


def _meta_expert2(data, classes, splits, m1=12.0, m2=25.0):
    """Hierarchical backoff p(class | Region, EI, Segment) -> p(class | Region, EI) -> p(class).

    The full key has only 28 levels over ~1,858 neurons, so several cells sit in a level
    with a handful of examples; add-one smoothing on the full key alone shrinks them
    towards the global prior instead of towards the anatomically similar coarser key.
    """
    meta, y = data["meta_train"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    full = (meta["Region"].astype(str) + "|" + meta["Excitatory_vs_Inhibitory"].astype(str)
            + "|" + meta["Segment"].astype(str)).to_numpy()
    coarse = (meta["Region"].astype(str) + "|"
              + meta["Excitatory_vs_Inhibitory"].astype(str)).to_numpy()
    out = np.zeros((len(y), len(classes)), np.float32)
    for fit, val in splits:
        glob = np.full(len(classes), 0.5)
        tab_c, tab_f = {}, {}
        for k1, k2, v in zip(full[fit], coarse[fit], y[fit]):
            tab_f.setdefault(k1, np.zeros(len(classes)))[ci[v]] += 1.0
            tab_c.setdefault(k2, np.zeros(len(classes)))[ci[v]] += 1.0
            glob[ci[v]] += 1.0
        glob /= glob.sum()
        for r in val:
            c_row = tab_c.get(coarse[r])
            p_c = ((c_row + m2 * glob) / (c_row.sum() + m2)) if c_row is not None else glob
            f_row = tab_f.get(full[r])
            out[r] = ((f_row + m1 * p_c) / (f_row.sum() + m1)) if f_row is not None else p_c
    return out


def _meta_expert(data, classes, splits, alpha=1.0):
    """Fold-scoped p(class | Region, Excitatory_vs_Inhibitory, Segment), smoothed."""
    meta = data["meta_train"]
    y = data["y"]
    key = (meta["Region"].astype(str) + "|" + meta["Excitatory_vs_Inhibitory"].astype(str)
           + "|" + meta["Segment"].astype(str)).to_numpy()
    out = np.zeros((len(y), len(classes)), np.float32)
    ci = {c: i for i, c in enumerate(classes)}
    for fit, val in splits:
        table = {}
        glob = np.full(len(classes), alpha)
        for k, v in zip(key[fit], y[fit]):
            table.setdefault(k, np.full(len(classes), alpha))[ci[v]] += 1.0
            glob[ci[v]] += 1.0
        glob /= glob.sum()
        for r in val:
            row = table.get(key[r])
            out[r] = (row / row.sum()) if row is not None else glob
    return out


def augmented6(data, seed):
    """augmented4 plus the Iteration-20 reference views.

    Enriching the feature stack with reference posteriors has paid at every step
    (0.8028 -> 0.8137 -> 0.8184 out-of-fold); these are the three strongest channels
    added since, and they are structurally distinct from the ones already present.
    """
    tr, te = list(augmented6_base(data, seed))
    return tr, te


def augmented6_base(data, seed):
    tr, te = list(augmented4(data, seed))
    n = len(data["y"])
    et, ee = [], []
    for f in ("atlaslam_et2.npz", "atlaslam_md.npz", "atlaslam_mdlin.npz"):
        path = Path("outputs/iteration19") / f
        if path.exists():
            a = np.load(path)["probs"]
            et.append(a[:n]); ee.append(a[n:])
    if et:
        tr = np.hstack([tr] + et).astype(np.float32)
        te = np.hstack([te] + ee).astype(np.float32)
    return tr, te


def augmented5(data, seed):
    """augmented4 plus the remaining distinct posterior channels.

    `meta2` (hierarchical metadata prior) and `nb` (count-native multinomial) are the two
    families with the least overlap with the reference networks, and the two extra atlas
    architectures add ensemble diversity inside the feature stack rather than outside it.
    """
    tr, te = list(augmented4(data, seed))
    n = len(data["y"])
    extra_tr, extra_te = [], []
    for f in (B.OUT / "atlas_nn3_block.npz", B.OUT / "refnn_atlasnn_md.npz"):
        if f.exists():
            d = np.load(f, allow_pickle=True)
            k = "fine" if "fine" in d.files else "probs"
            extra_tr.append(d[k][:n]); extra_te.append(d[k][n:])
    store = B.OUT / f"experts_oof_seed{seed}.npz"
    test = B.OUT / "experts_test.npz"
    if store.exists() and test.exists():
        a, b = np.load(store, allow_pickle=True), np.load(test, allow_pickle=True)
        for nm in ("meta2", "nb"):
            if nm in a.files and nm in b.files:
                extra_tr.append(a[nm]); extra_te.append(b[nm])
    if extra_tr:
        tr = np.hstack([tr] + extra_tr).astype(np.float32)
        te = np.hstack([te] + extra_te).astype(np.float32)
    return tr, te


def augmented4(data, seed):
    """694 adopted columns plus the Segment-aware atlas posteriors (Iteration 19)."""
    n = len(data["y"])
    tr, te = [data["x_train"]], [data["x_test"]]
    fixed = [B.OUT / "atlas_nn5_block.npz", B.OUT / "atlasextra_atlaslin.npz",
             Path("outputs/iteration19") / "atlaslam_lin.npz",
             Path("outputs/iteration19") / "atlaslam_nn.npz",
             # Iteration 21: the tuned ExtraTrees and the RandomForest replace the
             # untuned 2-seed ExtraTrees block, and the metadata-free linear view is added
             Path("outputs/iteration19") / "atlaslam_et2.npz",
             Path("outputs/iteration19") / "atlaslam_rf_0.1.npz",
             Path("outputs/iteration19") / "atlaslam_mdlin.npz"]
    for f in fixed:
        if f.exists():
            a = np.load(f)["probs"]
            tr.append(a[:n]); te.append(a[n:])
    for oof_path, test_path in ((B.OUT / f"atlasft_oof_seed{seed}.npz",
                                 B.OUT / "atlasft_test.npz"),
                                (Path("outputs/iteration19") / f"atlasftlam_oof_seed{seed}.npz",
                                 Path("outputs/iteration19") / "atlasftlam.npz")):
        if oof_path.exists():
            tr.append(np.load(oof_path)["probs"])
            te.append(np.load(test_path)["probs"] if test_path.exists()
                      else np.zeros((len(data["x_test"]), 60), np.float32))
    return np.hstack(tr).astype(np.float32), np.hstack(te).astype(np.float32)


def augmented3(data, seed):
    """694 adopted columns plus the four strongest atlas posteriors."""
    n = len(data["y"])
    tr, te = [data["x_train"]], [data["x_test"]]
    for f in ("atlas_nn3_block.npz", "atlas_nn5_block.npz", "atlasextra_atlaslin.npz"):
        a = np.load(B.OUT / f)["probs"]
        tr.append(a[:n]); te.append(a[n:])
    ft = B.OUT / f"atlasft_oof_seed{seed}.npz"
    if ft.exists():
        tr.append(np.load(ft)["probs"])
        t = B.OUT / "atlasft_test.npz"
        te.append(np.load(t)["probs"] if t.exists()
                  else np.zeros((len(data["x_test"]), 60), np.float32))
    return np.hstack(tr).astype(np.float32), np.hstack(te).astype(np.float32)


def augmented2(data, seed):
    """694 adopted columns + the strongest atlas posteriors (atlasnn3 fixed, atlasft OOF)."""
    n = len(data["y"])
    blocks_tr, blocks_te = [data["x_train"]], [data["x_test"]]
    a3 = np.load(B.OUT / "atlas_nn3_block.npz")["probs"]
    blocks_tr.append(a3[:n]); blocks_te.append(a3[n:])
    ft = B.OUT / f"atlasft_oof_seed{seed}.npz"
    if ft.exists():
        blocks_tr.append(np.load(ft)["probs"])
        te = B.OUT / "atlasft_test.npz"
        blocks_te.append(np.load(te)["probs"] if te.exists()
                         else np.zeros((len(data["x_test"]), a3.shape[1]), np.float32))
    return (np.hstack(blocks_tr).astype(np.float32),
            np.hstack(blocks_te).astype(np.float32))


def run(seed=18, folds=5, names=("etnog", "etgene", "nb", "knnp")):
    data = B.load_all()
    classes, y, X = data["classes"], data["y"], data["x_train"]
    cache = B.OUT / f"experts_oof_seed{seed}.npz"
    store = dict(np.load(cache, allow_pickle=True))
    allow = store["allow"]
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))

    fixed = _fixed_experts(data, len(y))
    ft = B.OUT / f"atlasft_oof_seed{seed}.npz"
    if ft.exists():
        fixed["atlasft"] = np.load(ft)["probs"].astype(np.float32)
    for nm in ("atlaslin", "atlaslin_g", "gliann"):
        f_ = B.OUT / f"atlasextra_{nm}.npz"
        if f_.exists():
            fixed[nm] = np.load(f_)["probs"][:len(y)].astype(np.float32)
    for nm in ("atlaslam_lin", "atlaslam_nn", "atlaslam_et", "atlaslam_et2",
               "atlaslam_md", "atlaslam_mdlin", "atlaslam_rf_0.1", "atlaslam_proto",
               "atlaslam_lin2", "atlaslam_nn3", "atlasknn", "atlascons",
               "atlascons2", "atlascons_md", "atlasftlam"):
        f_ = Path("outputs/iteration19") / f"{nm}.npz"
        if f_.exists():
            fixed[nm] = np.load(f_)["probs"][:len(y)].astype(np.float32)
    ftl = Path("outputs/iteration19") / f"atlasftlam_oof_seed{seed}.npz"
    if ftl.exists():
        fixed["atlasftlam"] = np.load(ftl)["probs"].astype(np.float32)
    rk = B.OUT / f"rankexpert_seed{seed}.npz"
    if rk.exists():
        fixed["rank"] = np.load(rk)["probs"].astype(np.float32)
    for name in names:
        if name in store:
            print(f"  {name}: cached"); continue
        t0 = time.time()
        if name in fixed:
            out = fixed[name]
        elif name == "meta":
            out = _meta_expert(data, classes, splits)
        elif name == "meta2":
            out = _meta_expert2(data, classes, splits)
        elif name.startswith("etaug6"):
            Xa, _ = augmented6(data, seed)
            bits = name.split("_")
            mf = float(bits[1]) if len(bits) > 1 else 0.25
            leaf = int(bits[2]) if len(bits) > 2 else 3
            out = np.zeros((len(y), len(classes)), np.float32)
            nsd = 10
            for fit, val in splits:
                acc = np.zeros((len(val), len(classes)), np.float32)
                for sd in range(nsd):
                    m_ = ExtraTreesClassifier(n_estimators=600, max_features=mf,
                                              min_samples_leaf=leaf, n_jobs=-1,
                                              random_state=sd).fit(Xa[fit], y[fit])
                    idx = {c: i for i, c in enumerate(classes)}
                    raw = m_.predict_proba(Xa[val])
                    for j, lab in enumerate(m_.classes_):
                        acc[:, idx[str(lab)]] += raw[:, j]
                out[val] = acc / np.maximum(acc.sum(1, keepdims=True), 1e-12)
        elif name.startswith("etaug5"):
            Xa, _ = augmented5(data, seed)
            bits = name.split("_")
            mf = float(bits[1]) if len(bits) > 1 else 0.25
            leaf = int(bits[2]) if len(bits) > 2 else 3
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                acc = np.zeros((len(val), len(classes)), np.float32)
                for sd in (0, 1, 2, 3, 4):
                    m_ = ExtraTreesClassifier(n_estimators=600, max_features=mf,
                                              min_samples_leaf=leaf, n_jobs=-1,
                                              random_state=sd).fit(Xa[fit], y[fit])
                    idx = {c: i for i, c in enumerate(classes)}
                    raw = m_.predict_proba(Xa[val])
                    for j, lab in enumerate(m_.classes_):
                        acc[:, idx[str(lab)]] += raw[:, j]
                out[val] = acc / (5 * np.maximum((acc / 5).sum(1, keepdims=True), 1e-12))
        elif name.startswith("etaug4"):
            Xa, _ = augmented4(data, seed)
            bits = name.split("_")
            mf = float(bits[1]) if len(bits) > 1 else "sqrt"
            leaf = int(bits[2]) if len(bits) > 2 else 2
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                acc = np.zeros((len(val), len(classes)), np.float32)
                for sd in range(10):
                    m_ = ExtraTreesClassifier(n_estimators=600, max_features=mf,
                                              min_samples_leaf=leaf, n_jobs=-1,
                                              random_state=sd).fit(Xa[fit], y[fit])
                    idx = {c: i for i, c in enumerate(classes)}
                    raw = m_.predict_proba(Xa[val])
                    for j, lab in enumerate(m_.classes_):
                        acc[:, idx[str(lab)]] += raw[:, j]
                out[val] = acc / np.maximum(acc.sum(1, keepdims=True), 1e-12)
        elif name == "xgbaug4":
            Xa, _ = augmented4(data, seed)
            import iteration18_experts as E
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                p = E.expert_xgb(Xa[fit], y[fit], Xa[val], classes)
                out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        elif name in ("etaug3", "xgbaug3"):
            Xa, _ = augmented3(data, seed)
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                if name == "etaug3":
                    p = M.fit_extra_trees(Xa[fit], pd.Series(y[fit]), list(classes),
                                          Xa[val], seeds=(0, 1, 2, 3, 4))
                else:
                    import iteration18_experts as E
                    p = E.expert_xgb(Xa[fit], y[fit], Xa[val], classes)
                out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        elif name in ("etaug", "xgbaug"):
            Xa, _ = augmented2(data, seed)
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                if name == "etaug":
                    p = M.fit_extra_trees(Xa[fit], pd.Series(y[fit]), list(classes),
                                          Xa[val], seeds=(0, 1, 2, 3, 4))
                else:
                    import iteration18_experts as E
                    p = E.expert_xgb(Xa[fit], y[fit], Xa[val], classes)
                out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        elif name in ("etnn", "xgbnn"):
            Xa, _ = augmented(data)
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                if name == "etnn":
                    p = M.fit_extra_trees(Xa[fit], pd.Series(y[fit]), list(classes),
                                          Xa[val], seeds=(0, 1, 2, 3, 4))
                else:
                    import iteration18_experts as E
                    p = E.expert_xgb(Xa[fit], y[fit], Xa[val], classes)
                out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        else:
            sl = GENE_SLICE if name == "etgene" else ctx_slice()
            out = np.zeros((len(y), len(classes)), np.float32)
            for fit, val in splits:
                p = M.fit_extra_trees(X[fit][:, sl], pd.Series(y[fit]), list(classes),
                                      X[val][:, sl], seeds=(0, 1, 2, 3, 4))
                out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        store[name] = out.astype(np.float32)
        acc = np.mean(classes[np.where(allow, out, -1).argmax(1)] == y)
        pc = B.prior_correct(out, y, classes)
        accp = np.mean(classes[np.where(allow, pc, -1).argmax(1)] == y)
        print(f"  {name:6s} OOF {acc:.4f} | prior-corrected {accp:.4f} | "
              f"mean max-p {out.max(1).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(cache, **store)
    print(f"updated {cache}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    nm = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else (
        "etnog", "etgene", "nb", "knnp", "meta", "sni", "atlaslr", "atlaset")
    run(seed=s, names=nm)
