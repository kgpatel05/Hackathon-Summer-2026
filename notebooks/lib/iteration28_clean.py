"""Source-free model: no data from the dataset the challenge was carved out of.

Rule clarification of 22 August: it is not in the spirit of the event to train on the
source data if discovered, though outside data may be used to inform decisions.  The
source data is `MERFISH_spinal_cord_0531.h5ad` - the published dataset containing the
scored cells themselves, their withheld genes and their labels.  Everything derived from
it is therefore removed: the atlas transfers, the neighbourhood-composition and
atlas-niche feature blocks, the Laminae/Segment correspondence, the fine-tuned reference
networks and the full 500-gene panel.

What remains is what the challenge released plus genuinely outside data:

    BASE  200 released genes (log1p), 9 QC columns, metadata one-hot          371
    EXT   posteriors transferred from SNI_merged_0531.h5ad - a DIFFERENT        60
          experiment on different animals, restricted to the 200 shared genes
    SPA   registered spatial coordinates                                        8
    NIC   niche expression over the challenge cells' own released counts       30

Every expert below is fitted on the released training cells or on SNI; none sees a cell
from the source atlas.

    python3 iteration28_clean.py features   build the source-free feature stack
    python3 iteration28_clean.py experts    out-of-fold and test probabilities
    python3 iteration28_clean.py submit     fit the pool, write prediction/prediction.csv
"""
from __future__ import annotations
import hashlib, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import no_source_data  # noqa: F401  - blocks any read of the source dataset
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration28")
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = OUT / "features.npz"
PARTITIONS = (18, 41, 59, 83)
ALPHA = 0.45
EPS = 1e-9
POOL_RIDGE = 1e-3
MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]


# ------------------------------------------------------------------ log-linear pooling
# Lifted here so the shipped pipeline depends on no module that can read the source data.
def _nll(theta, logs, yi, log_prior, block, l2, n_bias=0, l2_bias=0.0):
    """Negative OOF multinomial log-likelihood and its exact gradient.

    With n_bias > 0 the last `n_bias` entries of theta are a per-class logit offset,
    shrunk separately - a likelihood-fitted Dirichlet-style calibration of the pool.
    """
    core = theta[:len(theta) - n_bias] if n_bias else theta
    bias = theta[len(theta) - n_bias:] if n_bias else None
    w, a = core[:-1], core[-1]
    z = np.tensordot(w, logs, axes=(0, 0)) - a * log_prior[None, :] + block
    if n_bias:
        z = z + bias[None, :]
    lse = logsumexp(z, axis=1)
    rows = np.arange(len(yi))
    value = float(-np.mean(z[rows, yi] - lse) + l2 * np.sum(core ** 2))
    if n_bias:
        value += l2_bias * float(np.sum(bias ** 2))
    p = np.exp(z - lse[:, None])
    onehot = np.zeros_like(p)
    onehot[rows, yi] = 1.0
    resid = p - onehot                                   # (n, C)
    grad_w = np.einsum("mnc,nc->m", logs, resid) / len(yi) + 2 * l2 * w
    grad_a = float(-np.sum(resid * log_prior[None, :]) / len(yi) + 2 * l2 * a)
    grad = np.append(grad_w, grad_a)
    if n_bias:
        grad = np.append(grad, resid.mean(0) + 2 * l2_bias * bias)
    return value, grad


def pool_fit(logs, y, classes, log_prior, allow, rows=None, l2=1e-3, bias_l2=None):
    ci = {c: i for i, c in enumerate(classes)}
    yi = np.array([ci[v] for v in y])
    block = -50.0 * (~allow)          # soft during fitting; hard at decision time
    if rows is not None:
        logs, yi, block = logs[:, rows], yi[rows], block[rows]
    M_ = logs.shape[0]
    C = logs.shape[2]
    nb = C if bias_l2 is not None else 0
    x0 = np.append(np.full(M_, 1.0 / M_), 0.3)
    bounds = [(0.0, 3.0)] * M_ + [(0.0, 1.5)]
    if nb:
        x0 = np.append(x0, np.zeros(C))
        bounds += [(-3.0, 3.0)] * C
    res = minimize(_nll, x0, args=(logs, yi, log_prior, block, l2, nb, bias_l2 or 0.0),
                   method="L-BFGS-B", jac=True, bounds=bounds)
    if nb:
        return res.x[:M_], res.x[M_], res.x[M_ + 1:]
    return res.x[:-1], res.x[-1]


def pool_apply(logs, w, a, log_prior, allow, bias=None):
    z = np.tensordot(w, logs, axes=(0, 0)) - a * log_prior[None, :] + (-1e9 * (~allow))
    return z if bias is None else z + bias[None, :]





def compatibility_mask(meta_fit, y_fit, meta_eval, classes):
    """Hard constraints learned only from released training labels.

    Region, Excitatory_vs_Inhibitory and Segment are each a deterministic function of the
    label on the training cells, so a class never observed with a cell's metadata value is
    impossible for that cell.
    """
    allow = np.ones((len(meta_eval), len(classes)), dtype=bool)
    y_fit = np.asarray(y_fit, dtype=str)
    for column in MASK_COLS:
        fit_values = meta_fit[column].astype(str).to_numpy()
        known = set(fit_values)
        allowed = [set(fit_values[y_fit == cls]) for cls in classes]
        for row, value in enumerate(meta_eval[column].astype(str).to_numpy()):
            if value in known:
                allow[row] &= np.array([value in a for a in allowed])
    allow[~allow.any(1)] = True
    return allow


# ----------------------------------------------------------------- features
def build_features():
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str)
    classes = sorted(y.unique())
    genes = list(counts_train.columns)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    print(f"train={len(meta_train)} test={len(meta_test)} genes={len(genes)} "
          f"classes={len(classes)}", flush=True)

    t0 = time.time()
    enc = OneHotEncoder(handle_unknown="ignore").fit(
        pd.concat([meta_train[F.CATEGORICAL_META],
                   meta_test[F.CATEGORICAL_META]]).astype(str))
    BASE_TR = F.base_block(counts_train, meta_train, enc)
    BASE_TE = F.base_block(counts_test, meta_test, enc)
    print(f"[base]    {BASE_TR.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    (EXT_TR, EXT_TE), REF_X, REF_Y = F.reference_transfer(
        genes, classes, [counts_train, counts_test], label_column="voting")
    print(f"[sni]     {len(REF_X)} outside reference cells ({time.time()-t0:.0f}s)",
          flush=True)

    t0 = time.time()
    neuron = (~meta_all["Region"].isna()).to_numpy() & (meta_all["Region"] == 1).to_numpy()
    SPA = F.registered_spatial(meta_all, neuron)
    print(f"[spatial] {SPA.shape} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    EXPR = F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]))
    NIC = F.niche_expression(EXPR, meta_all, k=15, n_components=30)
    print(f"[niche]   {NIC.shape} ({time.time()-t0:.0f}s)", flush=True)

    n = len(meta_train)
    X_tr = np.hstack([BASE_TR, EXT_TR, SPA[:n], NIC[:n]]).astype(np.float32)
    X_te = np.hstack([BASE_TE, EXT_TE, SPA[n:], NIC[n:]]).astype(np.float32)
    np.savez_compressed(FEATURES, X_tr=X_tr, X_te=X_te,
                        classes=np.array(classes), n_base=BASE_TR.shape[1])
    print(f"[cache]   {FEATURES}  train {X_tr.shape}  test {X_te.shape}")


def load_features():
    if not FEATURES.exists():
        build_features()
    d = np.load(FEATURES, allow_pickle=True)
    return (d["X_tr"], d["X_te"], d["classes"].astype(str), int(d["n_base"]))


# ----------------------------------------------------------------- experts
def _align(model, X, classes):
    idx = {c: i for i, c in enumerate(classes)}
    raw = model.predict_proba(X)
    out = np.zeros((len(raw), len(classes)), np.float32)
    for j, lab in enumerate(model.classes_):
        out[:, idx[str(lab)]] = raw[:, j]
    return out


def _mlp(Xf, yf, Xe, classes, seeds, hidden=(512, 256), epochs=120, dropout=0.35,
         wd=3e-2, lr=2e-3):
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = StandardScaler().fit(Xf)
    idx = {c: i for i, c in enumerate(classes)}
    Xt = torch.tensor(sc.transform(Xf), dtype=torch.float32, device=dev)
    yt = torch.tensor([idx[v] for v in yf], device=dev)
    Xv = torch.tensor(sc.transform(Xe), dtype=torch.float32, device=dev)
    out = np.zeros((len(Xe), len(classes)), np.float32)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    n, bs = len(Xt), 512
    for s in range(seeds):
        torch.manual_seed(s)
        layers, d = [], Xf.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        net = nn.Sequential(*layers, nn.Linear(d, len(classes))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * max((n + bs - 1) // bs, 1))
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            net.train()
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                if len(j) < 8:
                    continue
                opt.zero_grad(); lossf(net(Xt[j]), yt[j]).backward()
                opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            out += torch.softmax(net(Xv), 1).cpu().numpy()
    return out / seeds


def _count_bayes(counts_fit, y_fit, counts_eval, classes):
    """Multinomial profiles estimated from the released TRAINING cells only."""
    prof = np.zeros((len(classes), counts_fit.shape[1]), np.float64)
    for i, c in enumerate(classes):
        m = y_fit == c
        prof[i] = counts_fit[m].sum(0) + 0.5 if m.any() else 0.5
    prof /= prof.sum(1, keepdims=True)
    ll = counts_eval @ np.log(prof).T
    depth = np.maximum(np.sqrt(counts_eval.sum(1, keepdims=True)), 1.0)
    return softmax(ll / depth * 4.0, axis=1).astype(np.float32)


def _meta_prior(meta, y, fit, val, classes, m1=12.0, m2=25.0):
    ci = {c: i for i, c in enumerate(classes)}
    full = (meta["Region"].astype(str) + "|" + meta["Excitatory_vs_Inhibitory"].astype(str)
            + "|" + meta["Segment"].astype(str)).to_numpy()
    coarse = (meta["Region"].astype(str) + "|"
              + meta["Excitatory_vs_Inhibitory"].astype(str)).to_numpy()
    glob = np.full(len(classes), 0.5)
    tab_c, tab_f = {}, {}
    for k1, k2, v in zip(full[fit], coarse[fit], y[fit]):
        tab_f.setdefault(k1, np.zeros(len(classes)))[ci[v]] += 1.0
        tab_c.setdefault(k2, np.zeros(len(classes)))[ci[v]] += 1.0
        glob[ci[v]] += 1.0
    glob /= glob.sum()
    out = np.zeros((len(val), len(classes)), np.float32)
    for r, i in enumerate(val):
        c_row = tab_c.get(coarse[i])
        p_c = ((c_row + m2 * glob) / (c_row.sum() + m2)) if c_row is not None else glob
        f_row = tab_f.get(full[i])
        out[r] = ((f_row + m1 * p_c) / (f_row.sum() + m1)) if f_row is not None else p_c
    return out


def augmented(X_tr, X_te):
    """The released-panel stack plus the outside-data posteriors as extra columns.

    Feeding reference posteriors in as features - rather than only combining them in the
    pool - has been worth more than combining alone at every previous step.
    """
    f = OUT / "sni_experts.npz"
    if not f.exists():
        return X_tr, X_te
    d = np.load(f, allow_pickle=True)
    n = len(X_tr)
    tr = [X_tr] + [d[k][:n] for k in sorted(d.files)]
    te = [X_te] + [d[k][n:] for k in sorted(d.files)]
    return np.hstack(tr).astype(np.float32), np.hstack(te).astype(np.float32)


def experts(seed):
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    X_tr, X_te, classes, n_base = load_features()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    ctr = counts_train.to_numpy(np.float32)
    cte = counts_test.to_numpy(np.float32)
    gene_sl = slice(0, 209)
    ctx_sl = slice(200, X_tr.shape[1])
    sni_sl = slice(n_base, n_base + 60)

    is_test = seed == "test"
    store = OUT / ("experts_test.npz" if is_test else f"experts_oof_seed{seed}.npz")
    d = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    d["classes"] = classes

    if is_test:
        splits = [(np.arange(len(y)), np.arange(len(X_te)))]
        d["allow"] = compatibility_mask(meta_train, y, meta_test, list(classes))
    else:
        skf = StratifiedKFold(5, shuffle=True, random_state=int(seed))
        splits = list(skf.split(X_tr, y))
        allow = np.ones((len(y), len(classes)), bool)
        for fit, val in splits:
            allow[val] = compatibility_mask(meta_train.iloc[fit], y[fit],
                                            meta_train.iloc[val], list(classes))
        d["allow"] = allow
    d["y"] = y

    def emit(name, fn, n_out):
        if name in d:
            print(f"  {name}: cached"); return
        t0 = time.time()
        out = np.zeros((n_out, len(classes)), np.float32)
        for fit, val in splits:
            out[val if not is_test else slice(None)] = fn(fit, val)
        out /= np.maximum(out.sum(1, keepdims=True), 1e-12)
        d[name] = out
        acc = np.mean(classes[np.where(d["allow"], out, -1).argmax(1)] == y) \
            if not is_test else float("nan")
        print(f"  {name:8s} {'test' if is_test else f'OOF {acc:.4f}'}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(store, **d)

    # The outside-data expert bank is a fixed block: identical for every fold, so it is
    # merged in here rather than refitted.  This was previously done by hand, which meant
    # the pipeline produced a 16-expert pool while the measured model had 26.
    sni_bank = OUT / "sni_experts.npz"
    if sni_bank.exists():
        bank = np.load(sni_bank, allow_pickle=True)
        for k in sorted(bank.files):
            d[k] = (bank[k][len(y):] if is_test else bank[k][:len(y)]).astype(np.float32)
        print(f"  merged {len(bank.files)} outside-data experts", flush=True)

    n_out = len(X_te) if is_test else len(y)
    XE = (lambda val: X_te) if is_test else (lambda val: X_tr[val])
    CE = (lambda val: cte) if is_test else (lambda val: ctr[val])
    S = (lambda k: tuple(range(k * (2 if is_test else 1))))

    XA_tr, XA_te = augmented(X_tr, X_te)
    XAE = (lambda val: XA_te) if is_test else (lambda val: XA_tr[val])
    emit("etaug", lambda fit, val: _et_mf(XA_tr[fit], y[fit], XAE(val), classes,
                                          0.25, 3, 10 if is_test else 5), n_out)
    emit("etaug2", lambda fit, val: _et_mf(XA_tr[fit], y[fit], XAE(val), classes,
                                           0.10, 1, 10 if is_test else 5), n_out)
    emit("xgbaug", lambda fit, val: _xgb(XA_tr[fit], y[fit], XAE(val), classes,
                                         3 if is_test else 1), n_out)
    emit("mlpaug", lambda fit, val: _mlp(XA_tr[fit], y[fit], XAE(val), classes,
                                         12 if is_test else 6), n_out)
    emit("et", lambda fit, val: M.fit_extra_trees(
        X_tr[fit], pd.Series(y[fit]), list(classes), XE(val),
        seeds=tuple(range(20 if is_test else 5))), n_out)
    emit("etgene", lambda fit, val: M.fit_extra_trees(
        X_tr[fit][:, gene_sl], pd.Series(y[fit]), list(classes), XE(val)[:, gene_sl],
        seeds=tuple(range(10 if is_test else 5))), n_out)
    emit("etctx", lambda fit, val: M.fit_extra_trees(
        X_tr[fit][:, ctx_sl], pd.Series(y[fit]), list(classes), XE(val)[:, ctx_sl],
        seeds=tuple(range(10 if is_test else 5))), n_out)
    emit("etwide", lambda fit, val: _et_wide(X_tr[fit], y[fit], XE(val), classes,
                                             10 if is_test else 5), n_out)
    emit("rf", lambda fit, val: _rf(X_tr[fit], y[fit], XE(val), classes,
                                    4 if is_test else 2), n_out)
    emit("xgb", lambda fit, val: _xgb(X_tr[fit], y[fit], XE(val), classes,
                                      3 if is_test else 1), n_out)
    emit("logit", lambda fit, val: _logit(X_tr[fit], y[fit], XE(val), classes), n_out)
    emit("mlp", lambda fit, val: _mlp(X_tr[fit], y[fit], XE(val), classes,
                                      12 if is_test else 6), n_out)
    emit("mlpwide", lambda fit, val: _mlp(X_tr[fit], y[fit], XE(val), classes,
                                          12 if is_test else 6, hidden=(1024,),
                                          dropout=0.45, wd=5e-2), n_out)
    emit("nb", lambda fit, val: _count_bayes(ctr[fit], y[fit], CE(val), classes), n_out)
    emit("sni", lambda fit, val: np.maximum(XE(val)[:, sni_sl], 1e-6), n_out)
    emit("meta", lambda fit, val: _meta_prior(
        meta_train if not is_test else pd.concat([meta_train, meta_test]),
        y, fit, val if not is_test else np.arange(len(meta_train), len(meta_train) + len(X_te)),
        classes), n_out)
    np.savez_compressed(store, **d)
    print(f"wrote {store}")


def _et_mf(Xf, yf, Xe, classes, mf, leaf, seeds):
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in range(seeds):
        m = ExtraTreesClassifier(n_estimators=600, max_features=mf,
                                 min_samples_leaf=leaf, n_jobs=-1,
                                 random_state=s).fit(Xf, yf)
        out += _align(m, Xe, classes)
    return out / seeds


def _et_wide(Xf, yf, Xe, classes, seeds):
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in range(seeds):
        m = ExtraTreesClassifier(n_estimators=600, max_features=0.15,
                                 min_samples_leaf=1, n_jobs=-1, random_state=s).fit(Xf, yf)
        out += _align(m, Xe, classes)
    return out / seeds


def _rf(Xf, yf, Xe, classes, seeds):
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in range(seeds):
        m = RandomForestClassifier(n_estimators=600, max_features="sqrt",
                                   min_samples_leaf=1, n_jobs=-1, random_state=s).fit(Xf, yf)
        out += _align(m, Xe, classes)
    return out / seeds


def _logit(Xf, yf, Xe, classes, C=0.03):
    sc = StandardScaler().fit(Xf)
    m = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(sc.transform(Xf), yf)
    return _align(m, sc.transform(Xe), classes)


def _xgb(Xf, yf, Xe, classes, seeds):
    import xgboost as xgb
    idx = {c: i for i, c in enumerate(classes)}
    dtr = xgb.DMatrix(Xf, label=np.array([idx[v] for v in yf]))
    dev = xgb.DMatrix(Xe)
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in range(seeds):
        params = dict(objective="multi:softprob", num_class=len(classes), eta=0.08,
                      max_depth=6, subsample=0.8, colsample_bytree=0.4,
                      min_child_weight=3, reg_lambda=2.0, tree_method="hist",
                      nthread=11, seed=s)
        out += xgb.train(params, dtr, num_boost_round=400).predict(dev)
    return out / seeds


# ----------------------------------------------------------------- pool and submission
def _load_store(tag, names=None):
    f = OUT / ("experts_test.npz" if tag == "test" else f"experts_oof_seed{tag}.npz")
    d = np.load(f, allow_pickle=True)
    avail = sorted(k for k in d.files if k not in ("allow", "y", "classes"))
    use = [n for n in (names or avail) if n in avail]
    logs = np.stack([np.log(np.maximum(d[n], EPS)) for n in use])
    return logs, use, d["allow"], d["classes"].astype(str)


def submit(tag="clean"):
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    glia = meta_train["Region"].isna().to_numpy()
    parts = [_load_store(s) for s in PARTITIONS]
    used = parts[0][1]
    for p_ in parts[1:]:
        used = [n for n in used if n in p_[1]]
    logs = np.concatenate([np.stack([p_[0][p_[1].index(n)] for n in used])
                           for p_ in parts], axis=1)
    allow = np.concatenate([p_[2] for p_ in parts], axis=0)
    yy = np.tile(y, len(parts))
    classes = parts[0][3]
    prior = pd.Series(yy).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    lp = np.log(prior)
    gl = np.tile(glia, len(parts))
    fits = {}
    for name, rows in (("glia", np.flatnonzero(gl)), ("neuron", np.flatnonzero(~gl))):
        fits[name] = pool_fit(logs, yy, classes, lp, allow, rows=rows, l2=POOL_RIDGE)
        w, a = fits[name]
        top = sorted(zip(used, w), key=lambda t: -t[1])[:7]
        print(f"[{name}] a={a:.3f}  " + "  ".join(f"{n}={v:.3f}" for n, v in top
                                                  if v > 5e-3))

    tl, tused, tallow, _ = _load_store("test", used)
    assert tused == used, (tused, used)
    glia_te = meta_test["Region"].isna().to_numpy()
    z = np.zeros((len(tallow), len(classes)))
    z[glia_te] = pool_apply(tl[:, glia_te], *fits["glia"], lp, tallow[glia_te])
    z[~glia_te] = pool_apply(tl[:, ~glia_te], *fits["neuron"], lp, tallow[~glia_te])
    pred = classes[z.argmax(1)]

    example = pd.read_csv("prediction/prediction.csv", nrows=0)
    sub = pd.DataFrame({"Cell_ID": meta_test.index.astype(str),
                        example.columns[1]: pred})
    assert len(sub) == len(meta_test) and not sub.Cell_ID.duplicated().any()
    assert np.array_equal(sub.Cell_ID.to_numpy(), meta_test.index.astype(str).to_numpy())
    assert set(pred) <= set(classes) and sub.iloc[:, 1].notna().all()
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    path = OUT / "predictions" / f"prediction_{tag}.csv"
    text = sub.to_csv(index=False).rstrip("\n")
    path.write_text(text)
    (OUT / "freeze_manifest.json").write_text(json.dumps({
        "candidate": tag, "file": str(path),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "experts": list(used), "fit_partitions": list(PARTITIONS),
        "ridge": POOL_RIDGE, "source_atlas_used": False,
    }, indent=2))
    print(f"\nwrote {path}\n  sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    print(f"  distinct labels {sub.iloc[:, 1].nunique()}/{len(classes)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "features"
    if cmd == "features":
        build_features()
    elif cmd == "experts":
        for s in sys.argv[2:] or list(PARTITIONS) + ["test"]:
            print(f"=== partition {s} ===", flush=True)
            experts(s if s == "test" else int(s))
    elif cmd == "submit":
        submit(sys.argv[2] if len(sys.argv) > 2 else "clean")
