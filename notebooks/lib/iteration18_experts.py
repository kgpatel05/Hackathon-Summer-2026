"""Out-of-fold probabilities for several diverse learners on the same 694 features.

Every previous ensemble attempt in this project mixed posteriors ARITHMETICALLY at a
fixed weight.  The incumbent ExtraTrees is severely under-confident (mean max-posterior
0.669 against 0.803 accuracy), so a fixed arithmetic weight hands a sharper but weaker
expert far more influence than its accuracy justifies.  This module produces the raw
per-expert OOF and test posteriors; `iteration18_logpool` combines them in log space,
where differing sharpness is absorbed by the fitted exponents.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration5_models as M


def _align(model, X, classes):
    idx = {c: i for i, c in enumerate(classes)}
    raw = model.predict_proba(X)
    out = np.zeros((len(raw), len(classes)), np.float32)
    for j, lab in enumerate(model.classes_):
        out[:, idx[str(lab)]] = raw[:, j]
    return out


def expert_et(Xf, yf, Xe, classes, seeds=(0, 1, 2, 3, 4)):
    return M.fit_extra_trees(Xf, pd.Series(yf), list(classes), Xe, seeds=seeds)


def expert_rf(Xf, yf, Xe, classes, seeds=(0, 1)):
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in seeds:
        m = RandomForestClassifier(n_estimators=600, max_features="sqrt",
                                   min_samples_leaf=1, n_jobs=-1,
                                   random_state=s).fit(Xf, yf)
        out += _align(m, Xe, classes)
    return out / len(seeds)


def expert_xgb(Xf, yf, Xe, classes, rounds=400, seeds=(0,)):
    import xgboost as xgb
    idx = {c: i for i, c in enumerate(classes)}
    yy = np.array([idx[v] for v in yf])
    dtr = xgb.DMatrix(Xf, label=yy)
    dev = xgb.DMatrix(Xe)
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in seeds:
        params = dict(objective="multi:softprob", num_class=len(classes), eta=0.08,
                      max_depth=6, subsample=0.8, colsample_bytree=0.4,
                      min_child_weight=3, reg_lambda=2.0, tree_method="hist",
                      nthread=11, seed=s)
        bst = xgb.train(params, dtr, num_boost_round=rounds)
        out += bst.predict(dev)
    return out / len(seeds)


def expert_logit(Xf, yf, Xe, classes, C=0.03):
    sc = StandardScaler().fit(Xf)
    m = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(sc.transform(Xf), yf)
    return _align(m, sc.transform(Xe), classes)


def expert_mlp(Xf, yf, Xe, classes, epochs=60, seeds=(0, 1, 2)):
    import torch
    import torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sc = StandardScaler().fit(Xf)
    idx = {c: i for i, c in enumerate(classes)}
    yy = torch.tensor([idx[v] for v in yf], device=dev)
    Xt = torch.tensor(sc.transform(Xf), dtype=torch.float32, device=dev)
    Xv = torch.tensor(sc.transform(Xe), dtype=torch.float32, device=dev)
    out = np.zeros((len(Xe), len(classes)), np.float32)
    for s in seeds:
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(Xt.shape[1], 512), nn.GELU(), nn.Dropout(0.35),
                            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.25),
                            nn.Linear(256, len(classes))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(net(Xt), yy).backward()
            opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            out += torch.softmax(net(Xv), 1).cpu().numpy()
    return out / len(seeds)


EXPERTS = {"et": expert_et, "xgb": expert_xgb, "logit": expert_logit,
           "mlp": expert_mlp, "rf": expert_rf}


def run(seed=18, folds=5, names=tuple(EXPERTS)):
    data = B.load_all()
    classes, y, X = data["classes"], data["y"], data["x_train"]
    cache = B.OUT / f"experts_oof_seed{seed}.npz"
    store = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    allow = np.ones((len(y), len(classes)), bool)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    for fit, val in splits:
        allow[val] = B.compat_mask(data["meta_train"].iloc[fit], y[fit],
                                   data["meta_train"].iloc[val], classes)
    store["allow"] = allow
    store["y"] = y
    store["classes"] = classes
    for name in names:
        if name in store:
            print(f"  {name}: cached")
            continue
        t0 = time.time()
        out = np.zeros((len(y), len(classes)), np.float32)
        for fit, val in splits:
            p = EXPERTS[name](X[fit], y[fit], X[val], classes)
            out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        store[name] = out
        acc_raw = np.mean(classes[np.where(allow, out, -1).argmax(1)] == y)
        corrected = B.prior_correct(out, y, classes)
        acc_pc = np.mean(classes[np.where(allow, corrected, -1).argmax(1)] == y)
        print(f"  {name:6s} OOF {acc_raw:.4f} | prior-corrected {acc_pc:.4f} "
              f"| mean max-p {out.max(1).mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(cache, **store)
    np.savez_compressed(cache, **store)
    print(f"wrote {cache}")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    run(seed=s, names=tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else tuple(EXPERTS))
