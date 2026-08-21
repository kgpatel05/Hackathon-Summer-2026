"""Hierarchical coarse re-weighting.

524 of the 986 out-of-fold errors are glia placed in the wrong published `1st round
cluster`; every neuron error is within-cluster.  The 60-way posterior is therefore
decomposed as p(fine) = p(coarse) * p(fine | coarse), and only the coarse factor is
replaced by a dedicated 14-way estimator fitted on the same 694 features.

Screening only: fitted on released training cells, evaluated out-of-fold.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration5_models as M

ET_KWARGS = dict(n_estimators=600, max_features="sqrt", min_samples_leaf=2, n_jobs=-1)


def group_map(classes):
    h = np.load(B.OUT / "hierarchy_maps.npz", allow_pickle=True)
    assert list(h["classes"].astype(str)) == list(classes)
    g = h["r1"].astype(str)
    groups = np.array(sorted(set(g)))
    gi = {k: i for i, k in enumerate(groups)}
    return g, groups, np.array([gi[x] for x in g])


def coarse_oof(x, y, gy, groups, seed, et_seeds=(0, 1, 2, 3, 4), n_splits=5):
    out = np.zeros((len(y), len(groups)), np.float32)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fit, val in skf.split(x, y):
        p = np.zeros((len(val), len(groups)), np.float32)
        for s in et_seeds:
            m = ExtraTreesClassifier(random_state=s, **ET_KWARGS).fit(x[fit], gy[fit])
            idx = {c: i for i, c in enumerate(groups)}
            raw = m.predict_proba(x[val])
            for j, lab in enumerate(m.classes_):
                p[:, idx[str(lab)]] += raw[:, j]
        out[val] = p / len(et_seeds)
    return out


def main(seed=18):
    data = B.load_all()
    classes, y, x = data["classes"], data["y"], data["x_train"]
    g, groups, col = group_map(classes)
    gy = np.array([g[list(classes).index(t)] for t in y])

    c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
    if seed == 18:
        inc_raw, allow = c["oof_raw"], c["oof_allow"]
    else:
        inc_raw, allow = B.oof_probabilities(data, seed=seed, raw=True)
    inc = B.prior_correct(inc_raw, y, classes)
    inc = np.where(allow, inc, 0.0)
    inc = inc / np.maximum(inc.sum(1, keepdims=True), 1e-12)
    base = float(np.mean(classes[inc.argmax(1)] == y))

    t0 = time.time()
    pc = coarse_oof(x, y, gy, groups, seed)
    pc = pc / np.maximum(pc.sum(1, keepdims=True), 1e-12)
    print(f"dedicated coarse model fitted ({time.time()-t0:.0f}s)")

    # coarse marginal implied by the incumbent
    P_inc = np.zeros((len(y), len(groups)), np.float32)
    for j in range(len(classes)):
        P_inc[:, col[j]] += inc[:, j]
    gi = {k: i for i, k in enumerate(groups)}
    gy_i = np.array([gi[t] for t in gy])
    print(f"incumbent OOF {base:.4f} | coarse acc: marginal "
          f"{np.mean(groups[P_inc.argmax(1)] == gy):.4f} vs dedicated "
          f"{np.mean(groups[pc.argmax(1)] == gy):.4f} "
          f"(oracle-coarse fine {np.mean(classes[np.where(col[None,:]==gy_i[:,None], inc, -1).argmax(1)]==y):.4f})")

    rows = []
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        ratio = (np.maximum(pc, 1e-9) / np.maximum(P_inc, 1e-9)) ** lam
        new = inc * ratio[:, col]
        new = np.where(allow, new, 0.0)
        pred = classes[new.argmax(1)]
        acc = float(np.mean(pred == y))
        rows.append({"lambda": lam, "acc": acc, "gain_pt": 100 * (acc - base),
                     "changed": int((pred != classes[inc.argmax(1)]).sum())})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    np.savez_compressed(B.OUT / f"coarse_oof_seed{seed}.npz", pc=pc, P_inc=P_inc,
                        inc=inc, allow=allow, groups=groups, col=col, y=y,
                        classes=classes)
    return tab


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    main(s)
