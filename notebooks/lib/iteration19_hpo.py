"""Hyperparameter search for the strongest challenge-side learner.

Every tree model in this project has run on hand-set hyperparameters inherited from
Iteration 5, when the design matrix had 371 columns; the augmented stack now has ~1,050,
most of them reference posteriors, so `max_features="sqrt"` samples 3% of the columns.
Configurations are selected on two fold partitions and confirmed on two that selected
nothing.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_experts2 as E2

GRID = [
    dict(model="et", max_features=0.15, min_samples_leaf=3, n_estimators=600),
    dict(model="et", max_features=0.25, min_samples_leaf=3, n_estimators=600),
    dict(model="et", max_features=0.25, min_samples_leaf=5, n_estimators=600),
    dict(model="et", max_features=0.40, min_samples_leaf=3, n_estimators=400),
    dict(model="rf", max_features=0.15, min_samples_leaf=1, n_estimators=400),
]


def oof(cfg, seed, data, seeds=(0, 1, 2)):
    X, y, classes = E2.augmented4(data, seed)[0], data["y"], data["classes"]
    out = np.zeros((len(y), len(classes)), np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for fit, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        acc = np.zeros((len(val), len(classes)), np.float32)
        for sd in seeds:
            cls = ExtraTreesClassifier if cfg["model"] == "et" else RandomForestClassifier
            m = cls(n_estimators=cfg["n_estimators"], max_features=cfg["max_features"],
                    min_samples_leaf=cfg["min_samples_leaf"], n_jobs=-1,
                    random_state=sd).fit(X[fit], y[fit])
            raw = m.predict_proba(X[val])
            for j, lab in enumerate(m.classes_):
                acc[:, idx[str(lab)]] += raw[:, j]
        out[val] = acc / len(seeds)
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def main(select=(18, 41), confirm=(59, 83)):
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    rows = []
    for cfg in GRID:
        t0 = time.time()
        accs = []
        for s in select:
            allow = np.load(B.OUT / f"experts_oof_seed{s}.npz")["allow"]
            p = oof(cfg, s, data)
            accs.append(float(np.mean(classes[np.where(allow, p, -1).argmax(1)] == y)))
        rows.append({**cfg, "select_mean": np.mean(accs),
                     "sec": time.time() - t0})
        print(f"  {cfg} -> {np.mean(accs):.4f} ({time.time()-t0:.0f}s)", flush=True)
    tab = pd.DataFrame(rows).sort_values("select_mean", ascending=False)
    print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = tab.iloc[0].to_dict()
    cfg = {k: best[k] for k in ("model", "max_features", "min_samples_leaf",
                                "n_estimators")}
    cfg["min_samples_leaf"] = int(cfg["min_samples_leaf"])
    cfg["n_estimators"] = int(cfg["n_estimators"])
    print(f"\nconfirming {cfg} on {confirm}")
    for s in confirm:
        allow = np.load(B.OUT / f"experts_oof_seed{s}.npz")["allow"]
        p = oof(cfg, s, data)
        base = float(np.mean(classes[np.where(
            allow, np.load(B.OUT / f"experts_oof_seed{s}.npz")["etaug4_0.08"],
            -1).argmax(1)] == y))
        acc = float(np.mean(classes[np.where(allow, p, -1).argmax(1)] == y))
        print(f"  partition {s}: tuned {acc:.4f} vs etaug4_0.08 {base:.4f} "
              f"({100*(acc-base):+.2f} pt)")
    Path(B.OUT.parent / "iteration19" / "hpo.csv").write_text(tab.to_csv(index=False))


if __name__ == "__main__":
    main()
