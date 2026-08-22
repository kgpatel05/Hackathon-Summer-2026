"""Tree-based reference models on the Segment-aware atlas design.

The linear softmax on this design already transfers at 0.8056 without seeing a challenge
cell, so it is worth spending compute on other model families over the same 136,612 rows.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration19_laminae as L

OUT = Path("outputs/iteration19")
OUT.mkdir(parents=True, exist_ok=True)


def build(name):
    out = OUT / f"{name}.npz"
    if out.exists():
        print(f"{name}: cached"); return
    Xa, ya, Xc, data = L.design()
    classes, y = data["classes"], data["y"]
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    t0 = time.time()
    if name.startswith("atlaslam_et"):
        from sklearn.ensemble import ExtraTreesClassifier
        bits = name.split("_")
        mf = float(bits[2]) if len(bits) > 2 else 0.1
        leaf = int(bits[3]) if len(bits) > 3 else 2
        ntree = int(bits[4]) if len(bits) > 4 else 300
        nseed = int(bits[5]) if len(bits) > 5 else 2
        probs = np.zeros((len(Xc), len(classes)), np.float32)
        for sd in range(nseed):
            m = ExtraTreesClassifier(n_estimators=ntree, max_features=mf,
                                     min_samples_leaf=leaf, n_jobs=-1,
                                     random_state=sd).fit(Xa, ya)
            p = m.predict_proba(Xc)
            aligned = np.zeros_like(probs)
            aligned[:, m.classes_.astype(int)] = p
            probs += aligned
        probs /= nseed
    elif name.startswith("atlaslam_rf"):
        from sklearn.ensemble import RandomForestClassifier
        bits = name.split("_")
        mf = float(bits[2]) if len(bits) > 2 else 0.1
        probs = np.zeros((len(Xc), len(classes)), np.float32)
        for sd in (0, 1):
            m = RandomForestClassifier(n_estimators=400, max_features=mf,
                                       min_samples_leaf=1, n_jobs=-1,
                                       random_state=sd).fit(Xa, ya)
            p = m.predict_proba(Xc)
            aligned = np.zeros_like(probs)
            aligned[:, m.classes_.astype(int)] = p
            probs += aligned
        probs /= 2
    elif name == "atlaslam_proto":
        # nearest-class-mean in the aligned design, softened by within-class scatter:
        # a metric view rather than a discriminative one
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        Xt = torch.tensor(Xa, device=dev)
        yt = torch.tensor(ya, device=dev)
        cent = torch.stack([Xt[yt == k].mean(0) if (yt == k).any()
                            else torch.zeros(Xt.shape[1], device=dev)
                            for k in range(len(classes))])
        scat = torch.stack([Xt[yt == k].std(0).mean() if (yt == k).sum() > 1
                            else torch.tensor(1.0, device=dev)
                            for k in range(len(classes))])
        q = torch.tensor(Xc, device=dev)
        d2 = torch.cdist(q, cent) ** 2 / (scat[None, :] ** 2 + 1e-6)
        probs = torch.softmax(-0.5 * d2 / d2.mean(), 1).cpu().numpy()
    elif name == "atlaslam_xgb":
        import xgboost as xgb
        params = dict(objective="multi:softprob", num_class=len(classes), eta=0.15,
                      max_depth=7, subsample=0.8, colsample_bytree=0.35,
                      min_child_weight=8, reg_lambda=2.0, tree_method="hist",
                      nthread=11, seed=0)
        bst = xgb.train(params, xgb.DMatrix(Xa, label=ya), num_boost_round=220)
        probs = bst.predict(xgb.DMatrix(Xc))
    else:
        raise SystemExit(name)
    probs = np.asarray(probs, np.float32)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    np.savez_compressed(out, probs=probs, classes=classes)
    pred = classes[probs[:len(y)].argmax(1)]
    print(f"{name}: standalone {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f}) "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["atlaslam_et", "atlaslam_xgb"]):
        build(nm)
