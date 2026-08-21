"""Do the challenge-trained experts survive a new mouse, or do they memorise cohort id?

The base feature block one-hot encodes Mouse_ID and Section_ID.  On a validation cohort
those levels are unseen, so `handle_unknown="ignore"` emits an all-zero identity block and
every split the trees learned on it becomes uninformative.  This refits the main
challenge-side experts twice - random cell folds, and leave-one-mouse-out folds - and
reports the drop.  A reference expert that never sees a challenge cell is included as a
control: its accuracy must not move at all.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_experts as E
import iteration18_experts2 as E2
import iteration15_optimal_transport as I15
import iteration5_models as M


def identity_columns(data):
    """Columns of the stack that encode Mouse_ID / Section_ID identity."""
    import iteration5_features as F
    from sklearn.preprocessing import OneHotEncoder
    meta_train, meta_test = data["meta_train"], data["meta_test"]
    enc = OneHotEncoder(handle_unknown="ignore").fit(
        pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str))
    widths = [len(c) for c in enc.categories_]
    names = list(F.CATEGORICAL_META)
    b = I15.block_offsets()
    start = b["BASE"][0] + (b["BASE"][1] - b["BASE"][0]) - sum(widths)
    cols, off = [], start
    for name, w in zip(names, widths):
        if name in ("Mouse_ID", "Section_ID"):
            cols.extend(range(off, off + w))
        off += w
    return np.array(cols, dtype=int)


def oof(X, y, classes, splits, kind, seeds=(0, 1)):
    out = np.zeros((len(y), len(classes)), np.float32)
    for fit, val in splits:
        if kind == "et":
            p = M.fit_extra_trees(X[fit], pd.Series(y[fit]), list(classes), X[val],
                                  seeds=seeds)
        elif kind == "etaug":
            from sklearn.ensemble import ExtraTreesClassifier
            p = np.zeros((len(val), len(classes)), np.float32)
            idx = {c: i for i, c in enumerate(classes)}
            for sd in seeds:
                m = ExtraTreesClassifier(n_estimators=600, max_features=0.25,
                                         min_samples_leaf=3, n_jobs=-1,
                                         random_state=sd).fit(X[fit], y[fit])
                raw = m.predict_proba(X[val])
                for j, lab in enumerate(m.classes_):
                    p[:, idx[str(lab)]] += raw[:, j]
        else:
            p = E.expert_xgb(X[fit], y[fit], X[val], classes)
        out[val] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return out


def main():
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    meta = data["meta_train"]
    allow = np.load(B.OUT / "experts_oof_seed18.npz", allow_pickle=True)["allow"]
    mouse = meta["Mouse_ID"].astype(str).to_numpy()
    cell_splits = list(StratifiedKFold(5, shuffle=True, random_state=18).split(
        np.zeros(len(y)), y))
    mouse_splits = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, mouse))

    ident = identity_columns(data)
    X694 = data["x_train"]
    Xaug = E2.augmented4(data, 18)[0]
    print(f"identity columns (Mouse_ID + Section_ID one-hot): {len(ident)} of "
          f"{X694.shape[1]}\n")

    def acc(p):
        return float(np.mean(classes[np.where(allow, p, -1).argmax(1)] == y))

    rows = []
    for tag, X, kind in (("et  (694)", X694, "et"),
                         ("etaug4 (aug)", Xaug, "etaug")):
        for split_name, splits in (("random cells", cell_splits),
                                   ("new mouse", mouse_splits)):
            t0 = time.time()
            a_full = acc(oof(X, y, classes, splits, kind))
            Xz = X.copy(); Xz[:, ident] = 0.0
            a_zero = acc(oof(Xz, y, classes, splits, kind))
            rows.append({"expert": tag, "held out": split_name,
                         "with identity": a_full, "identity zeroed": a_zero,
                         "delta": a_zero - a_full})
            print(f"  {tag:13s} {split_name:12s} with-id {a_full:.4f}  "
                  f"id-zeroed {a_zero:.4f}  ({a_zero-a_full:+.4f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    d = np.load(B.OUT / "experts_oof_seed18.npz", allow_pickle=True)
    print(f"\n  control: atlaslam_lin (never sees a challenge cell) "
          f"{acc(d['atlaslam_lin']):.4f} - unaffected by the split by construction")
    print("\n" + pd.DataFrame(rows).to_string(index=False,
                                              float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
