"""Shared Iteration-18 infrastructure: incumbent probabilities, caches, scoring.

`build` caches (a) 5-fold OOF probabilities on the 5,000 released training cells and
(b) full-fit probabilities on the 5,000 test cells, for the exact adopted 694-feature
20-seed ExtraTrees production configuration.  Nothing here reads recovered test truth.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as I15

OUT = Path("outputs/iteration18")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.45
PROD_SEEDS = tuple(range(20))
CV_SEEDS = (0, 1, 2, 3, 4)
MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]


def load_all():
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.array(sorted(set(y)))
    x_train, x_test = I15.load_incumbent()
    return dict(
        counts_train=counts_train, meta_train=meta_train,
        counts_test=counts_test, meta_test=meta_test,
        genes=np.asarray(counts_train.columns.astype(str)),
        y=y, classes=classes, x_train=x_train, x_test=x_test,
    )


def compat_mask(meta_fit, y_fit, meta_eval, classes):
    return I15.compatibility_mask(meta_fit, np.asarray(y_fit, dtype=str),
                                  meta_eval, list(classes))


def prior_correct(probs, y_fit, classes, alpha=ALPHA):
    return M.correct_prior(probs, M.prior_vector(pd.Series(y_fit), list(classes)), alpha)


def decode(probs, allow, classes):
    return np.asarray(classes)[np.where(allow, probs, -1.0).argmax(1)]


def oof_probabilities(data, n_splits=5, seed=18, et_seeds=CV_SEEDS, raw=False):
    """Fold-scoped OOF probabilities. `raw` returns pre-prior-correction values."""
    x, y, classes = data["x_train"], data["y"], data["classes"]
    meta = data["meta_train"]
    out = np.zeros((len(y), len(classes)), np.float32)
    allow = np.ones((len(y), len(classes)), bool)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (fit, val) in enumerate(skf.split(x, y)):
        t0 = time.time()
        p = M.fit_extra_trees(x[fit], pd.Series(y[fit]), list(classes), x[val],
                              seeds=et_seeds)
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        if not raw:
            p = prior_correct(p, y[fit], classes)
        out[val] = p
        allow[val] = compat_mask(meta.iloc[fit], y[fit], meta.iloc[val], classes)
        print(f"  fold {k+1}/{n_splits} ({time.time()-t0:.0f}s)", flush=True)
    return out, allow


def test_probabilities(data, et_seeds=PROD_SEEDS, raw=False):
    x, y, classes = data["x_train"], data["y"], data["classes"]
    p = M.fit_extra_trees(x, pd.Series(y), list(classes), data["x_test"], seeds=et_seeds)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    if not raw:
        p = prior_correct(p, y, classes)
    allow = compat_mask(data["meta_train"], y, data["meta_test"], classes)
    return p, allow


def main():
    data = load_all()
    cache = OUT / "incumbent_probs.npz"
    t0 = time.time()
    print("OOF (5-fold, 5 ET seeds) ...", flush=True)
    oof_raw, oof_allow = oof_probabilities(data, raw=True)
    print(f"test (20 ET seeds) ...", flush=True)
    test_raw, test_allow = test_probabilities(data, raw=True)
    np.savez_compressed(cache, oof_raw=oof_raw, oof_allow=oof_allow,
                        test_raw=test_raw, test_allow=test_allow,
                        classes=data["classes"], y=data["y"])
    print(f"wrote {cache} in {time.time()-t0:.0f}s")

    oof = prior_correct(oof_raw, data["y"], data["classes"])
    pred = decode(oof, oof_allow, data["classes"])
    print(f"OOF accuracy {np.mean(pred == data['y']):.4f}")


if __name__ == "__main__":
    main()
