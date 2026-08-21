"""The (cell, class) candidate ranker packaged as a pool expert.

On its own the ranker only matches the flat model, but it reaches that accuracy through
per-(cell, class) evidence - atlas multinomial likelihood, per-class kNN distance,
multi-scale per-class neighbourhood composition - that no flat expert can express, so its
errors are structurally different.  Scores over the top-K candidates are softmaxed into a
60-vector with a small floor for the classes outside the candidate set.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_classfeat as CF
import iteration18_ranker as R1
import iteration18_ranker2 as R2

FLOOR = 1e-5
PARAMS = dict(objective="rank:pairwise", eta=0.04, max_depth=6, subsample=0.85,
              colsample_bytree=0.65, min_child_weight=8, reg_lambda=3.0,
              tree_method="hist", nthread=11)


def _to_vector(order, score, n_class, temperature=1.0):
    out = np.full((len(order), n_class), FLOOR, np.float32)
    p = softmax(score / temperature, axis=1)
    np.put_along_axis(out, order, np.maximum(p, FLOOR).astype(np.float32), axis=1)
    return out / out.sum(1, keepdims=True)


def build(seed=18, rounds=450, folds=5, seeds=(0, 1, 2)):
    import xgboost as xgb
    out_path = B.OUT / f"rankexpert_seed{seed}.npz"
    if out_path.exists():
        print(f"rankexpert {seed}: cached"); return
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    d = np.load(B.OUT / f"experts_oof_seed{seed}.npz", allow_pickle=True)
    probs, allow = B.prior_correct(d["et"], y, classes), d["allow"]
    cf = CF.load()
    pcs = R2.cell_pcs(data)
    P, order, names, _ = R2.assemble(data, probs, allow, cf, 0, pcs)
    n, K, Fdim = P.shape
    cand = classes[order]
    label = (cand == y[:, None]).astype(int)

    score = np.zeros((n, K), np.float32)
    t0 = time.time()
    for fit, val in StratifiedKFold(folds, shuffle=True, random_state=seed).split(
            np.zeros(n), y):
        dtr = xgb.DMatrix(P[fit].reshape(-1, Fdim), label=label[fit].reshape(-1),
                          feature_names=names)
        dtr.set_group(np.full(len(fit), K))
        dva = xgb.DMatrix(P[val].reshape(-1, Fdim), feature_names=names)
        acc = np.zeros(len(val) * K, np.float32)
        for s in seeds:
            acc += xgb.train({**PARAMS, "seed": s}, dtr, num_boost_round=rounds
                             ).predict(dva)
        score[val] = (acc / len(seeds)).reshape(len(val), K)
    vec = _to_vector(order, score, len(classes))
    np.savez_compressed(out_path, probs=vec, classes=classes)
    acc = np.mean(classes[np.where(allow, vec, -1).argmax(1)] == y)
    print(f"rankexpert partition {seed}: OOF {acc:.4f} ({time.time()-t0:.0f}s)")


def build_test(rounds=450, seeds=(0, 1, 2)):
    import xgboost as xgb
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    d = np.load(B.OUT / "experts_oof_seed18.npz", allow_pickle=True)
    t = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    cf = CF.load()
    pcs = R2.cell_pcs(data)
    Ptr, order_tr, names, _ = R2.assemble(
        data, B.prior_correct(d["et"], y, classes), d["allow"], cf, 0, pcs)
    Pte, order_te, _, _ = R2.assemble(
        data, B.prior_correct(t["et"], y, classes), t["allow"], cf, len(y), pcs)
    n, K, Fdim = Ptr.shape
    label = (classes[order_tr] == y[:, None]).astype(int)
    dtr = xgb.DMatrix(Ptr.reshape(-1, Fdim), label=label.reshape(-1),
                      feature_names=names)
    dtr.set_group(np.full(n, K))
    dte = xgb.DMatrix(Pte.reshape(-1, Fdim), feature_names=names)
    acc = np.zeros(len(Pte) * K, np.float32)
    for s in seeds:
        acc += xgb.train({**PARAMS, "seed": s}, dtr, num_boost_round=rounds).predict(dte)
    vec = _to_vector(order_te, (acc / len(seeds)).reshape(len(Pte), K), len(classes))
    np.savez_compressed(B.OUT / "rankexpert_test.npz", probs=vec, classes=classes)
    print("wrote rankexpert_test.npz")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "test":
        build_test()
    else:
        for s in [int(x) for x in (sys.argv[1:] or [18])]:
            build(s)
