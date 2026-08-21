"""DIAGNOSTIC ONLY -- which genes carry the confusions we cannot resolve?

Not importable by any module that produces a prediction.  For each dominant confusion,
fits a binary discriminant on the full published panel over NON-challenge atlas cells,
ranks the genes by contribution, and reports how the discriminative mass splits between
the released 200 and the withheld 300 -- and whether the released genes that do carry
signal have a proxy relationship with the withheld ones.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_atlas as A
import iteration19_ceiling as C19
import iteration5_features as F

PAIRS = [("oligodendrocyte_1", "oligodendrocyte_progenitor_2"),
         ("oligodendrocyte_2", "oligodendrocyte_progenitor_2"),
         ("astrocyte_1", "astrocyte_2"),
         ("astrocyte_1", "endothelial"),
         ("astrocyte_1", "oligodendrocyte_1"),
         ("meninges_1", "meninges_2")]


def main():
    atlas = A.load()
    al = atlas["labels"].astype(str)
    full = np.load(C19.FULL_CACHE, allow_pickle=True)
    order = {c: i for i, c in enumerate(full["ids"].astype(str))}
    take = np.array([order[c] for c in np.asarray(atlas["ids"]).astype(str)])
    counts = full["counts"][take].astype(np.float32)
    genes = full["genes"].astype(str)
    released = np.array([g in set(np.asarray(atlas["genes"]).astype(str)) for g in genes])
    expr = F.log_cpm(counts)
    print(f"{released.sum()} released, {(~released).sum()} withheld genes\n")

    rows = []
    for u, v in PAIRS:
        m = np.isin(al, [u, v])
        X = StandardScaler().fit_transform(expr[m])
        yb = (al[m] == v).astype(int)
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(X, yb)
        w = np.abs(lr.coef_[0])
        share = w[released].sum() / w.sum()
        top = np.argsort(-w)[:15]
        acc500 = lr.score(X, yb)
        lr2 = LogisticRegression(C=0.05, max_iter=2000).fit(X[:, released], yb)
        rows.append({"pair": f"{u}|{v}", "n": int(m.sum()),
                     "released_share_of_|weight|": share,
                     "in-sample 500g": acc500, "in-sample 200g": lr2.score(X[:, released], yb),
                     "top15 released": int(released[top].sum())})
        print(f"{u} vs {v}  (n={m.sum()})")
        print("  top 15 discriminative genes: " +
              ", ".join(f"{genes[i]}{'' if released[i] else '*'}" for i in top))
        print(f"  released share of |weight| {share:.3f} | "
              f"top-15 released {int(released[top].sum())}/15")
        # best released proxy for the withheld discriminative direction
        d_with = np.zeros(X.shape[1]); d_with[~released] = lr.coef_[0][~released]
        s_with = X @ d_with
        corr = np.array([abs(np.corrcoef(X[:, j], s_with)[0, 1]) for j in np.flatnonzero(released)])
        best = np.flatnonzero(released)[np.argsort(-corr)[:5]]
        print(f"  withheld direction best released proxies: " +
              ", ".join(f"{genes[j]} r={corr[np.argsort(-corr)][k]:.2f}"
                        for k, j in enumerate(best)))
        rid = np.flatnonzero(released)
        ridge = LogisticRegression(C=1.0, max_iter=2000)
        from sklearn.linear_model import RidgeCV
        r2 = RidgeCV(alphas=[1, 10, 100, 1000]).fit(X[:, rid], s_with)
        print(f"  R^2 of the withheld direction from ALL released genes: "
              f"{r2.score(X[:, rid], s_with):.3f}\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n* = withheld gene")


if __name__ == "__main__":
    main()
