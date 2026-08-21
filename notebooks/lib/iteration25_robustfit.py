"""Should the pool exponents be fitted under cohort shift rather than on random cells?

Cash prizes are decided on a validation cohort, so the deployed exponents should reflect
how good each expert is on cells from an animal it has never seen - not on cells drawn at
random from animals it has. Under leave-one-mouse-out evaluation, two fitting regimes are
compared, both scored on the SAME held-out probabilities:

  A  exponents fitted on random-cell out-of-fold probabilities  (what is deployed now)
  B  exponents fitted on leave-one-mouse-out probabilities      (matched to deployment)

If B wins, the production fit should use the cohort-shifted store.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_submit as S

EPS = 1e-9


def load(tag):
    d = np.load(B.OUT / f"experts_oof_seed{tag}.npz", allow_pickle=True)
    names = sorted(n for n in S.ADOPTED if n in d.files)
    return (np.stack([np.log(np.maximum(d[n], EPS)) for n in names]),
            names, d["allow"], d["y"].astype(str), d["classes"].astype(str))


def main(random_tags=("18", "41", "59", "83")):
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    mouse = data["meta_train"]["Mouse_ID"].astype(str).to_numpy()
    lm, names_m, allow_m, y, classes = load("mouse")
    missing = [n for n in S.ADOPTED if n not in names_m]
    if missing:
        print(f"missing from the cohort-shifted store: {missing}")
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    lp = np.log(prior)

    rand = []
    for t in random_tags:
        lr, names_r, allow_r, _, _ = load(t)
        idx = [names_r.index(n) for n in names_m]
        rand.append(lr[idx])
    lr_mean = np.concatenate(rand, axis=1)
    y_rep = np.tile(y, len(random_tags))
    allow_rep = np.tile(allow_m, (len(random_tags), 1))
    glia_rep = np.tile(glia, len(random_tags))
    mouse_rep = np.tile(mouse, len(random_tags))

    res = {}
    for tag in ("A random-cell fit", "B cohort-shifted fit"):
        pred = np.empty(len(y), dtype=object)
        for _, val in GroupKFold(n_splits=5).split(np.zeros(len(y)), y, mouse):
            held = set(mouse[val])
            for is_glia in (True, False):
                branch = glia if is_glia else ~glia
                vv = val[branch[val]]
                if len(vv) == 0:
                    continue
                if tag.startswith("A"):
                    br_rep = glia_rep if is_glia else ~glia_rep
                    rows = np.flatnonzero(~np.isin(mouse_rep, list(held)) & br_rep)
                    w, a = LP.fit(lr_mean, y_rep, classes, lp, allow_rep, rows=rows)
                else:
                    rows = np.flatnonzero(~np.isin(mouse, list(held)) & branch)
                    w, a = LP.fit(lm, y, classes, lp, allow_m, rows=rows)
                pred[vv] = classes[LP.apply(lm[:, vv], w, a, lp, allow_m[vv]).argmax(1)]
        acc = float(np.mean(pred == y))
        res[tag] = acc
        print(f"  {tag:22s} leave-one-mouse-out accuracy {acc:.4f}", flush=True)

    print(f"\ncohort-shifted fit is {100*(res['B cohort-shifted fit'] - res['A random-cell fit']):+.2f} pt")


if __name__ == "__main__":
    main()
