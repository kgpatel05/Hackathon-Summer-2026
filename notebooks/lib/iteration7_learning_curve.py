"""Iteration 7 - is there ANY estimation gap left for pretraining to close?

Pretraining can only help where the model is sample-limited. So: hold the feature set
fixed and vary the amount of labelled training data. If accuracy is still climbing at
100% of the training set, more/better-initialised parameters can help. If it has
flattened, pretraining has nothing to close.

Run separately for neurons and glia, because they are different problems (neurons carry
`Segment`, glia do not) and neurons are 36.5% of the test set - untested until now.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7"); OUT.mkdir(parents=True, exist_ok=True)
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
glia = meta_train["Region"].isna().to_numpy()

cache = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
                allow_pickle=True)
X = np.hstack([cache[k] for k in ["BASE_TR", "EXT_TR", "SPA_TR", "ATL_TR"]]).astype(np.float32)

FRACTIONS = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
rows = []

def curve(name, mask):
    Xs, ys = X[mask], y[mask]
    classes = sorted(set(ys))
    print(f"\n=== {name}: {len(ys)} cells, {len(classes)} classes ===", flush=True)
    for frac in FRACTIONS:
        accs, bals = [], []
        for fold, (tr, va) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys)):
            if frac < 1.0:
                # subsample the TRAIN part only; validation stays full size
                keep, _ = next(iter(StratifiedShuffleSplit(
                    1, train_size=frac, random_state=fold).split(Xs[tr], ys[tr])))
                tr = tr[keep]
            p = M.fit_extra_trees(Xs[tr], pd.Series(ys[tr]), classes, Xs[va], seeds=(0, 1))
            p = M.correct_prior(p, M.prior_vector(ys[tr], classes), 0.45)
            pred = np.array(classes)[p.argmax(1)]
            accs.append(accuracy_score(ys[va], pred))
            bals.append(balanced_accuracy_score(ys[va], pred))
        a, b = float(np.mean(accs)), float(np.mean(bals))
        n = int(frac * len(ys) * 0.8)
        print(f"  {frac*100:5.0f}% of train (~{n:5d} cells)  acc={a:.4f} bal={b:.4f}",
              flush=True)
        rows.append({"group": name, "fraction": frac, "n_train": n,
                     "accuracy": a, "balanced": b})

t0 = time.time()
curve("neurons", ~glia)
curve("glia", glia)
curve("all cells", np.ones(len(y), bool))

df = pd.DataFrame(rows)
df.to_csv(OUT / "learning_curve.csv", index=False)

print("\n=== marginal gain from the last data doubling ===", flush=True)
for g in df.group.unique():
    s = df[df.group == g].sort_values("fraction")
    d50 = s[s.fraction == 0.5].accuracy.iloc[0]
    d100 = s[s.fraction == 1.0].accuracy.iloc[0]
    d85 = s[s.fraction == 0.85].accuracy.iloc[0]
    print(f"  {g:10s} 50%->100%: {d100-d50:+.4f}   85%->100%: {d100-d85:+.4f}", flush=True)
print(f"\ntotal {time.time()-t0:.0f}s -> {OUT/'learning_curve.csv'}", flush=True)
