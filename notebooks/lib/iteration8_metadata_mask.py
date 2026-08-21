"""Iteration 8a - hard metadata-compatibility mask at prediction time.

Region, Excitatory_vs_Inhibitory and Segment are each a deterministic function of the
label on the training cells:

    Region                    60/60 classes have a single value
    Excitatory_vs_Inhibitory  60/60
    Segment                   NaN for every glial class; for neurons it behaves as a
                              cluster id - seg 16 -> MV_in_Gabra1, 11 -> DM_ex_Zfhx3

Iteration 3 tried a soft version of this and found it was a no-op, because the trees had
already learned the constraint. That is still almost true: the submitted prediction
violates Region 0 times, Excitatory_vs_Inhibitory 0 times, and Segment 6 times - and all
6 of those are currently wrong, so they are free to redirect.

Six cells is 0.12 pt. That is normally beneath notice, but the leaderboard of 19 August
has us 3rd at 0.7784 with 2nd place at 0.7786 - a ONE cell gap.

The mask is built from training labels only and applied as a hard zero on incompatible
classes before the argmax. No test label is involved, and the rule is fixed rather than
fitted, so there is nothing to overfit.

PRE-REGISTERED: the mask can only ever redirect a prediction the model already ranks as
impossible under training-observed metadata. Adopt if CV is non-negative - a constraint
that is provably satisfied by the training data cannot be harmful in expectation, so the
bar here is "does not hurt", not "significantly helps".
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(20))
N_SPLITS, N_REPEATS = 5, 5
MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
K = len(CLASSES)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X_TR = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
                  c["ATL_TR"]]).astype(np.float32)
X_TE = np.hstack([c["BASE_TE"], c["EXT_TE"], c["SPA_TE"], c["NIC_TE"],
                  c["ATL_TE"]]).astype(np.float32)
n = len(X_TR)
print(f"train={n} test={len(X_TE)} features={X_TR.shape[1]} classes={K}", flush=True)


def build_mask(train_rows, meta_eval):
    """(n_eval, K) boolean: is class k compatible with this cell's metadata?

    Built from `train_rows` only. A class/value pair never seen in training is
    disallowed; a metadata value never seen at all leaves the class unconstrained.
    """
    allow = np.ones((len(meta_eval), K), bool)
    for col in MASK_COLS:
        tr_vals = meta_train[col].iloc[train_rows].astype(str).to_numpy()
        tr_lab = y[train_rows]
        seen = {}
        for k, cls in enumerate(CLASSES):
            seen[k] = set(tr_vals[tr_lab == cls])
        known = set(tr_vals)
        ev = meta_eval[col].astype(str).to_numpy()
        for i, v in enumerate(ev):
            if v not in known:
                continue
            allow[i] &= np.array([v in seen[k] for k in range(K)])
    # never mask a cell down to nothing
    empty = ~allow.any(1)
    allow[empty] = True
    return allow


folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=7).split(y, y))
ok_plain = np.zeros((N_REPEATS, n), bool)
ok_mask = np.zeros((N_REPEATS, n), bool)
touched = 0
t0 = time.time()
for f, (tr, va) in enumerate(folds):
    p = M.fit_extra_trees(X_TR[tr], pd.Series(y[tr]), CLASSES, X_TR[va], seeds=SEEDS)
    p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
    plain = CLASS_ARR[p.argmax(1)]
    allow = build_mask(tr, meta_train.iloc[va])
    masked = CLASS_ARR[np.where(allow, p, -1.0).argmax(1)]
    touched += int((plain != masked).sum())
    ok_plain[f // N_SPLITS, va] = plain == y[va]
    ok_mask[f // N_SPLITS, va] = masked == y[va]
    if f % 5 == 0:
        print(f"  fold {f+1}/{len(folds)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== {N_SPLITS}x{N_REPEATS} CV, {len(SEEDS)} seeds ===", flush=True)
for tag, ok in (("no mask (submitted)", ok_plain), ("hard metadata mask", ok_mask)):
    a = ok.mean(1)
    print(f"  {tag:22s} acc={a.mean():.4f} +/-{a.std():.4f} glia={ok[:, glia].mean():.4f}",
          flush=True)
gain = ok_mask.mean() - ok_plain.mean()
p_val, _ = M.paired_mcnemar(ok_mask.ravel(), ok_plain.ravel())
print(f"\n  cells redirected across all folds: {touched}", flush=True)
print(f"  gain {gain:+.5f}  p={p_val:.4g}", flush=True)
print(f"  VERDICT: {'ADOPT' if gain >= 0 else 'DO NOT ADOPT'} (bar: does not hurt)",
      flush=True)

# ---------------------------------------------------------------- test prediction
probs = M.fit_extra_trees(X_TR, pd.Series(y), CLASSES, X_TE, seeds=SEEDS)
probs = M.correct_prior(probs, M.prior_vector(pd.Series(y), CLASSES), 0.45)
plain = CLASS_ARR[probs.argmax(1)]
allow = build_mask(np.arange(n), meta_test)
masked = CLASS_ARR[np.where(allow, probs, -1.0).argmax(1)]
changed = np.flatnonzero(plain != masked)
print(f"\n=== test set ===", flush=True)
print(f"  cells redirected by the mask: {len(changed)}", flush=True)
for i in changed:
    print(f"    {meta_test.index[i]}  {plain[i]} -> {masked[i]}", flush=True)

sub = pd.DataFrame({"Cell_ID": meta_test.index.astype(str),
                    "MERFISH_cell_type_annotation.y": masked})
assert len(sub) == 5000 and not sub.Cell_ID.duplicated().any()
assert np.array_equal(sub.Cell_ID.to_numpy(), meta_test.index.astype(str).to_numpy())
assert set(masked) <= set(CLASSES)
out = OUT / "prediction_metadata_mask.csv"
out.write_text(sub.to_csv(index=False).rstrip("\n"))
print(f"\nwrote {out} (submission NOT overwritten)", flush=True)
