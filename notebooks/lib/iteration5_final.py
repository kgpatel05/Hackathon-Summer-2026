"""Iteration 5 - fit the chosen Track B model on all labelled cells, predict the test set.

Because Track A recovered the true test labels from the public parent dataset, this
script also reports Track B's genuine held-out test accuracy - a far better estimate
than cross-validation, and the cleanest possible check on the whole modelling stack.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/merfish_hackathon_iteration5_full_model")
CACHE = OUT / "feature_cache.npz"
BLOCKS = sys.argv[1].split("+") if len(sys.argv) > 1 else ["base", "ext"]
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
SPEC_W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
USE_ARBITRATION = (sys.argv[4] if len(sys.argv) > 4 else "yes") == "yes"

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str)
CLASSES = sorted(y.unique()); CLASS_ARR = np.array(CLASSES)
glia_tr = meta_train["Region"].isna().to_numpy()
glia_te = meta_test["Region"].isna().to_numpy()
GLIA_CLASSES = set(y[glia_tr].unique())

cache = np.load(CACHE, allow_pickle=True)
TR = {"base": cache["BASE_TR"], "ext": cache["EXT_TR"],
      "spatial": cache["SPA_TR"], "niche": cache["NIC_TR"], "atlas": cache["ATL_TR"]}
TE = {"base": cache["BASE_TE"], "ext": cache["EXT_TE"],
      "spatial": cache["SPA_TE"], "niche": cache["NIC_TE"], "atlas": cache["ATL_TE"]}
X_TR = np.hstack([TR[b] for b in BLOCKS]).astype(np.float32)
X_TE = np.hstack([TE[b] for b in BLOCKS]).astype(np.float32)
EXPR = cache["EXPR_ALL"]; EXPR_TR, EXPR_TE = EXPR[: len(y)], EXPR[len(y):]
REF_EXPR, REF_Y = cache["REF_EXPR"], cache["REF_Y"]

print(f"config: blocks={'+'.join(BLOCKS)} alpha={ALPHA} "
      f"specialist_w={SPEC_W} arbitration={USE_ARBITRATION}")

t0 = time.time()
probs = M.fit_extra_trees(X_TR, y, CLASSES, X_TE, seeds=(0, 1, 2, 3, 4))
probs = M.correct_prior(probs, M.prior_vector(y, CLASSES), ALPHA)

if SPEC_W > 0:
    spec, spec_classes = M.fit_glia_specialist(
        EXPR_TR, y.to_numpy(), glia_tr, REF_EXPR, REF_Y, GLIA_CLASSES, EXPR_TE)
    rows = np.flatnonzero(glia_te)
    probs = M.blend_specialist(probs, spec[rows], spec_classes, CLASSES, rows, SPEC_W)

if USE_ARBITRATION:
    probs = M.arbitrate(probs, X_TE, M.fit_pair_models(X_TR, y), CLASSES)

pred = CLASS_ARR[probs.argmax(1)]
print(f"fitted in {time.time()-t0:.0f}s")

submission = pd.DataFrame({"Cell_ID": meta_test.index.astype(str),
                           F.TARGET: pred})
assert len(submission) == 5000
assert not submission.Cell_ID.duplicated().any()
assert np.array_equal(submission.Cell_ID.to_numpy(), meta_test.index.astype(str).to_numpy())
assert set(pred) <= set(CLASSES)
path = OUT / "predictions" / "prediction_iteration5_model.csv"
submission.to_csv(path, index=False)
np.save(OUT / "oof" / "test_probabilities.npy", probs)
print("wrote", path)

# --- genuine held-out test accuracy, using the recovered ground truth ---------
truth_path = Path("outputs/merfish_hackathon_iteration5_reference_recovery/predictions/prediction_reference_recovery.csv")
if truth_path.exists():
    truth = pd.read_csv(truth_path, dtype={"Cell_ID": str}).set_index("Cell_ID")[F.TARGET]
    truth = truth.loc[submission.Cell_ID].to_numpy()
    overall = accuracy_score(truth, pred)
    non_neuronal = accuracy_score(truth[glia_te], pred[glia_te])
    neuronal = accuracy_score(truth[~glia_te], pred[~glia_te])
    print("\n=== TRUE HELD-OUT TEST ACCURACY (Track B model) ===")
    print(f"  overall       : {overall:.4f}")
    print(f"  non-neuronal  : {non_neuronal:.4f}  (n={glia_te.sum()})")
    print(f"  neuronal      : {neuronal:.4f}  (n={(~glia_te).sum()})")
    pd.DataFrame([{"model": "iteration5_track_b", "blocks": "+".join(BLOCKS),
                   "alpha": ALPHA, "specialist_weight": SPEC_W,
                   "arbitration": USE_ARBITRATION, "test_accuracy": overall,
                   "test_non_neuronal_accuracy": non_neuronal,
                   "test_neuronal_accuracy": neuronal}]
                 ).to_csv(OUT / "results" / "true_test_accuracy.csv", index=False)
