"""Iteration 5 - Track B stage 2: specialist, arbitration, stacking, submission."""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/merfish_hackathon_iteration5_full_model")
CACHE = OUT / "feature_cache.npz"
N_SPLITS = 5
N_REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BEST_BLOCKS = sys.argv[2].split("+") if len(sys.argv) > 2 else ["base", "ext"]

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str)
y_arr = y.to_numpy()
CLASSES = sorted(y.unique())
CLASS_ARR = np.array(CLASSES)
glia_mask = meta_train["Region"].isna().to_numpy()
glia_mask_test = meta_test["Region"].isna().to_numpy()
GLIA_CLASSES = set(y[glia_mask].unique())
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

cache = np.load(CACHE, allow_pickle=True)
BLOCKS_TR = {"base": cache["BASE_TR"], "ext": cache["EXT_TR"],
             "spatial": cache["SPA_TR"], "niche": cache["NIC_TR"]}
BLOCKS_TE = {"base": cache["BASE_TE"], "ext": cache["EXT_TE"],
             "spatial": cache["SPA_TE"], "niche": cache["NIC_TE"]}
REF_EXPR, REF_Y = cache["REF_EXPR"], cache["REF_Y"]
EXPR_ALL = cache["EXPR_ALL"]
EXPR_TR, EXPR_TE = EXPR_ALL[: len(y)], EXPR_ALL[len(y):]
COUNTS_TR = counts_train.to_numpy(np.float32)
COUNTS_TE = counts_test.to_numpy(np.float32)

X_TR = np.hstack([BLOCKS_TR[b] for b in BEST_BLOCKS]).astype(np.float32)
X_TE = np.hstack([BLOCKS_TE[b] for b in BEST_BLOCKS]).astype(np.float32)
print(f"blocks={'+'.join(BEST_BLOCKS)} features={X_TR.shape[1]}")

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
folds = list(cv.split(X_TR, y))


def evaluate(name, fold_fn):
    correct = np.zeros((N_REPEATS, len(y)), bool)
    t0 = time.time()
    for i, (tr, va) in enumerate(folds):
        pred = fold_fn(tr, va)
        correct[i // N_SPLITS, va] = pred == y_arr[va]
    acc = correct.mean(1)
    glia = correct[:, glia_mask].mean(1)
    print(f"{name:44s} acc={acc.mean():.4f} ±{acc.std(ddof=0):.4f} "
          f"non-neuro={glia.mean():.4f}  ({time.time()-t0:.0f}s)")
    return {"strategy": name, "accuracy": acc.mean(), "accuracy_sd": acc.std(ddof=0),
            "non_neuronal_accuracy": glia.mean(), "seconds": time.time() - t0}, correct


def global_probs(tr, va, alpha):
    probs = M.fit_extra_trees(X_TR[tr], y.iloc[tr], CLASSES, X_TR[va])
    return M.correct_prior(probs, M.prior_vector(y.iloc[tr], CLASSES), alpha)


rows, store = [], {}

# --- alpha selection -------------------------------------------------------
for alpha in [0.35, 0.45, 0.55]:
    row, correct = evaluate(f"A_reference_alpha_{alpha}", 
                            lambda tr, va, a=alpha: CLASS_ARR[global_probs(tr, va, a).argmax(1)])
    rows.append(row); store[row["strategy"]] = correct
BEST_ALPHA = float(max(rows, key=lambda r: r["accuracy"])["strategy"].split("_")[-1])
BASELINE_KEY = f"A_reference_alpha_{BEST_ALPHA}"
print(f"--> selected alpha={BEST_ALPHA}\n")

# --- B4 glia specialist ----------------------------------------------------
def with_specialist(tr, va, weight):
    probs = global_probs(tr, va, BEST_ALPHA)
    tr_glia = glia_mask[tr]
    if tr_glia.sum() == 0:
        return CLASS_ARR[probs.argmax(1)]
    spec, spec_classes = M.fit_glia_specialist(
        EXPR_TR[tr], y_arr[tr], tr_glia, REF_EXPR, REF_Y, GLIA_CLASSES, EXPR_TR[va]
    )
    rows_va = np.flatnonzero(glia_mask[va])
    probs = M.blend_specialist(probs, spec[rows_va], spec_classes, CLASSES, rows_va, weight)
    return CLASS_ARR[probs.argmax(1)]

for weight in [0.25, 0.40]:
    row, correct = evaluate(f"B_glia_specialist_w{weight}",
                            lambda tr, va, w=weight: with_specialist(tr, va, w))
    rows.append(row); store[row["strategy"]] = correct

# --- B5 pairwise arbitration ----------------------------------------------
def with_arbitration(tr, va):
    probs = global_probs(tr, va, BEST_ALPHA)
    models = M.fit_pair_models(X_TR[tr], y.iloc[tr])
    probs = M.arbitrate(probs, X_TR[va], models, CLASSES)
    return CLASS_ARR[probs.argmax(1)]

row, correct = evaluate("C_pairwise_arbitration", with_arbitration)
rows.append(row); store[row["strategy"]] = correct

# --- B6 + B9 count-aware member and learned stacking -----------------------
def with_stacking(tr, va):
    inner = StratifiedKFold(3, shuffle=True, random_state=7)
    members_oof = {k: np.zeros((len(tr), len(CLASSES)), np.float32)
                   for k in ["et", "nb"]}
    for itr, iva in inner.split(X_TR[tr], y.iloc[tr]):
        gi, gv = tr[itr], tr[iva]
        members_oof["et"][iva] = M.fit_extra_trees(
            X_TR[gi], y.iloc[gi], CLASSES, X_TR[gv], seeds=(0,))
        members_oof["nb"][iva] = M.fit_multinomial_nb(
            COUNTS_TR[gi], y.iloc[gi], CLASSES, COUNTS_TR[gv])

    meta_X = np.hstack([members_oof["et"], members_oof["nb"], BLOCKS_TR["ext"][tr]])
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, max_iter=1500, n_jobs=-1).fit(meta_X, y.iloc[tr])

    full_et = M.fit_extra_trees(X_TR[tr], y.iloc[tr], CLASSES, X_TR[va])
    full_nb = M.fit_multinomial_nb(COUNTS_TR[tr], y.iloc[tr], CLASSES, COUNTS_TR[va])
    probs = M.align_proba(meta, np.hstack([full_et, full_nb, BLOCKS_TR["ext"][va]]), CLASSES)
    return CLASS_ARR[probs.argmax(1)]

row, correct = evaluate("D_learned_stacking_et_nb", with_stacking)
rows.append(row); store[row["strategy"]] = correct

# --- combined best ---------------------------------------------------------
def combined(tr, va, weight=0.25):
    probs = global_probs(tr, va, BEST_ALPHA)
    tr_glia = glia_mask[tr]
    if tr_glia.sum():
        spec, spec_classes = M.fit_glia_specialist(
            EXPR_TR[tr], y_arr[tr], tr_glia, REF_EXPR, REF_Y, GLIA_CLASSES, EXPR_TR[va])
        rows_va = np.flatnonzero(glia_mask[va])
        probs = M.blend_specialist(probs, spec[rows_va], spec_classes, CLASSES, rows_va, weight)
    models = M.fit_pair_models(X_TR[tr], y.iloc[tr])
    probs = M.arbitrate(probs, X_TR[va], models, CLASSES)
    return CLASS_ARR[probs.argmax(1)]

row, correct = evaluate("E_specialist_plus_arbitration", combined)
rows.append(row); store[row["strategy"]] = correct

frame = pd.DataFrame(rows)
base_correct = store[BASELINE_KEY].ravel()
frame["mcnemar_p_vs_best_alpha"] = [
    M.paired_mcnemar(store[n].ravel(), base_correct)[0] for n in frame.strategy
]
frame.to_csv(OUT / "results" / "stage2_strategies.csv", index=False)
np.savez_compressed(OUT / "oof" / "stage2_correct.npz", **store)
print("\n", frame.to_string(index=False))
print(f"\nBEST: {frame.loc[frame.accuracy.idxmax(), 'strategy']} "
      f"= {frame.accuracy.max():.4f}")
