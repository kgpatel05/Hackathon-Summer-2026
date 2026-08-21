"""Iteration 5 - Track B driver: build feature blocks, scan strategies, submit."""
import sys, time, json
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
for sub in ["results", "oof", "predictions"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

N_SPLITS, N_REPEATS, ALPHA = 5, int(sys.argv[1]) if len(sys.argv) > 1 else 3, 0.45

CACHE = OUT / "feature_cache.npz"

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str)
CLASSES = sorted(y.unique())
GENES = list(counts_train.columns)
glia_mask_train = meta_train["Region"].isna().to_numpy()
GLIA_CLASSES = set(y[glia_mask_train].unique())

print(f"train={len(y)} test={len(meta_test)} genes={len(GENES)} classes={len(CLASSES)}")
print(f"non-neuronal classes={len(GLIA_CLASSES)} cells={glia_mask_train.sum()}")

meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

# ---------------------------------------------------------------- feature blocks
if CACHE.exists():
    _c = np.load(CACHE, allow_pickle=True)
    _c_keys = set(_c.files)
    BASE_TR, BASE_TE = _c["BASE_TR"], _c["BASE_TE"]
    EXT_TR, EXT_TE = _c["EXT_TR"], _c["EXT_TE"]
    SPA_TR, SPA_TE = _c["SPA_TR"], _c["SPA_TE"]
    NIC_TR, NIC_TE = _c["NIC_TR"], _c["NIC_TE"]
    REF_EXPR, REF_Y = _c["REF_EXPR"], _c["REF_Y"]
    EXPR_ALL = _c["EXPR_ALL"]
    print(f"[cache] loaded feature blocks from {CACHE}")
    standalone = np.array(CLASSES)[EXT_TR.argmax(1)]
    print(f"[external] standalone overall={accuracy_score(y, standalone):.4f} "
          f"non-neuronal={accuracy_score(y[glia_mask_train], standalone[glia_mask_train]):.4f}")
else:
  _c_keys = set()
  t0 = time.time()
  encoder = OneHotEncoder(handle_unknown="ignore").fit(
      pd.concat([meta_train[F.CATEGORICAL_META], meta_test[F.CATEGORICAL_META]]).astype(str)
  )
  BASE_TR = F.base_block(counts_train, meta_train, encoder)
  BASE_TE = F.base_block(counts_test, meta_test, encoder)
  print(f"[base] {BASE_TR.shape} ({time.time()-t0:.0f}s)")

  t0 = time.time()
  (EXT_TR, EXT_TE), REF_X, REF_Y = F.reference_transfer(
      GENES, CLASSES, [counts_train, counts_test], label_column="voting"
  )
  assert EXT_TR.shape[1] == 60, "reference transfer must emit 60 class columns"
  assert set(REF_Y) <= set(CLASSES), "reference taxonomy does not match the challenge"
  standalone = np.array(CLASSES)[EXT_TR.argmax(1)]
  print(f"[external] reference cells={len(REF_X)} label column='voting' "
        f"({time.time()-t0:.0f}s)")
  print(f"[external] standalone accuracy overall={accuracy_score(y, standalone):.4f} "
        f"non-neuronal={accuracy_score(y[glia_mask_train], standalone[glia_mask_train]):.4f}")

  neuron_all = (~meta_all["Region"].isna()).to_numpy() & (meta_all["Region"] == 1).to_numpy()
  t0 = time.time()
  SPATIAL = F.registered_spatial(meta_all, neuron_all)
  SPA_TR, SPA_TE = SPATIAL[: len(meta_train)], SPATIAL[len(meta_train):]
  print(f"[spatial] registered {SPATIAL.shape} ({time.time()-t0:.0f}s)")

  t0 = time.time()
  EXPR_ALL = F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()]))
  NICHE = F.niche_expression(EXPR_ALL, meta_all, k=15, n_components=30)
  NIC_TR, NIC_TE = NICHE[: len(meta_train)], NICHE[len(meta_train):]
  print(f"[niche] {NICHE.shape} ({time.time()-t0:.0f}s)")

  np.savez_compressed(CACHE, BASE_TR=BASE_TR, BASE_TE=BASE_TE, EXT_TR=EXT_TR,
                      EXT_TE=EXT_TE, SPA_TR=SPA_TR, SPA_TE=SPA_TE, NIC_TR=NIC_TR,
                      NIC_TE=NIC_TE, REF_EXPR=F.log_cpm(REF_X), REF_Y=REF_Y,
                      EXPR_ALL=EXPR_ALL)
  print(f"[cache] saved {CACHE}")

EXPR_TR, EXPR_TE = EXPR_ALL[: len(meta_train)], EXPR_ALL[len(meta_train):]
COUNTS_TR = counts_train.to_numpy(np.float32)
COUNTS_TE = counts_test.to_numpy(np.float32)

BLOCKS_TR = {"base": BASE_TR, "ext": EXT_TR, "spatial": SPA_TR, "niche": NIC_TR}
BLOCKS_TE = {"base": BASE_TE, "ext": EXT_TE, "spatial": SPA_TE, "niche": NIC_TE}
if "ATL_TR" in _c_keys:
    BLOCKS_TR["atlas"], BLOCKS_TE["atlas"] = _c["ATL_TR"], _c["ATL_TE"]

_PLACEHOLDER_STRATEGIES = {
    "S0_base_iteration2_repro":          ["base"],
    "S1_base_plus_external":             ["base", "ext"],
    "S2_S1_plus_registered_spatial":     ["base", "ext", "spatial"],
    "S3_S1_plus_niche":                  ["base", "ext", "niche"],
    "S4_S1_plus_spatial_plus_niche":     ["base", "ext", "spatial", "niche"],
}

STRATEGIES = _PLACEHOLDER_STRATEGIES
if "atlas" in BLOCKS_TR:
    STRATEGIES["S5_S2_plus_atlas_transfer"] = ["base", "ext", "spatial", "atlas"]


def assemble(blocks, table):
    return np.hstack([table[b] for b in blocks]).astype(np.float32)

# ---------------------------------------------------------------- evaluation
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
folds = list(cv.split(BASE_TR, y))
y_arr = y.to_numpy()

results, oof_store = [], {}
for name, blocks in STRATEGIES.items():
    X = assemble(blocks, BLOCKS_TR)
    t0 = time.time()
    per_repeat, correct = [], np.zeros((N_REPEATS, len(y)), bool)
    probs_sum = np.zeros((len(y), len(CLASSES)), np.float32)

    for i, (tr, va) in enumerate(folds):
        repeat = i // N_SPLITS
        probs = M.fit_extra_trees(X[tr], y.iloc[tr], CLASSES, X[va])
        probs = M.correct_prior(probs, M.prior_vector(y.iloc[tr], CLASSES), ALPHA)
        pred = np.array(CLASSES)[probs.argmax(1)]
        correct[repeat, va] = pred == y_arr[va]
        probs_sum[va] += probs

    per_repeat = correct.mean(1)
    oof_store[name] = correct
    glia_acc = correct[:, glia_mask_train].mean(1)
    results.append({
        "strategy": name, "blocks": "+".join(blocks),
        "accuracy": per_repeat.mean(), "accuracy_sd": per_repeat.std(ddof=0),
        "non_neuronal_accuracy": glia_acc.mean(),
        "n_features": X.shape[1], "seconds": time.time() - t0,
    })
    print(f"{name:34s} acc={per_repeat.mean():.4f} ±{per_repeat.std(ddof=0):.4f} "
          f"non-neuro={glia_acc.mean():.4f}  ({time.time()-t0:.0f}s)")

frame = pd.DataFrame(results)
baseline = oof_store["S0_base_iteration2_repro"].ravel()
frame["mcnemar_p_vs_S0"] = [
    M.paired_mcnemar(oof_store[n].ravel(), baseline)[0] for n in frame.strategy
]
frame.to_csv(OUT / "results" / "strategy_scan.csv", index=False)
np.savez_compressed(OUT / "oof" / "strategy_correct.npz", **oof_store)
print("\n", frame[["strategy", "accuracy", "non_neuronal_accuracy", "mcnemar_p_vs_S0"]].to_string(index=False))
