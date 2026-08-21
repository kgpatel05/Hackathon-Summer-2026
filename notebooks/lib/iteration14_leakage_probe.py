"""Iteration 14 - DIAGNOSTIC ONLY: what does test-label leakage buy?

*** NOTHING THIS SCRIPT PRODUCES MAY EVER BE SUBMITTED. ***

It trains on recovered TEST LABELS, which is disqualifying under any reading of the
rules. It writes only to outputs/quarantine/ (git-ignored) and never touches
prediction/prediction.csv. It exists to measure a quantity, not to produce a model.

THE EXPERIMENT
--------------
Replace a fraction f of the 5,000 training cells with f*5,000 TEST cells carrying their
true labels, keeping the training set size constant at 5,000. Fit the deployed
694-feature model. Then report TWO numbers that are usually conflated:

  ALL-TEST accuracy   over all 5,000 test cells, including the leaked ones. This is what
                      a leaderboard would show. It is dominated by memorisation: the
                      model has literally seen those cells with their answers.

  HELD-OUT accuracy   over ONLY the test cells that were NOT leaked. This is the honest
                      question - does seeing part of the test set teach the model
                      anything transferable about the rest of it?

WHAT THE ARITHMETIC ALREADY PREDICTS
------------------------------------
If leaked cells are memorised perfectly and nothing else changes, ALL-TEST accuracy is
    f * 1.0 + (1 - f) * 0.789
i.e. +2.1 pt at f=0.10, +3.2 pt at f=0.15, +5.3 pt at f=0.25. Recovering that curve
would confirm the gain is pure memorisation and worth nothing.

The interesting outcome would be HELD-OUT accuracy rising materially above 0.789, which
would mean the test cells carry distributional information the training cells lack. Given
that train and test are an IID split of the same 10 mice, 6 batches and 108 sections
(SCORECARD S10o), the prior is that HELD-OUT stays flat - there is no domain shift for
extra in-domain labels to correct.

A control is included: the same number of test cells added with SHUFFLED labels. If
ALL-TEST accuracy still rises there, the gain is coming from something other than the
labels being correct.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/quarantine")
OUT.mkdir(parents=True, exist_ok=True)
FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.25)
REPEATS = 3
SEEDS = tuple(range(10))
ALPHA = 0.45

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y_train = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y_train))
CLASS_ARR = np.array(CLASSES)
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
n_tr, n_te = len(meta_train), len(meta_test)
GENES = list(counts_train.columns)

# recovered test labels - the thing that makes this disqualifying
key = pd.read_csv(OUT / "answer_key_lookup_DO_NOT_SUBMIT.csv", dtype=str)
key = dict(zip(key.iloc[:, 0].str.strip(), key.iloc[:, 1].str.strip()))
y_test = np.array([key[c] for c in meta_test.index.astype(str)])
print(f"train={n_tr} test={n_te} classes={len(CLASSES)}", flush=True)

# ---------------------------------------------------------------- deployed 694 stack
c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
COMP = F.atlas_composition(meta_all, CLASSES, k=10)
ANIC = F.atlas_niche(meta_all, GENES, k=50, n_components=30)
blk = np.load("outputs/iteration9/atlas_et_block.npz", allow_pickle=True)
X_TR = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"],
                  COMP[:n_tr], ANIC[:n_tr], c["ATL_TR"],
                  blk["ATL_ET_TR"], blk["COARSE_TR"]]).astype(np.float32)
X_TE = np.hstack([c["BASE_TE"], c["EXT_TE"], c["SPA_TE"], c["NIC_TE"],
                  COMP[n_tr:], ANIC[n_tr:], c["ATL_TE"],
                  blk["ATL_ET_TE"], blk["COARSE_TE"]]).astype(np.float32)
print(f"feature stacks {X_TR.shape} / {X_TE.shape}", flush=True)

MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]


def metadata_mask(meta_eval, meta_fit, y_fit):
    allow = np.ones((len(meta_eval), len(CLASSES)), bool)
    for col in MASK_COLS:
        vals = meta_fit[col].astype(str).to_numpy()
        seen = [set(vals[y_fit == cls]) for cls in CLASSES]
        known = set(vals)
        for i, v in enumerate(meta_eval[col].astype(str).to_numpy()):
            if v in known:
                allow[i] &= np.array([v in s for s in seen])
    allow[~allow.any(1)] = True
    return allow


def run(frac, rep, shuffle_labels=False):
    rng = np.random.default_rng(1000 * rep + int(frac * 1000))
    k = int(round(frac * n_tr))
    if k == 0:
        Xf, yf, mf, leaked = X_TR, y_train, meta_train, np.zeros(n_te, bool)
    else:
        leaked_idx = rng.choice(n_te, size=k, replace=False)
        drop_idx = rng.choice(n_tr, size=k, replace=False)
        keep = np.setdiff1d(np.arange(n_tr), drop_idx)
        y_leak = y_test[leaked_idx]
        if shuffle_labels:
            y_leak = rng.permutation(y_leak)
        Xf = np.vstack([X_TR[keep], X_TE[leaked_idx]])
        yf = np.concatenate([y_train[keep], y_leak])
        mf = pd.concat([meta_train.iloc[keep].drop(columns=[F.TARGET]),
                        meta_test.iloc[leaked_idx]])
        leaked = np.zeros(n_te, bool); leaked[leaked_idx] = True

    p = M.fit_extra_trees(Xf, pd.Series(yf), CLASSES, X_TE, seeds=SEEDS)
    p = M.correct_prior(p, M.prior_vector(pd.Series(yf), CLASSES), ALPHA)
    allow = metadata_mask(meta_test, mf, yf)
    pred = CLASS_ARR[np.where(allow, p, -1.0).argmax(1)]
    hit = pred == y_test
    return {"fraction": frac, "repeat": rep, "shuffled": shuffle_labels,
            "n_leaked": int(leaked.sum()),
            "all_test": hit.mean(),
            "held_out": hit[~leaked].mean() if (~leaked).any() else np.nan,
            "on_leaked": hit[leaked].mean() if leaked.any() else np.nan}


rows = []
print("\n=== true labels leaked ===", flush=True)
for frac in FRACTIONS:
    for rep in range(REPEATS if frac > 0 else 1):
        t0 = time.time()
        r = run(frac, rep)
        rows.append(r)
        print(f"  f={frac:.2f} rep{rep}  ALL-TEST {r['all_test']:.4f}  "
              f"HELD-OUT {r['held_out']:.4f}  on-leaked "
              f"{r['on_leaked'] if not np.isnan(r['on_leaked']) else float('nan'):.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== control: same cells added with SHUFFLED labels ===", flush=True)
for frac in (0.15,):
    for rep in range(REPEATS):
        r = run(frac, rep, shuffle_labels=True)
        rows.append(r)
        print(f"  f={frac:.2f} rep{rep}  ALL-TEST {r['all_test']:.4f}  "
              f"HELD-OUT {r['held_out']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "leakage_probe_DO_NOT_SUBMIT.csv", index=False)

print("\n=== summary (mean over repeats) ===", flush=True)
real = df[~df.shuffled].groupby("fraction").agg(
    all_test=("all_test", "mean"), held_out=("held_out", "mean"),
    on_leaked=("on_leaked", "mean"))
base_all = real.loc[0.0, "all_test"]
base_held = real.loc[0.0, "held_out"]
real["all_gain_pt"] = 100 * (real.all_test - base_all)
real["held_gain_pt"] = 100 * (real.held_out - base_held)
real["predicted_if_pure_memorisation"] = [
    f * 1.0 + (1 - f) * base_all for f in real.index]
print(real.round(4).to_string(), flush=True)

print(f"\nwrote {OUT/'leakage_probe_DO_NOT_SUBMIT.csv'} (quarantined, git-ignored)",
      flush=True)
print("NOTHING HERE IS SUBMITTABLE - it is trained on recovered test labels.", flush=True)
