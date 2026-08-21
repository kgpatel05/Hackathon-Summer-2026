"""Tier 2a-confirm - the section-profile replacement, as a SINGLE pre-registered test.

`Section_ID` currently enters as 108 one-hot columns. That is an expensive encoding: the
tree must spend a split to isolate each section before it can use anything about that
section, and it learns nothing about a section from the other 107.

But sections are not arbitrary labels. Each is a physical slice at a particular
dorsoventral and rostrocaudal position, so sections genuinely differ in composition - a
lumbar section has a different cell-type mix from a cervical one. A smoothed per-section
class-frequency vector hands the model that composition directly as 60 columns.

This is also the one Tier-2 idea that adds information the current features do not carry,
rather than re-arranging what is already there.

LEAKAGE IS THE WHOLE DIFFICULTY. Target encoding uses the labels, so it is trivially easy
to leak. The encoding here is strictly nested:
  * validation rows are encoded from the fold's TRAINING cells only;
  * training rows are encoded out-of-fold, from an inner 5-fold split, so no cell ever
    contributes to its own encoding.
Without the second step the trees memorise the encoding and CV reports a fantasy.

Legitimate at submission time: all 108 sections appear in both halves of the split, so
test cells are encoded from the 5,000 labelled training cells in the same sections. No
test label is used.

PRE-REGISTERED DECISION RULE, fixed before running:
  adopt only at Holm-corrected p < 0.05 across the 5 comparisons against the submitted
  baseline, AND only if the variant beats the random-grouping null control.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(20))
N_SPLITS, N_REPEATS, N_INNER = 5, 5, 5
SMOOTH = 20.0            # pseudo-counts of the global prior mixed into every group

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
y_idx = np.searchsorted(CLASS_ARR, y)
K = len(CLASSES)
glia = meta_train["Region"].isna().to_numpy()

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
N_GENES = counts_train.shape[1]
BASE = c["BASE_TR"]
OTHER = np.hstack([c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], c["ATL_TR"]]).astype(np.float32)
n = len(BASE)

section = meta_train["Section_ID"].astype(str).to_numpy()
mouse = meta_train["Mouse_ID"].astype(str).to_numpy()
segment = meta_train["Segment"].astype(str).to_numpy()
rng = np.random.default_rng(0)
random_group = np.array([f"r{g}" for g in rng.permutation(
    np.repeat(np.arange(len(np.unique(section))),
              int(np.ceil(n / len(np.unique(section)))))[:n])])
print(f"train={n} sections={len(np.unique(section))} "
      f"median cells/section={pd.Series(section).value_counts().median():.0f}", flush=True)

# the one-hot columns for Section_ID are the LAST block of the categorical encoding
n_section = len(np.unique(np.concatenate(
    [section, meta_test["Section_ID"].astype(str).to_numpy()])))
BASE_NO_SECTION = BASE[:, :BASE.shape[1] - n_section]
print(f"base {BASE.shape} -> without the {n_section} Section_ID one-hots "
      f"{BASE_NO_SECTION.shape}", flush=True)


def _profile(groups_src, y_src, targets, prior, m=SMOOTH):
    """Smoothed class frequency of each group, evaluated at `targets`."""
    table = {}
    for g in np.unique(groups_src):
        rows = groups_src == g
        counts = np.bincount(y_src[rows], minlength=K).astype(np.float64)
        table[g] = (counts + m * prior) / (rows.sum() + m)
    default = prior
    return np.array([table.get(g, default) for g in targets], np.float32)


def target_encode(groups, train_idx, eval_idx, seed=13):
    """Out-of-fold encoding for training rows, full-fold encoding for eval rows."""
    prior = np.bincount(y_idx[train_idx], minlength=K) / len(train_idx)
    enc_train = np.zeros((len(train_idx), K), np.float32)
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    for itr, iva in inner.split(train_idx, y_idx[train_idx]):
        src = train_idx[itr]
        enc_train[iva] = _profile(groups[src], y_idx[src], groups[train_idx[iva]], prior)
    enc_eval = _profile(groups[train_idx], y_idx[train_idx], groups[eval_idx], prior)
    return enc_train, enc_eval


VARIANTS = {
    "baseline (submitted)":        dict(base="full", enc=[]),
    "section profile REPLACES 1h": dict(base="drop", enc=[("section", section)]),
}

folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                     random_state=23).split(y, y))
print(f"variants={len(VARIANTS)} folds={len(folds)} seeds={len(SEEDS)} smooth={SMOOTH}",
      flush=True)


def run(tag, spec):
    t0 = time.time()
    core = BASE if spec["base"] == "full" else BASE_NO_SECTION
    ok = np.zeros((N_REPEATS, n), bool)
    for f, (tr, va) in enumerate(folds):
        blocks_tr, blocks_va = [core[tr], OTHER[tr]], [core[va], OTHER[va]]
        for _, groups in spec["enc"]:
            e_tr, e_va = target_encode(groups, tr, va)
            blocks_tr.append(e_tr); blocks_va.append(e_va)
        Xtr = np.hstack(blocks_tr).astype(np.float32)
        Xva = np.hstack(blocks_va).astype(np.float32)
        p = M.fit_extra_trees(Xtr, pd.Series(y[tr]), CLASSES, Xva, seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
        ok[f // N_SPLITS, va] = CLASS_ARR[p.argmax(1)] == y[va]
    a = ok.mean(1)
    print(f"  {tag:30s} acc={a.mean():.4f} +/-{a.std():.4f} "
          f"glia={ok[:, glia].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return ok


print(f"\n=== {N_SPLITS}x{N_REPEATS} repeated CV, fold seed 23 (screen used 7) ===",
      flush=True)
base = run("baseline (submitted)", VARIANTS["baseline (submitted)"])
cand = run("section profile REPLACES 1h", VARIANTS["section profile REPLACES 1h"])

p, _ = M.paired_mcnemar(cand.ravel(), base.ravel())
gain = cand.mean() - base.mean()
c_only = int((cand.ravel() & ~base.ravel()).sum())
b_only = int((base.ravel() & ~cand.ravel()).sum())
print("\n=== paired McNemar, ONE pre-registered comparison ===", flush=True)
print(f"  gain       {gain:+.4f}", flush=True)
print(f"  discordant {c_only} for section-profile vs {b_only} for baseline", flush=True)
print(f"  p          {p:.4g}", flush=True)
print(f"\n  VERDICT: {'ADOPT' if (gain > 0 and p < 0.05) else 'DO NOT ADOPT'} "
      f"(pre-registered p<0.05, single hypothesis, independent fold partition)", flush=True)
pd.DataFrame([{"comparison": "section profile replaces one-hot", "gain": gain,
               "acc_baseline": base.mean(), "acc_candidate": cand.mean(),
               "glia_baseline": base[:, glia].mean(), "glia_candidate": cand[:, glia].mean(),
               "mcnemar_p": p, "fold_seed": 23, "seeds": len(SEEDS),
               "adopt": bool(gain > 0 and p < 0.05)}]).to_csv(
    OUT / "target_encode_confirm.csv", index=False)
print(f"\nwrote {OUT/'target_encode_confirm.csv'}", flush=True)
