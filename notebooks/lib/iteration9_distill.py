"""Iteration 9 - knowledge distillation from a 500-gene atlas TEACHER into a 200-gene
STUDENT, both trained entirely on non-challenge parent-atlas cells.

MECHANISM
---------
The 60-class taxonomy was produced by clustering the parent atlas on all 500 panel genes.
The organisers released 200 of them.  Our atlas block is a logistic probe fitted on the
136,612 non-challenge atlas cells with HARD labels: each cell contributes one of 60
symbols, ~5.9 bits.  A teacher that reads all 500 genes can instead emit a full 60-way
posterior per atlas cell - "0.6 oligodendrocyte_1 / 0.3 oligodendrocyte_2" - which encodes
the inter-class similarity geometry that our errors live in (SCORECARD 10d: the dominant
confusions are oligo_1/oligo_progenitor_2, astrocyte_1/_2, meninges_1/_2).  Training the
200-gene student against those soft targets is the textbook "dark knowledge" transfer.

The student is a function of the 200 released genes only, at fit time and at predict time.

PRIOR EVIDENCE - READ THIS BEFORE RUNNING
-----------------------------------------
Cheap diagnostics run before this script was written (60,000 non-challenge atlas cells,
teacher on split A = 20k, student on split B, held-out split C; MPS, torch):

  teacher, 500 genes, held-out atlas                     0.8379
  teacher, 200 genes, held-out atlas (the null teacher)  0.5671

  student trained on 20k atlas cells, evaluated standalone on the 5,000 labelled
  challenge training cells:
    mlp256  HARD labels                          0.5584
    mlp256  distilled from 500-gene teacher T=4   0.5794   (+2.10 pt)
    mlp256  distilled from 200-gene teacher T=4   0.5706   (+1.22 pt)   <-- NULL CONTROL
    linear  HARD labels                          0.5578
    linear  distilled from 500-gene teacher T=2   0.5608   (+0.30 pt)

  student trained on 40k atlas cells (same eval):
    mlp256  HARD labels                          0.5808
    mlp256  distilled from 500-gene teacher T=4   0.5910   (+1.02 pt)
    mlp256  distilled from 200-gene teacher T=4   0.5699   (-1.09 pt)
    linear  HARD labels                          0.5832
    linear  distilled from 500-gene teacher T=4   0.5806   (-0.26 pt)   <-- incumbent form

  student trained on ALL 136,621 non-challenge atlas cells, 25 epochs, same eval
  (this is the scale this script actually runs at):
    teacher 500 genes, 2-fold out-of-fold atlas accuracy   0.8627
    mlp256  HARD labels                          0.6110
    mlp256  distilled from 500-gene teacher T=4   0.6098   (-0.12 pt)
    mlp256  0.5 hard + 0.5 distilled T=4          0.6132   (+0.22 pt)
    incumbent deployed logistic probe            0.6040   (SCORECARD 11d)

Four facts follow, and they are why the expected value of this script is near zero:

  1. The dark-knowledge increment DECAYS TO ZERO as the student's data grows:
     +2.10 pt at 20k cells, +1.02 pt at 40k, -0.12 pt at 136,621.  That is the signature
     of an ESTIMATION gain and nothing else, exactly as SCORECARD 10e predicts from the
     data-processing inequality - the student can only ever converge to P(Y | 200 genes),
     which hard labels also converge to.  At the scale this script uses, the gap the soft
     targets exist to close is already closed.  SCORECARD 10h reached the same conclusion
     from the challenge side: there is no estimation gap left anywhere in this problem.
  2. The incumbent functional form (linear / logistic, which is what the deployed atlas
     block is) gains NOTHING from distillation even at 40k - it is -0.26 pt.  The gain
     seen at small n exists only for the higher-capacity MLP, and most of it is the MLP
     climbing back out of the overfitting hole that hard labels put it in.
  3. The best full-scale variant (0.5 hard + 0.5 soft, 0.6132) beats the deployed logistic
     by +0.92 pt standalone - but so does the plain HARD-label MLP, by +0.70 pt.  The part
     attributable to reading the withheld 300 genes is +0.22 pt.
  4. The transfer coefficient from "better atlas block" to "better submission" has been
     measured once already: SCORECARD 11d improved the atlas block standalone by +0.74 pt
     (0.6040 -> 0.6114, mouse-centred harmonisation) and the 529-feature stack moved
     +0.12 pt on one fold partition and -0.10 pt on another, both non-significant.  The
     5,000-cell ET already conditions on Mouse_ID and absorbs most of what the block
     carries.  A short 2x1 smoke pass of this very script reproduced that: swapping in a
     student that BEATS the incumbent standalone still lost 0.30-0.80 pt on the stack.

Expected stack gain from the distillation term specifically: ~0.0 pt, well below the
~0.3 pt measurability floor.

LEGITIMACY - FLAGGED, NOT HIDDEN
--------------------------------
No challenge cell (train or test) is read with a withheld gene at any point.  The 300
withheld genes are read ONLY for non-challenge parent-atlas cells, and only to fit the
teacher.  The student, which is the only thing ever applied to a challenge cell, takes a
200-column input.

Is that a difference of DEGREE or of KIND from what we already do?

  DEGREE, in information terms.  The atlas hard labels are themselves a 500-gene product -
  they came from clustering on 500 genes - so the atlas block already routes 500-gene
  information into our model, in 5.9-bit quantised form.  Distillation widens that same
  channel to ~60 floats.  Nothing new about the LABEL enters: the student's asymptotic
  target is E[teacher | X_200] = P(Y | X_200), the same Bayes posterior hard labels
  converge to.  Distillation therefore cannot break the 200-gene ceiling; it can only
  reach it with fewer cells.  That is simultaneously the strongest legitimacy argument
  and the strongest argument that this will not help.

  KIND, in reviewability terms.  The hard labels are the atlas's PUBLISHED annotation - a
  public data product that a competitor is plainly expected to use.  The teacher posterior
  is a NEW measurement we manufacture by re-reading the panel the organisers deliberately
  withheld.  A reviewer reading this file sees the withheld 300 genes indexed inside a
  training loop.  Explaining the data-processing inequality to a judge, in order to defend
  a +0.1 pt effect, is a bad trade.

RECOMMENDATION: the 500-gene-teacher variant should NOT ship even if it passes.  If any
distilled block ships, it should be the 200-gene self-distilled variant, which is
unimpeachable - and that variant measured WORSE than hard labels at 40k.  Run this to
settle the question with a number, not to find a submission.

PRE-REGISTERED DECISION RULE
----------------------------
Candidate: STUDENT_D500 replaces the incumbent ATL block (dimensionality held fixed at 60,
every tree setting held fixed).  Adopt only if ALL of:

  (a) it beats the submitted stack on fold partition seed 7 at exact McNemar p < 0.05,
      computed on ONE out-of-fold prediction per cell (5,000 independent outcomes -
      SCORECARD 11e: flattening 5 repeats is anti-conservative);
  (b) the same comparison replicates on fold partition seed 23 at p < 0.05; AND
  (c) STUDENT_D500 beats STUDENT_D200 (the null control) at p < 0.05 on partition 7.

SECONDARY candidate: STUDENT_MIX500 (0.5 hard + 0.5 soft), which was the best variant
at full scale in the diagnostic.  Because it is a second shot at the same hypothesis it
must clear Holm-corrected p < 0.025 on BOTH partitions and beat its OWN matched null
(STUDENT_MIX200) at p < 0.05.  It is reported unconditionally; it is only adoptable
under those tighter thresholds.

(c) is the load-bearing clause.  Without it a pass would only show that soft targets
regularise an MLP, which the 200-gene teacher does too, and which buys us nothing we could
not have had legitimately.  STUDENT_HARD is the second control: it isolates what
distillation adds over ordinary label transfer at matched architecture and capacity.

RUNTIME.  A full run is 2 fold partitions x 5 configs x 25 folds x 10 ET seeds, about
45-60 minutes on this machine, plus ~5 minutes of torch.  Pass "quick" on the command
line for a 5x1 single-partition screen (~5 minutes) that answers clause (a) only.

The temperature T=4 and mixing weight lambda=1.0 are FIXED here.  They were chosen on
held-out ATLAS accuracy in the diagnostic above, which reads no challenge label, so this
script performs no tuning against the quantity it reports.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from torch_device import get_device

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")

QUICK = "quick" in sys.argv
ET_SEEDS = tuple(range(10))
STUDENT_SEEDS = (0, 1, 2)
N_SPLITS, N_REPEATS = (5, 1) if QUICK else (5, 5)
PARTITIONS = (7,) if QUICK else (7, 23)   # screen, then independent replication
ALPHA = 0.45
TEMPERATURE = 4.0             # fixed; chosen on held-out ATLAS accuracy, not on challenge
LAMBDA = 1.0
HIDDEN = 256
EPOCHS = 40
BATCH = 1024
TEACHER_FOLDS = 2             # teacher targets are OUT-OF-FOLD over the atlas

DEVICE = get_device()


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
GENES = list(counts_train.columns)
glia = meta_train["Region"].isna().to_numpy()
n_tr = len(meta_train)

cache = np.load(CACHE, allow_pickle=True)
CORE_TR = np.hstack([cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"],
                     cache["NIC_TR"]]).astype(np.float32)
ATL_TR = cache["ATL_TR"].astype(np.float32)          # incumbent 60-column atlas block
print(f"core={CORE_TR.shape} incumbent atlas block={ATL_TR.shape}", flush=True)

t0 = time.time()
with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    atlas_genes = [g.decode() for g in h["var/_index"][:]]
    X_atlas = sparse.csr_matrix(
        (h["X/data"][:].astype(np.float32), h["X/indices"][:], h["X/indptr"][:]),
        shape=(len(ids), len(atlas_genes)))
    cats = [c.decode() for c in h["obs/MERFISH cell type annotation/categories"][:]]
    codes = h["obs/MERFISH cell type annotation/codes"][:]

assert len(atlas_genes) == 500, f"atlas should carry 500 genes, has {len(atlas_genes)}"
gene_pos = {g: i for i, g in enumerate(atlas_genes)}
missing = [g for g in GENES if g not in gene_pos]
assert not missing, f"{len(missing)} released genes absent from the atlas"
RELEASED_COLS = np.array([gene_pos[g] for g in GENES])
WITHHELD = [g for g in atlas_genes if g not in set(GENES)]
print(f"atlas {X_atlas.shape}: {len(GENES)} released + {len(WITHHELD)} withheld", flush=True)

atlas_labels = np.array([F._normalise_label(cats[c]) if c >= 0 else "NA" for c in codes])
position = {c: i for i, c in enumerate(ids)}
is_challenge = np.zeros(len(ids), bool)
for index in (meta_train.index, meta_test.index):
    is_challenge[[position[str(c)] for c in index]] = True

usable = (~is_challenge) & np.isin(atlas_labels, CLASSES)
usable &= np.asarray(X_atlas.sum(1)).ravel() > 0
atlas_rows = np.flatnonzero(usable)
A500 = np.asarray(X_atlas[atlas_rows].todense(), np.float32)
a_y = np.array([CLASS_INDEX[l] for l in atlas_labels[atlas_rows]])
del X_atlas
print(f"teacher/student training pool: {len(atlas_rows)} non-challenge atlas cells "
      f"({time.time()-t0:.0f}s)", flush=True)
assert not is_challenge[atlas_rows].any(), "a challenge cell leaked into the atlas pool"

A200 = A500[:, RELEASED_COLS]
CH200 = np.vstack([counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32)])


def standardise(matrix, mean, std):
    return ((matrix - mean) / std).astype(np.float32)


Z500 = F.log_cpm(A500)
Z200 = F.log_cpm(A200)
ZCH = F.log_cpm(CH200)
del A500, A200
M500 = (Z500.mean(0), Z500.std(0) + 1e-6)
M200 = (Z200.mean(0), Z200.std(0) + 1e-6)
S500 = standardise(Z500, *M500)
S200 = standardise(Z200, *M200)
SCH = standardise(ZCH, *M200)      # challenge cells use the ATLAS 200-gene statistics
del Z500, Z200, ZCH


# ----------------------------------------------------------------------
# torch student / teacher
# ----------------------------------------------------------------------
def make_net(n_in, hidden):
    if hidden == 0:
        return nn.Linear(n_in, len(CLASSES))
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(hidden, len(CLASSES)))


def fit_net(features, rows, targets, soft=None, hidden=HIDDEN, temperature=1.0,
            lam=0.0, seed=0, epochs=EPOCHS):
    """Cross-entropy on hard `targets`, optionally mixed with KL to `soft`."""
    torch.manual_seed(seed)
    X = torch.tensor(features[rows], device=DEVICE)
    hard = torch.tensor(targets[rows], device=DEVICE)
    soft_t = None if soft is None else torch.tensor(soft, device=DEVICE)
    net = make_net(features.shape[1], hidden).to(DEVICE)
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    n = len(rows)
    for _ in range(epochs):
        order = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH):
            batch = order[start:start + BATCH]
            logits = net(X[batch])
            loss = (1.0 - lam) * cross_entropy(logits, hard[batch])
            if soft_t is not None and lam > 0:
                loss = loss + lam * temperature ** 2 * nn.functional.kl_div(
                    torch.log_softmax(logits / temperature, 1),
                    soft_t[batch], reduction="batchmean")
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    net.eval()
    return net


def predict(net, features, chunk=20000):
    out = np.empty((len(features), len(CLASSES)), np.float32)
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            block = torch.tensor(features[start:start + chunk], device=DEVICE)
            out[start:start + chunk] = torch.softmax(net(block), 1).cpu().numpy()
    return out


def soften(probs, temperature):
    logits = np.log(probs + 1e-9) / temperature
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(1, keepdims=True)).astype(np.float32)


def teacher_targets(features, tag):
    """Out-of-fold teacher posteriors, so no student ever sees a memorised target."""
    t = time.time()
    out = np.zeros((len(features), len(CLASSES)), np.float32)
    rng = np.random.default_rng(0)
    fold_of = rng.integers(0, TEACHER_FOLDS, len(features))
    accuracy = []
    for fold in range(TEACHER_FOLDS):
        train = np.flatnonzero(fold_of != fold)
        held = np.flatnonzero(fold_of == fold)
        net = fit_net(features, train, a_y, seed=fold)
        out[held] = predict(net, features[held])
        accuracy.append(float((out[held].argmax(1) == a_y[held]).mean()))
    print(f"[teacher {tag}] out-of-fold atlas accuracy "
          f"{np.mean(accuracy):.4f} ({time.time()-t:.0f}s)", flush=True)
    return soften(out, TEMPERATURE), float(np.mean(accuracy))


def student_block(soft, tag, hidden=HIDDEN, lam=LAMBDA):
    """Average STUDENT_SEEDS students, return their 60 probabilities on challenge cells."""
    t = time.time()
    stacked = np.zeros((len(SCH), len(CLASSES)), np.float32)
    for seed in STUDENT_SEEDS:
        net = fit_net(S200, np.arange(len(S200)), a_y, soft=soft, hidden=hidden,
                      temperature=TEMPERATURE, lam=0.0 if soft is None else lam,
                      seed=seed)
        stacked += predict(net, SCH)
    stacked /= len(STUDENT_SEEDS)
    standalone = float((CLASS_ARR[stacked[:n_tr].argmax(1)] == y).mean())
    print(f"[student {tag}] standalone on 5,000 labelled challenge cells "
          f"{standalone:.4f} ({time.time()-t:.0f}s)", flush=True)
    return stacked, standalone


# ----------------------------------------------------------------------
# build the blocks
# ----------------------------------------------------------------------
print("\n=== teachers ===", flush=True)
SOFT500, teacher500_acc = teacher_targets(S500, "500 genes  [reads the withheld 300]")
SOFT200, teacher200_acc = teacher_targets(S200, "200 genes  [NULL, released only]")

print("\n=== students (200 released genes only) ===", flush=True)
BLOCKS = {}
BLOCKS["student_hard"], acc_hard = student_block(None, "HARD labels  [control]")
BLOCKS["student_d500"], acc_d500 = student_block(SOFT500, "distil 500g  [candidate]")
BLOCKS["student_d200"], acc_d200 = student_block(SOFT200, "distil 200g  [null control]")
BLOCKS["student_mix500"], acc_mix5 = student_block(
    SOFT500, "0.5 hard + 0.5 distil 500g  [secondary candidate]", lam=0.5)
BLOCKS["student_mix200"], acc_mix2 = student_block(
    SOFT200, "0.5 hard + 0.5 distil 200g  [its null control]", lam=0.5)
BLOCKS["student_linear_d500"], acc_lin = student_block(
    SOFT500, "distil 500g, LINEAR  [incumbent form]", hidden=0)

incumbent = float((CLASS_ARR[ATL_TR.argmax(1)] == y).mean())
print(f"\n[incumbent] deployed logistic atlas block standalone {incumbent:.4f}", flush=True)

standalone = pd.DataFrame([
    {"block": "incumbent logistic (deployed)", "standalone": incumbent},
    {"block": "student_hard (control)", "standalone": acc_hard},
    {"block": "student_d500 (candidate)", "standalone": acc_d500},
    {"block": "student_d200 (null control)", "standalone": acc_d200},
    {"block": "student_mix500 (secondary candidate)", "standalone": acc_mix5},
    {"block": "student_mix200 (its null control)", "standalone": acc_mix2},
    {"block": "student_linear_d500", "standalone": acc_lin},
    {"block": "teacher 500g (out-of-fold ATLAS acc)", "standalone": teacher500_acc},
    {"block": "teacher 200g (out-of-fold ATLAS acc)", "standalone": teacher200_acc},
])
print(standalone.to_string(index=False), flush=True)
standalone.to_csv(OUT / "distill_standalone.csv", index=False)


# ----------------------------------------------------------------------
# stack evaluation
# ----------------------------------------------------------------------
CONFIGS = {
    "submitted (incumbent atlas block)": ATL_TR,
    "replace with student_hard": BLOCKS["student_hard"][:n_tr],
    "replace with student_d500": BLOCKS["student_d500"][:n_tr],
    "replace with student_d200": BLOCKS["student_d200"][:n_tr],
    "replace with student_mix500": BLOCKS["student_mix500"][:n_tr],
    "replace with student_mix200": BLOCKS["student_mix200"][:n_tr],
    "append student_d500 to incumbent": np.hstack([ATL_TR, BLOCKS["student_d500"][:n_tr]]),
}
BASELINE = "submitted (incumbent atlas block)"


def run_partition(partition_seed):
    """One repeated-CV pass. Returns per-repeat correctness, (n_repeats, 5000)."""
    folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                         random_state=partition_seed).split(y, y))
    results = {}
    for name, block in CONFIGS.items():
        t = time.time()
        X = np.hstack([CORE_TR, block.astype(np.float32)]).astype(np.float32)
        ok = np.zeros((N_REPEATS, n_tr), bool)
        for f, (tr, va) in enumerate(folds):
            probs = M.fit_extra_trees(X[tr], pd.Series(y[tr]), CLASSES, X[va],
                                      seeds=ET_SEEDS)
            probs = M.correct_prior(probs, M.prior_vector(pd.Series(y[tr]), CLASSES),
                                    ALPHA)
            ok[f // N_SPLITS, va] = CLASS_ARR[probs.argmax(1)] == y[va]
        results[name] = ok
        mean = ok.mean(1)
        print(f"  {name:36s} acc={mean.mean():.5f} +/-{mean.std():.5f} "
              f"glia={ok[:, glia].mean():.4f} features={X.shape[1]} "
              f"({time.time()-t:.0f}s)", flush=True)
    return results


rows = []
for partition_seed in PARTITIONS:
    print(f"\n=== {N_SPLITS}x{N_REPEATS} CV, fold partition seed {partition_seed}, "
          f"{len(ET_SEEDS)} ET seeds ===", flush=True)
    results = run_partition(partition_seed)
    base = results[BASELINE]
    # SCORECARD 11e: exact McNemar uses repeat 0 only - one OOF outcome per cell, so the
    # 5,000 paired trials really are 5,000 independent biological cells.
    for name, ok in results.items():
        if name == BASELINE:
            continue
        p_stack, table = M.paired_mcnemar(ok[0], base[0])
        rows.append({"partition": partition_seed, "variant": name,
                     "accuracy": float(ok.mean()), "baseline": float(base.mean()),
                     "gain": float(ok.mean() - base.mean()),
                     "glia": float(ok[:, glia].mean()),
                     "mcnemar_p_repeat0": p_stack,
                     "discordant": f"{table[0][1]}/{table[1][0]}"})
        print(f"  {name:36s} gain={ok.mean()-base.mean():+.5f} "
              f"p={p_stack:.4g} (discordant {table[0][1]}/{table[1][0]})", flush=True)
    if partition_seed == PARTITIONS[0]:
        p_null, table_null = M.paired_mcnemar(results["replace with student_d500"][0],
                                              results["replace with student_d200"][0])
        gain_null = float(results["replace with student_d500"].mean()
                          - results["replace with student_d200"].mean())
        p_mix, table_mix = M.paired_mcnemar(results["replace with student_mix500"][0],
                                            results["replace with student_mix200"][0])
        gain_mix = float(results["replace with student_mix500"].mean()
                         - results["replace with student_mix200"].mean())
        print(f"\n  NULL-CONTROL CONTRAST  d500   vs d200  : {gain_null:+.5f} "
              f"p={p_null:.4g} (discordant {table_null[0][1]}/{table_null[1][0]})",
              flush=True)
        print(f"  NULL-CONTROL CONTRAST  mix500 vs mix200: {gain_mix:+.5f} "
              f"p={p_mix:.4g} (discordant {table_mix[0][1]}/{table_mix[1][0]})",
              flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "distill_stack.csv", index=False)


# ----------------------------------------------------------------------
# pre-registered verdict
# ----------------------------------------------------------------------
def clause(partition_seed):
    row = frame[(frame.partition == partition_seed)
                & (frame.variant == "replace with student_d500")]
    if row.empty:
        return False, 0.0, 1.0
    gain, p_value = float(row.gain.iloc[0]), float(row.mcnemar_p_repeat0.iloc[0])
    return bool(gain > 0 and p_value < 0.05), gain, p_value


ok_a, gain_a, p_a = clause(PARTITIONS[0])
ok_b, gain_b, p_b = clause(PARTITIONS[1]) if len(PARTITIONS) > 1 else (False, 0.0, 1.0)
ok_c = bool(gain_null > 0 and p_null < 0.05)

print("\n=== PRE-REGISTERED DECISION RULE ===", flush=True)
print(f"  (a) screen  partition {PARTITIONS[0]}: gain={gain_a:+.5f} p={p_a:.4g} "
      f"-> {'PASS' if ok_a else 'FAIL'}", flush=True)
label_b = f"partition {PARTITIONS[1]}" if len(PARTITIONS) > 1 else "NOT RUN (quick)"
print(f"  (b) replicate {label_b}: gain={gain_b:+.5f} p={p_b:.4g} "
      f"-> {'PASS' if ok_b else 'FAIL'}", flush=True)
print(f"  (c) beats 200g-teacher null: gain={gain_null:+.5f} p={p_null:.4g} "
      f"-> {'PASS' if ok_c else 'FAIL'}", flush=True)
adopt = ok_a and ok_b and ok_c
print(f"\n  VERDICT: {'ALL CLAUSES PASS' if adopt else 'DO NOT ADOPT'}", flush=True)
if adopt:
    print("  NOTE: even on a pass, shipping this means telling the organisers we fitted a\n"
          "  teacher on the 300 withheld genes of atlas cells. See the LEGITIMACY section.\n"
          "  The 200-gene self-distilled block is the shippable form; check whether\n"
          "  'replace with student_d200' also passes before touching the submission.",
          flush=True)

np.savez_compressed(OUT / "distill_blocks.npz",
                    **{k: v for k, v in BLOCKS.items()})
print(f"\nwrote {OUT/'distill_standalone.csv'}, {OUT/'distill_stack.csv'}, "
      f"{OUT/'distill_blocks.npz'}", flush=True)
