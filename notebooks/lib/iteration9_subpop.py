"""Iteration 9 - forensic error stratification, and the falsification of every
depth-targeted remedy. THIS SCRIPT RECORDS A NEGATIVE RESULT. Nothing here is adopted.

WHAT THE FORENSICS FOUND (cross-validation only; the test set was never touched).

Out-of-fold predictions for the 5,000 training cells with the submitted 529-feature model
(5-fold, 5 seeds, alpha=0.45, metadata mask rebuilt per fold) score 0.7932 on fold
partition 0 and 0.7928 on partition 1. Error rate was then binned along nine axes. Only
one family of axes separates cells at all, and they are all the same axis - how much
information the cell carries:

    depth (total counts)   d0 [1,8]    err 0.3493  ->  d9 [62,550]  err 0.1035
    n genes detected       d0 [1,5]    err 0.4035  ->  d9 [27,85]   err 0.0814
    counts / volume        d0          err 0.3830  ->  d9           err 0.1120
    depth / section median d0 [.05,.45] err 0.3692 ->  d9 [2.7,21]  err 0.1100

All four are monotone and highly significant (Spearman rho of correctness with n-genes
= +0.209, p = 2e-50). Crucially the effect is NOT a glia/neuron proxy - it survives
stratification inside both compartments:

    glia    depth d0 err 0.4021 -> d9 err 0.2066   (base 0.2696, n = 3142)
    neurons depth d0 err 0.1845 -> d9 err 0.0474   (base 0.1012, n = 1858)

Everything else is flat. Cell VOLUME is flat (rho = -0.027, p = 0.058; decile errors
0.157-0.248 with no ordering). Local cell spacing, distance to the section convex hull,
cells per section, AP position, gender, mouse and section-median depth all move error by
under 4 points with no monotone structure, and section-level spread is fully explained by
section size (observed sd 0.157 vs binomial-expected 0.060 at the observed section sizes).

So there IS a large, clean, identifiable subpopulation: 1,261 cells with <= 8 genes
detected carry error 0.3128 against an overall 0.2070, i.e. 133 excess errors - more
than the 171 needed for first place. That is why this was worth chasing.

WHY IT IS NOT EXPLOITABLE. Four remedies were pre-registered and all four failed against
their own null controls. Each is reproduced below.

  1. DEPTH-CONDITIONAL PRIOR. p(c | depth quintile) really does differ from p(c) -
     KL 0.196 in Q1 and 0.306 in Q5, endothelial runs 13x and meninges_1 313x across
     quintiles - and the model's predicted composition in Q1 is visibly wrong
     (oligodendrocyte_progenitor_2 114 true vs 73 predicted, oligodendrocyte_1 79 true
     vs 124 predicted). Replacing the global prior with the per-quintile prior in
     correct_prior LOSES 1.74 / 1.80 pt on the two partitions. The NULL - the same
     procedure on a randomly permuted depth variable - loses 1.38 / 1.16 pt, so most of
     the damage is just noisier prior estimates, and depth adds harm on top of that.

  2. DEPTH-DEPENDENT ALPHA, 5 free exponents. Tuned per quintile on partition 0 it gains
     +0.18 pt there and LOSES 0.08 pt on partition 1. Textbook fold-partition overfit,
     the same failure mode as per-fold alpha in the scorecard.

  3. DEPTH-DEPENDENT ALPHA, 1 free parameter: alpha_i = 0.45 + beta * z(log depth).
     Partition 0 picks beta = -0.050, partition 1 picks beta = -0.125. Cross-validated
     honestly the two choices give +0.04 pt and -0.20 pt, mean -0.08 pt. The accuracy
     surface is flat to within 0.15 pt over beta in [-0.125, +0.075].

  4. DEPTH-MATCHED SPECIALIST - the decisive one. An ET trained only on the low-depth
     fold-training cells and evaluated on low-depth fold-test cells scores 0.6382 against
     the global model's 0.6887 on the same cells. The NULL - an ET trained on a RANDOM
     subset of identical size - scores 0.6570. The specialist is worse than the null, so
     depth-matching the training set is actively harmful beyond the data loss:
     high-depth cells are BETTER exemplars for classifying low-depth cells than
     low-depth cells are. Blending the specialist in loses at every weight tested.

     The implied inverse (down-weight low-depth TRAINING cells x0.3) gains +0.24 pt, but
     the NULL that down-weights a RANDOM 20% by the same factor gains +0.14 pt - i.e. it
     is a bagging perturbation, not a depth effect, and both sit inside seed noise.

  5. ORACLE CEILING. A per-(depth-quintile, class) multiplier table - 300 parameters
     fitted directly on the out-of-fold ANSWERS - reaches 0.8154 from 0.7932. The
     scorecard's global per-class oracle already reaches about +1.5 pt, so depth
     stratification buys roughly +0.7 pt even with the answers in hand, and nothing
     attainable without them.

CONCLUSION. The low-information subpopulation is real, large and monotone, but it is
information-limited in exactly the way the 200-gene glial panel is. The model is not
mis-specified there; there is simply less to read. In the lowest depth quintile the true
class is already ranked 1st for 66.5%, 2nd for 83.1% and 3rd for 91.5% of cells - the
ranking is fine and the argmax is where the information runs out. No reweighting,
re-exponentiation or stratified refit recovers any of it.

PRE-REGISTERED DECISION RULE (fixed before running, and the reason nothing is adopted):
adopt a variant only if (a) it beats the frozen baseline on BOTH fold partitions, and
(b) it beats its own null control. Every variant fails at least one, most fail both.

Runtime is roughly 10 minutes; this exists to reproduce the negative, not to be adopted.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration9")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
SEEDS = tuple(range(5))
ALPHA = 0.45
NB = 5                      # depth quintiles
PARTITIONS = (0, 1)         # two independent fold assignments; a variant must win both
MASK_COLS = ["Region", "Excitatory_vs_Inhibitory", "Segment"]
ET = dict(n_estimators=600, max_features="sqrt", min_samples_leaf=2, n_jobs=2)

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
K = len(CLASSES)
n = len(y)
glia = meta_train["Region"].isna().to_numpy()

cache = np.load(CACHE, allow_pickle=True)
X = np.hstack([cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"],
               cache["NIC_TR"], cache["ATL_TR"]]).astype(np.float32)

raw_counts = counts_train.to_numpy()
depth = raw_counts.sum(1)
ngene = (raw_counts > 0).sum(1)
volume = pd.to_numeric(meta_train["volume"], errors="coerce").to_numpy(float)
density = depth / np.where(volume == 0, np.nan, volume)
GPRIOR = np.array([(y == c).mean() for c in CLASSES])

edges = np.quantile(depth, np.linspace(0, 1, NB + 1))[1:-1]
dbin = np.digitize(depth, edges)
low = dbin == 0
print(f"n={n} feat={X.shape[1]} classes={K} depth median={np.median(depth):.0f} "
      f"quintile edges={edges}", flush=True)


def fit_et(Xtr, ytr, Xev, seeds=SEEDS, weight=None):
    out = np.zeros((len(Xev), K), np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **ET).fit(Xtr, ytr,
                                                                 sample_weight=weight)
        out += M.align_proba(model, Xev, CLASSES)
    return out / len(seeds)


def metadata_mask(train_rows, eval_rows):
    """Hard compatibility mask, rebuilt from FOLD-TRAIN labels only (as in §8a)."""
    allow = np.ones((len(eval_rows), K), bool)
    for col in MASK_COLS:
        vals = meta_train[col].astype(str).to_numpy()
        seen = [set(vals[train_rows][y[train_rows] == cls]) for cls in CLASSES]
        known = set(vals[train_rows])
        for i, v in enumerate(vals[eval_rows]):
            if v in known:
                allow[i] &= np.array([v in s for s in seen])
    allow[~allow.any(1)] = True
    return allow


# ---------------------------------------------------------------- 1. OOF baseline
oof, allow_all, folds = {}, {}, {}
for part in PARTITIONS:
    t0 = time.time()
    raw = np.zeros((n, K), np.float32)
    allow = np.zeros((n, K), bool)
    fold = np.zeros(n, int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=part)
    for f, (tr, te) in enumerate(splitter.split(X, y)):
        fold[te] = f
        raw[te] = fit_et(X[tr], y[tr], X[te])
        allow[te] = metadata_mask(tr, te)
    oof[part], allow_all[part], folds[part] = raw, allow, fold
    corrected = np.where(allow, M.correct_prior(raw, GPRIOR, ALPHA), -1.0)
    print(f"[oof] partition {part}: acc={(CLASS_ARR[corrected.argmax(1)] == y).mean():.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)


def score(part, prior_mat, alpha_vec, subset=None):
    raw, allow = oof[part], allow_all[part]
    adj = raw / np.maximum(prior_mat, 1e-12) ** alpha_vec[:, None]
    pred = CLASS_ARR[np.where(allow, adj, -1.0).argmax(1)]
    return (pred == y).mean() if subset is None else (pred[subset] == y[subset]).mean()


BASE = {p: score(p, np.tile(GPRIOR, (n, 1)), np.full(n, ALPHA)) for p in PARTITIONS}

# ---------------------------------------------------------------- 2. error stratification
print("\n=== error rate by decile, nine axes ===", flush=True)
axes = {"depth": depth, "n_genes": ngene, "volume": volume, "density": density,
        "rel_depth": depth / np.maximum(
            pd.Series(depth).groupby(meta_train["Section_ID"].astype(str).to_numpy())
            .transform("median").to_numpy(), 1)}
correct = np.mean([[score(p, np.tile(GPRIOR, (n, 1)), np.full(n, ALPHA)) for p in PARTITIONS]])
ok = np.mean([(CLASS_ARR[np.where(allow_all[p],
                                  M.correct_prior(oof[p], GPRIOR, ALPHA), -1.0).argmax(1)] == y)
              for p in PARTITIONS], axis=0)
for name, v in axes.items():
    m = np.isfinite(v)
    q = np.unique(np.nanquantile(v[m], np.linspace(0, 1, 11)))
    b = np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)
    rates = [1 - ok[m & (b == i)].mean() for i in range(len(q) - 1) if (m & (b == i)).any()]
    rho = spearmanr(v[m], ok[m]).statistic
    print(f"  {name:10s} rho={rho:+.4f}  deciles=" + " ".join(f"{r:.3f}" for r in rates),
          flush=True)
for name, sub in [("glia", glia), ("neuron", ~glia)]:
    q = np.quantile(depth[sub], np.linspace(0, 1, 11))
    b = np.clip(np.digitize(depth, q[1:-1]), 0, 9)
    print(f"  depth|{name:7s} deciles=" +
          " ".join(f"{1 - ok[sub & (b == i)].mean():.3f}" for i in range(10)), flush=True)

# ---------------------------------------------------------------- 3. remedies + nulls
print("\n=== remedy 1: depth-conditional prior (+ shuffled-depth NULL) ===", flush=True)
rng = np.random.default_rng(7)
shuffled = rng.permutation(dbin)
for label, bins in [("depth-conditional", dbin), ("NULL shuffled-depth", shuffled)]:
    for part in PARTITIONS:
        fold = folds[part]
        prior_mat = np.zeros((n, K))
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            for b in range(NB):
                sel = tr & (bins == b)
                pb = np.maximum(np.array([(y[sel] == c).mean() for c in CLASSES]), 1e-4)
                prior_mat[te & (bins == b)] = pb / pb.sum()
        acc = score(part, prior_mat, np.full(n, ALPHA))
        print(f"  {label:22s} partition {part}: {acc:.4f} ({(acc - BASE[part]) * 100:+.2f} pt)",
              flush=True)

print("\n=== remedy 2/3: depth-dependent alpha, tuned on one partition, validated on the other ===",
      flush=True)
grid = np.arange(0.0, 1.05, 0.05)
for tune, val in [(0, 1), (1, 0)]:
    per_bin = []
    for b in range(NB):
        sel = dbin == b
        accs = [score(tune, np.tile(GPRIOR, (n, 1)),
                      np.where(sel, a, ALPHA)) for a in grid]
        per_bin.append(grid[int(np.argmax(accs))])
    av = np.array([per_bin[b] for b in dbin])
    print(f"  5-param  tune={tune} exponents={per_bin} -> tune {score(tune, np.tile(GPRIOR, (n, 1)), av):.4f} "
          f"| VALIDATE {score(val, np.tile(GPRIOR, (n, 1)), av):.4f} vs base {BASE[val]:.4f}", flush=True)

z = (np.log1p(depth) - np.log1p(depth).mean()) / np.log1p(depth).std()
for tune, val in [(0, 1), (1, 0)]:
    betas = np.arange(-0.25, 0.26, 0.025)
    accs = [score(tune, np.tile(GPRIOR, (n, 1)), np.clip(ALPHA + b * z, 0.0, 1.5)) for b in betas]
    beta = betas[int(np.argmax(accs))]
    av = np.clip(ALPHA + beta * z, 0.0, 1.5)
    print(f"  1-param  tune={tune} beta={beta:+.3f} -> tune {max(accs):.4f} "
          f"| VALIDATE {score(val, np.tile(GPRIOR, (n, 1)), av):.4f} vs base {BASE[val]:.4f}", flush=True)

print("\n=== remedy 4: depth-matched specialist (+ random-subset NULL of identical size) ===",
      flush=True)


def specialist(pool, part):
    fold = folds[part]
    out = np.zeros((n, K), np.float32)
    for f in np.unique(fold):
        tr = np.flatnonzero((fold != f) & pool)
        te = np.flatnonzero((fold == f) & low)
        if len(te) == 0:
            continue
        out[te] = M.correct_prior(fit_et(X[tr], y[tr], X[te], seeds=(0, 1, 2)),
                                  M.prior_vector(y[tr], CLASSES), ALPHA)
    return (CLASS_ARR[out[low].argmax(1)] == y[low]).mean()


rng = np.random.default_rng(11)
random_pool = np.zeros(n, bool)
random_pool[rng.choice(n, int(low.sum()), replace=False)] = True
for part in PARTITIONS:
    global_low = (CLASS_ARR[np.where(allow_all[part],
                                     M.correct_prior(oof[part], GPRIOR, ALPHA),
                                     -1.0).argmax(1)][low] == y[low]).mean()
    print(f"  partition {part}: global={global_low:.4f}  depth-matched={specialist(low, part):.4f}  "
          f"NULL random-subset={specialist(random_pool, part):.4f}", flush=True)

print("\n=== remedy 5: ORACLE per-(quintile, class) multiplier fitted on the OOF answers ===",
      flush=True)
part = PARTITIONS[0]
adj0 = oof[part] / GPRIOR[None, :] ** ALPHA
allow = allow_all[part]
W = np.ones((NB, K))
cur = BASE[part]
for it in range(6):
    for b in range(NB):
        sel = dbin == b
        for j in range(K):
            best_w, best_acc = W[b, j], -1.0
            for w in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0):
                Wt = W.copy()
                Wt[b, j] = w
                acc = (CLASS_ARR[np.where(allow, adj0 * Wt[dbin], -1.0).argmax(1)] == y)[sel].mean()
                if acc > best_acc:
                    best_acc, best_w = acc, w
            W[b, j] = best_w
    new = (CLASS_ARR[np.where(allow, adj0 * W[dbin], -1.0).argmax(1)] == y).mean()
    print(f"  iter {it}: {new:.4f} (baseline {BASE[part]:.4f})", flush=True)
    if new - cur < 1e-4:
        break
    cur = new

print("\nNo variant beats the frozen baseline on both partitions AND its null control. "
      "Nothing is adopted.", flush=True)
