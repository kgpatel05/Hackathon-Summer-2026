"""Iteration 12 - count-native NMF gene programs learned from the external atlas.

Sparse single-gene splits ignore coordinated lineage programs.  This experiment learns
32 additive programs with KL-divergence MiniBatchNMF on 40,000 non-challenge parent-atlas
cells restricted to the released 200 genes.  Challenge features are their non-negative
program activities plus reconstruction error.

The null independently permutes every gene across atlas cells before fitting NMF.  It
preserves gene marginals, per-gene sparsity, feature width and algorithmic capacity while
destroying gene co-expression.  Screen: partition 853, five ET seeds; advance only for
>0.30 point gain, p<0.05, and >0.20 point advantage over the null.  No test label or
withheld gene is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import MiniBatchNMF
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "nmf_programs.npz"
PARTITION = 853
SEEDS = tuple(range(5))
ALPHA = 0.45
N_COMPONENTS = 32
N_ATLAS = 40000


def frequencies(x: np.ndarray) -> np.ndarray:
    total = x.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return (100.0 * x / total).astype(np.float32)


def build(counts_train: pd.DataFrame, counts_test: pd.DataFrame,
          meta_train: pd.DataFrame, meta_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    genes = list(counts_train.columns)
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        ids = np.asarray([x.decode() for x in handle["obs/_index"][:]])
        atlas_genes = [x.decode() for x in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )[:, [lookup[g] for g in genes]]
    challenge = set(meta_train.index.astype(str)) | set(meta_test.index.astype(str))
    donor = np.flatnonzero(np.asarray([cell not in challenge for cell in ids]))
    rng = np.random.default_rng(20260819)
    donor = rng.choice(donor, min(N_ATLAS, len(donor)), replace=False)
    atlas = frequencies(np.asarray(matrix[donor].todense(), np.float32))
    challenge_x = frequencies(np.vstack([counts_train.to_numpy(np.float32),
                                         counts_test.to_numpy(np.float32)]))

    def model_features(fit_x: np.ndarray, seed: int) -> np.ndarray:
        model = MiniBatchNMF(
            n_components=N_COMPONENTS, init="nndsvda", batch_size=1024,
            beta_loss="kullback-leibler", max_iter=150, max_no_improvement=15,
            random_state=seed,
        ).fit(fit_x)
        activity = model.transform(challenge_x).astype(np.float32)
        reconstruction = activity @ model.components_
        error = np.mean(
            challenge_x * np.log((challenge_x + 1e-6) / (reconstruction + 1e-6))
            - challenge_x + reconstruction, axis=1, keepdims=True
        )
        return np.hstack([np.log1p(activity), np.log1p(np.maximum(error, 0))]).astype(np.float32)

    t0 = time.time()
    real = model_features(atlas, 853)
    print(f"real NMF fitted in {time.time()-t0:.1f}s", flush=True)
    null_atlas = atlas.copy()
    for gene in range(null_atlas.shape[1]):
        null_atlas[:, gene] = null_atlas[rng.permutation(len(null_atlas)), gene]
    t0 = time.time()
    null = model_features(null_atlas, 854)
    print(f"null NMF fitted in {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(CACHE, real=real, null=null)
    return real, null


def oof(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    correct = np.zeros(len(y), bool); labels = np.asarray(classes)
    for train, valid in StratifiedKFold(5, shuffle=True, random_state=PARTITION).split(y, y):
        p = M.fit_extra_trees(x[train], pd.Series(y[train]), classes, x[valid], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        correct[valid] = labels[p.argmax(1)] == y[valid]
    return correct


def main() -> None:
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy(); classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    if CACHE.exists():
        cached = np.load(CACHE); real, null = cached["real"], cached["null"]
    else:
        real, null = build(counts_train, counts_test, meta_train, meta_test)
    baseline = current_stack(meta_all, classes, list(counts_train.columns))
    configs = {
        "baseline_694": baseline,
        "+ NMF programs": np.hstack([baseline, real[:len(y)]]).astype(np.float32),
        "+ shuffled-gene NMF null": np.hstack([baseline, null[:len(y)]]).astype(np.float32),
    }
    print(f"program block={real.shape[1]} partition={PARTITION}", flush=True)
    results = {}; t0 = time.time()
    for name, x in configs.items():
        results[name] = oof(x, y, classes)
        print(f"finished {name} ({time.time()-t0:.1f}s)", flush=True)
    base = results["baseline_694"]; rows = []
    for name, ok in results.items():
        if name == "baseline_694": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, base)
            wins = int((ok & ~base).sum()); losses = int((base & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100 * (ok.mean() - base.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "nmf_programs_screen.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real_row, null_row = rows[1], rows[2]
    passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
              real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
