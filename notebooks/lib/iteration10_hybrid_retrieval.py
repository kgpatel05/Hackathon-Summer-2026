"""Iteration 10 - anatomy-constrained expression retrieval from the parent atlas.

Existing atlas blocks use either expression globally or position locally.  This block
models their interaction: for each challenge cell, retrieve the 256 nearest external
atlas cells in the same section, select the 16 best expression matches among them using
only the released 200 genes, and emit a similarity/spatially weighted label distribution.

All 10,000 challenge cells are removed from the donor atlas first.  A matched null keeps
the exact donors and weights but permutes donor labels within section.  The external
labels are the same public annotation already used by atlas_transfer; no challenge label,
test label, or withheld gene is read.

Pre-registered screen: partition seed 211, five estimator seeds.  Advance only for gain
>0.30 points over the 694-feature incumbent, paired exact McNemar p<0.05, and >0.20
points beyond the null.  Confirm on untouched partition seed 229 with 20 estimator seeds;
adopt only for gain >0.20 points and p<0.05.

Usage:
    python3 notebooks/lib/iteration10_hybrid_retrieval.py screen
    python3 notebooks/lib/iteration10_hybrid_retrieval.py confirm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "hybrid_retrieval.npz"
ALPHA = 0.45
K_SPATIAL = 256
K_EXPRESSION = 16
TEMPERATURE = 0.05
SCREEN_PARTITION = 211
CONFIRM_PARTITION = 229
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def unit_log_cpm(counts: np.ndarray) -> np.ndarray:
    x = F.log_cpm(counts).astype(np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def build_retrieval(counts_all: np.ndarray, meta_all: pd.DataFrame,
                    genes: list[str], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        atlas_genes = [g.decode() for g in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        columns = np.asarray([lookup[g] for g in genes])
        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )
        donor_counts = np.asarray(matrix[donors][:, columns].todense(), np.float32)

    donor_x = unit_log_cpm(donor_counts)
    query_x = unit_log_cpm(counts_all)
    donor_sections = sections[donors]
    query_sections = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    class_index = {name: i for i, name in enumerate(classes)}
    other = len(classes)
    donor_code = np.asarray([class_index.get(name, other) for name in labels[donors]])

    rng = np.random.default_rng(20260819)
    null_code = donor_code.copy()
    for section in np.unique(donor_sections):
        rows = np.flatnonzero(donor_sections == section)
        null_code[rows] = donor_code[rng.permutation(rows)]

    real = np.zeros((len(meta_all), other + 1), np.float32)
    null = np.zeros_like(real)
    for section in np.unique(query_sections):
        qrows = np.flatnonzero(query_sections == section)
        drows = np.flatnonzero(donor_sections == section)
        spatial_k = min(K_SPATIAL, len(drows))
        tree = cKDTree(np.column_stack([ax[donors[drows]], ay[donors[drows]]]))
        distance, local_nn = tree.query(query_xy[qrows], k=spatial_k)
        if distance.ndim == 1:
            distance, local_nn = distance[:, None], local_nn[:, None]
        candidate_rows = drows[local_nn]

        for start in range(0, len(qrows), 64):
            stop = min(start + 64, len(qrows))
            candidates = candidate_rows[start:stop]
            similarity = np.einsum(
                "bkd,bd->bk", donor_x[candidates], query_x[qrows[start:stop]],
                optimize=True,
            )
            expression_k = min(K_EXPRESSION, similarity.shape[1])
            chosen = np.argpartition(similarity, -expression_k, axis=1)[:, -expression_k:]
            chosen_sim = np.take_along_axis(similarity, chosen, axis=1)
            chosen_dist = np.take_along_axis(distance[start:stop], chosen, axis=1)
            selected = np.take_along_axis(candidates, chosen, axis=1)

            radius = np.maximum(distance[start:stop, -1:], 1.0)
            weight = np.exp((chosen_sim - chosen_sim.max(axis=1, keepdims=True)) /
                            TEMPERATURE)
            weight *= np.exp(-0.5 * (chosen_dist / radius) ** 2)
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-8)

            for b, row in enumerate(qrows[start:stop]):
                real[row] = np.bincount(
                    donor_code[selected[b]], weights=weight[b], minlength=other + 1
                )
                null[row] = np.bincount(
                    null_code[selected[b]], weights=weight[b], minlength=other + 1
                )
    return real, null


def oof_probabilities(x: np.ndarray, y: np.ndarray, classes: list[str], partition: int,
                      seeds: tuple[int, ...]) -> np.ndarray:
    probabilities = np.zeros((len(y), len(classes)), np.float32)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for train, valid in folds.split(y, y):
        probs = M.fit_extra_trees(
            x[train], pd.Series(y[train]), classes, x[valid], seeds=seeds
        )
        probabilities[valid] = M.correct_prior(
            probs, M.prior_vector(pd.Series(y[train]), classes), ALPHA
        )
    return probabilities


def main(mode: str) -> None:
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    class_array = np.asarray(classes)
    genes = list(counts_train.columns)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    if CACHE.exists():
        cached = np.load(CACHE)
        real, null = cached["real"], cached["null"]
        print(f"loaded {CACHE}: {real.shape}", flush=True)
    else:
        t0 = time.time()
        counts_all = np.vstack([counts_train.to_numpy(), counts_test.to_numpy()])
        real, null = build_retrieval(counts_all, meta_all, genes, classes)
        np.savez_compressed(CACHE, real=real, null=null, classes=np.asarray(classes))
        print(f"built {CACHE}: {real.shape} in {time.time()-t0:.1f}s", flush=True)

    print(f"standalone train accuracy: real="
          f"{(class_array[real[:len(y), :len(classes)].argmax(1)] == y).mean():.4f} "
          f"null={(class_array[null[:len(y), :len(classes)].argmax(1)] == y).mean():.4f}",
          flush=True)
    baseline = current_stack(meta_all, classes, genes)
    configs = {
        "baseline_694": baseline,
        "+ hybrid retrieval": np.hstack([baseline, real[:len(y)]]).astype(np.float32),
        "+ shuffled retrieval (null)": np.hstack([baseline, null[:len(y)]]).astype(np.float32),
    }
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    print(f"mode={mode} partition={partition} estimator_seeds={len(seeds)}", flush=True)

    ok = {}
    for name, x in configs.items():
        t0 = time.time()
        probs = oof_probabilities(x, y, classes, partition, seeds)
        ok[name] = class_array[probs.argmax(1)] == y
        print(f"finished {name} ({x.shape[1]} features) in {time.time()-t0:.1f}s", flush=True)

    base_ok = ok["baseline_694"]
    rows = []
    for name, correct in ok.items():
        gain = correct.mean() - base_ok.mean()
        if name == "baseline_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(correct, base_ok)
            wins = int((correct & ~base_ok).sum())
            losses = int((base_ok & ~correct).sum())
        rows.append({"mode": mode, "partition": partition, "config": name,
                     "accuracy": correct.mean(), "gain_pt": 100 * gain,
                     "wins": wins, "losses": losses, "p": p_value})
        print(f"{name:30s} acc={correct.mean():.4f} gain={100*gain:+.2f}pt "
              f"{wins}w/{losses}l p={p_value:.5g}", flush=True)

    real_row, null_row = rows[1], rows[2]
    if mode == "screen":
        passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
                  real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
        verdict = "ADVANCE TO CONFIRM" if passed else "REJECT"
    else:
        passed = real_row["gain_pt"] > 0.20 and real_row["p"] < 0.05
        verdict = "ADOPT" if passed else "REJECT"
    result_path = OUT / f"hybrid_retrieval_{mode}.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    print(f"VERDICT: {verdict}; wrote {result_path}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "confirm"}:
        raise SystemExit("mode must be 'screen' or 'confirm'")
    main(mode)
