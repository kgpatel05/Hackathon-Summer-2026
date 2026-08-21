"""Iteration 12 - transductive phenotype-cluster evidence pooling.

Physical neighbors have only 9.4% label homophily, but cells close in the complete
694-feature phenotype stack can still pool evidence.  This experiment builds 256
label-free MiniBatchKMeans clusters after StandardScaler + PCA(50) on all challenge
training features.  Inside each OOF fold, each validation cell receives a smoothed class
histogram computed from *fold-training* members of its cluster.  No validation label
enters its posterior.

A row-permuted cluster assignment preserves cluster sizes but destroys phenotype.  The
incumbent OOF probabilities are reused from the untouched partition-307 CatBoost screen.
The only candidate is a fixed 90/10 probability blend.  Advance only for >0.30 point
gain, p<0.05, and >0.20 point advantage over the null blend.  No test label is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("outputs/iteration10/catboost_screen_oof.npz")
PARTITION = 307
N_CLUSTERS = 256
SMOOTHING = 10.0
WEIGHT = 0.10


def cluster_posterior(cluster: np.ndarray, y: np.ndarray,
                      classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), len(classes)), np.float32)
    class_index = {name: j for j, name in enumerate(classes)}
    folds = StratifiedKFold(5, shuffle=True, random_state=PARTITION)
    for train, valid in folds.split(y, y):
        global_counts = np.asarray([(y[train] == c).sum() for c in classes], float)
        global_prior = global_counts / global_counts.sum()
        table = np.zeros((N_CLUSTERS, len(classes)), np.float64)
        totals = np.zeros(N_CLUSTERS, np.float64)
        for row in train:
            table[cluster[row], class_index[y[row]]] += 1
            totals[cluster[row]] += 1
        table += SMOOTHING * global_prior[None, :]
        table /= (totals[:, None] + SMOOTHING)
        out[valid] = table[cluster[valid]]
    return out


def main() -> None:
    counts_train, meta_train, _, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = np.asarray(sorted(set(y)))
    cached = np.load(CACHE, allow_pickle=True)
    if not np.array_equal(cached["y"].astype(str), y):
        raise ValueError("cached OOF order mismatch")
    et = cached["et"].astype(np.float32)
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes.tolist(), list(counts_train.columns))
    scaled = StandardScaler().fit_transform(x)
    embedding = PCA(50, random_state=307).fit_transform(scaled)
    cluster = MiniBatchKMeans(
        N_CLUSTERS, batch_size=1024, n_init=10, random_state=307
    ).fit_predict(embedding)
    rng = np.random.default_rng(20260819)
    null_cluster = rng.permutation(cluster)
    pooled = cluster_posterior(cluster, y, classes)
    null = cluster_posterior(null_cluster, y, classes)
    print(f"clusters={N_CLUSTERS} median size={np.median(np.bincount(cluster)):.1f} "
          f"empty={N_CLUSTERS-len(np.unique(cluster))}", flush=True)

    variants = {
        "ExtraTrees incumbent": et,
        "cluster posterior": pooled,
        "0.90 ET + 0.10 cluster": (1-WEIGHT)*et + WEIGHT*pooled,
        "0.90 ET + 0.10 random-cluster null": (1-WEIGHT)*et + WEIGHT*null,
    }
    base_ok = classes[et.argmax(1)] == y; rows = []
    for name, pmat in variants.items():
        ok = classes[pmat.argmax(1)] == y
        if name == "ExtraTrees incumbent": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum()); losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "cluster_pooling_screen.csv", index=False)
    np.savez_compressed(OUT / "cluster_pooling_screen.npz", et=et, pooled=pooled,
                        null=null, cluster=cluster, y=y, classes=classes)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real, null_row = rows[2], rows[3]
    passed = (real["gain_pt"] > 0.30 and real["p"] < 0.05 and
              real["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
