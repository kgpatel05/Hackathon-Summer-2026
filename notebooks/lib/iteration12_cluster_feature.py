"""Iteration 12 - unsupervised joint-phenotype cluster as an ExtraTrees feature.

The 256-cluster posterior was 66.4% accurate alone but harmed a forced probability blend.
This follow-up asks the lower-risk question: can the incumbent learn *when* a compact
64-cluster joint phenotype is useful?  Clusters are fitted label-free after scaling and
PCA(50); a one-hot encoding is appended.  A row permutation is the matched width/marginal
null.  Screen: partition 919, five ET seeds; advance only for >0.30 point gain, p<0.05,
and >0.20 point advantage over null.  No test label is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration12"); OUT.mkdir(parents=True, exist_ok=True)
PARTITION = 919; SEEDS = tuple(range(5)); ALPHA = 0.45; N_CLUSTERS = 64


def oof(x: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    ok = np.zeros(len(y), bool); labels = np.asarray(classes)
    for train, valid in StratifiedKFold(5, shuffle=True, random_state=PARTITION).split(y, y):
        p = M.fit_extra_trees(x[train], pd.Series(y[train]), classes, x[valid], seeds=SEEDS)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[train]), classes), ALPHA)
        ok[valid] = labels[p.argmax(1)] == y[valid]
    return ok


def main() -> None:
    counts, meta, _, meta_test = F.load_challenge(); y = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y)); meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    base = current_stack(meta_all, classes, list(counts.columns))
    embedding = PCA(50, random_state=919).fit_transform(StandardScaler().fit_transform(base))
    cluster = MiniBatchKMeans(N_CLUSTERS, n_init=10, batch_size=1024,
                              random_state=919).fit_predict(embedding)
    try: encoder = OneHotEncoder(sparse_output=False)
    except TypeError: encoder = OneHotEncoder(sparse=False)
    real = encoder.fit_transform(cluster[:, None]).astype(np.float32)
    null = real[np.random.default_rng(20260819).permutation(len(real))]
    configs = {"baseline_694": base,
               "+ phenotype cluster": np.hstack([base, real]),
               "+ permuted cluster null": np.hstack([base, null])}
    print(f"clusters={real.shape[1]} median size={np.median(np.bincount(cluster)):.1f}", flush=True)
    results = {}; t0 = time.time()
    for name, x in configs.items():
        results[name] = oof(x.astype(np.float32), y, classes)
        print(f"finished {name} ({time.time()-t0:.1f}s)", flush=True)
    incumbent = results["baseline_694"]; rows = []
    for name, ok in results.items():
        if name == "baseline_694": p, wins, losses = 1.0, 0, 0
        else:
            p, _ = M.paired_mcnemar(ok, incumbent)
            wins = int((ok & ~incumbent).sum()); losses = int((incumbent & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-incumbent.mean()),
                     "wins": wins, "losses": losses, "p": p})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "cluster_feature_screen.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    real_row, null_row = rows[1], rows[2]
    passed = (real_row["gain_pt"] > 0.30 and real_row["p"] < 0.05 and
              real_row["gain_pt"] - null_row["gain_pt"] > 0.20)
    print("VERDICT: " + ("ADVANCE TO CONFIRM" if passed else "REJECT"), flush=True)


if __name__ == "__main__": main()
