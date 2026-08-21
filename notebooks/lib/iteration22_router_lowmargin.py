"""Stability audit for low-margin routing to an opinion-neighbour alternative.

The matched control in the opinion-kNN experiment suggested a new hypothesis: local
neighbours may be useful for *which label* to propose, while the frozen pool margin is
better for *when* to switch.  The hypothesis is fixed at k=40 and 0.55% coverage, then
audited across ten new 60/40 cell splits and two independent expert OOF partitions.

These repeated splits reuse cells and are not ten independent datasets; the output is a
stability diagnostic, not a license to call twenty p-values independent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import iteration22_router_common as C
import iteration22_router_opinion_knn as K


K_NEIGHBOURS = 40
COVERAGE = 0.0055
SPLIT_SEEDS = tuple(range(22101, 22111))


def main() -> None:
    t0 = time.time()
    b18 = C.load_experts(18)
    y = b18["y"]
    idx = np.arange(len(y))
    z18 = C.pool_logits(b18)
    base18 = b18["classes"][z18.argmax(1)]
    feat18 = K.opinion_features(b18, z18)
    glia = b18["meta"]["Region"].isna().to_numpy()
    query_banks = {s: C.load_experts(s) for s in (59, 83)}
    query_z = {s: C.pool_logits(query_banks[s]) for s in query_banks}
    query_feat = {s: K.opinion_features(query_banks[s], query_z[s]) for s in query_banks}
    rows = []

    for split_seed in SPLIT_SEEDS:
        design, confirm = train_test_split(idx, test_size=0.40, random_state=split_seed,
                                           stratify=y)
        scaler, pca, ref_vec = K.fit_geometry(feat18[design])
        for seed in (59, 83):
            bank, z = query_banks[seed], query_z[seed]
            base = bank["classes"][z.argmax(1)]
            q = K.transform(query_feat[seed], scaler, pca)
            local = K.local_distribution(
                ref_vec, base18[design], y[design], q[confirm], base[confirm],
                bank["allow"][confirm], bank["classes"], K_NEIGHBOURS, design, confirm)
            alt = bank["classes"][local.argmax(1)]
            eligible = np.flatnonzero(alt != base[confirm])
            order = np.argsort(-z[confirm], axis=1)
            margin = (z[confirm][np.arange(len(confirm)), order[:, 0]]
                      - z[confirm][np.arange(len(confirm)), order[:, 1]])
            n_change = min(int(round(COVERAGE * len(confirm))), len(eligible))
            take = eligible[np.argsort(margin[eligible])[:n_change]]
            pred = base[confirm].copy(); pred[take] = alt[take]
            r = C.metric_row("lowmargin_opinion_router", pred, base[confirm], y[confirm],
                             glia[confirm], -margin)
            rng = np.random.default_rng(split_seed + seed)
            rnd = rng.choice(eligible, n_change, replace=False)
            pr = base[confirm].copy(); pr[rnd] = alt[rnd]
            rc = C.metric_row("random_same_coverage", pr, base[confirm], y[confirm],
                              glia[confirm], -margin)
            for row in (r, rc):
                row.update(split_seed=split_seed, partition=seed, k=K_NEIGHBOURS,
                           fixed_coverage=COVERAGE,
                           device="cpu (PCA/cosine kNN; MPS not applicable)")
                rows.append(row)
        print(f"split {split_seed} complete", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(C.OUT / "lowmargin_stability.csv", index=False)
    router = result[result.candidate == "lowmargin_opinion_router"]
    control = result[result.candidate == "random_same_coverage"]
    summary = {
        "mechanism": "opinion-space alternative, pool-margin abstention",
        "k": K_NEIGHBOURS, "fixed_coverage": COVERAGE,
        "cell_splits": len(SPLIT_SEEDS), "query_partitions": [59, 83],
        "evaluations": int(len(router)),
        "positive_evaluations": int((router.net > 0).sum()),
        "zero_evaluations": int((router.net == 0).sum()),
        "negative_evaluations": int((router.net < 0).sum()),
        "mean_net_cells": float(router.net.mean()),
        "median_net_cells": float(router.net.median()),
        "mean_gain_pt": float(100 * router.net.mean() / 2000),
        "mean_random_net_cells": float(control.net.mean()),
        "wins_total": int(router.wins.sum()), "losses_total": int(router.losses.sum()),
        "confirmed": False,
        "test_scoring_authorized": False,
        "reason": "Repeated splits reuse labels; stability gate requires >=16/20 positive and mean net >=2",
        "runtime_sec": time.time() - t0,
        "device": "cpu", "device_reason": "PCA/cosine kNN; MPS not applicable",
    }
    stable = summary["positive_evaluations"] >= 16 and summary["mean_net_cells"] >= 2.0
    summary["stability_gate_passed"] = stable
    summary["confirmed"] = stable
    summary["test_scoring_authorized"] = stable
    (C.OUT / "lowmargin_freeze.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(router[["split_seed", "partition", "wins", "losses", "net", "coverage"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
