"""Cross-fitted selective error correction with abstention and matched controls.

Screening is restricted to expert OOF partition 18.  The winning objective, model
capacity and abstention threshold are frozen, then evaluated unchanged on partitions
41, 59 and 83.  No recovered test label is imported here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration22_router_common as C


SCREEN_SEED = 18
CONFIRM_SEEDS = (41, 59, 83)
CV_SEED = 22026
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
CONFIGS = {
    "discordant_d2": dict(kind="discordant", max_depth=2, rounds=260,
                           min_child_weight=20, reg_lambda=12.0),
    "discordant_d3": dict(kind="discordant", max_depth=3, rounds=320,
                           min_child_weight=25, reg_lambda=16.0),
    "utility_d2": dict(kind="utility", max_depth=2, rounds=300,
                        min_child_weight=20, reg_lambda=12.0),
}


def _params(config: dict, seed: int) -> dict:
    kind = config["kind"]
    return dict(
        objective=("binary:logistic" if kind == "discordant" else "reg:squarederror"),
        eval_metric=("logloss" if kind == "discordant" else "rmse"),
        eta=0.035, max_depth=config["max_depth"], subsample=0.80,
        colsample_bytree=0.70, min_child_weight=config["min_child_weight"],
        reg_lambda=config["reg_lambda"], reg_alpha=0.15,
        tree_method="hist", nthread=8, seed=seed,
    )


def crossfit(bank: dict, config: dict) -> dict:
    import xgboost as xgb

    z = C.pool_logits(bank)
    cand, source = C.proposals(bank, z)
    x, feature_names = C.build_features(bank, z, cand, source)
    y = bank["y"]
    classes = bank["classes"]
    base_idx = z.argmax(1)
    base = classes[base_idx]
    utility = (classes[cand] == y[:, None]).astype(np.int8)
    utility -= (base == y).astype(np.int8)[:, None]
    scores = np.full(cand.shape, -10.0, np.float32)
    folds = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
    t0 = time.time()
    for fold, (fit, val) in enumerate(folds.split(np.zeros(len(y)), y)):
        xf = x[fit].reshape(-1, x.shape[2])
        uf = utility[fit].reshape(-1)
        if config["kind"] == "discordant":
            keep = uf != 0
            label = (uf[keep] > 0).astype(np.float32)
            dfit = xgb.DMatrix(xf[keep], label=label, feature_names=feature_names)
        else:
            weight = np.where(uf == 0, 0.20, 1.0).astype(np.float32)
            dfit = xgb.DMatrix(xf, label=uf.astype(np.float32), weight=weight,
                               feature_names=feature_names)
        model = xgb.train(_params(config, CV_SEED + fold), dfit,
                          num_boost_round=config["rounds"])
        dval = xgb.DMatrix(x[val].reshape(-1, x.shape[2]), feature_names=feature_names)
        scores[val] = model.predict(dval).reshape(len(val), cand.shape[1])
    best_j = scores.argmax(1)
    rows = np.arange(len(y))
    best_score = scores[rows, best_j]
    alt = classes[cand[rows, best_j]]
    return dict(base=base, alt=alt, score=best_score, utility=utility,
                cand=cand, z=z, runtime=time.time() - t0, features=x.shape[2])


def _controls(base: np.ndarray, alt: np.ndarray, truth: np.ndarray,
              glia: np.ndarray, score: np.ndarray, z: np.ndarray,
              n_change: int, seed: int) -> list[dict]:
    eligible = np.flatnonzero(alt != base)
    n_change = min(n_change, len(eligible))
    out = []
    rng = np.random.default_rng(9000 + seed)
    random_idx = rng.choice(eligible, n_change, replace=False)
    pred = base.copy(); pred[random_idx] = alt[random_idx]
    out.append(C.metric_row("random_same_coverage", pred, base, truth, glia, score))

    order = np.argsort(-z, axis=1)
    margin = z[np.arange(len(z)), order[:, 0]] - z[np.arange(len(z)), order[:, 1]]
    margin_idx = eligible[np.argsort(margin[eligible])[:n_change]]
    pred = base.copy(); pred[margin_idx] = alt[margin_idx]
    out.append(C.metric_row("low_margin_same_coverage", pred, base, truth, glia, score))

    score_idx = eligible[np.argsort(-score[eligible])[:n_change]]
    pred = base.copy(); pred[score_idx] = alt[score_idx]
    out.append(C.metric_row("score_rank_same_coverage", pred, base, truth, glia, score))
    return out


def evaluate(seed: int, tag: str, config: dict, threshold: float) -> tuple[pd.DataFrame, dict]:
    bank = C.load_experts(seed)
    run = crossfit(bank, config)
    truth = bank["y"]
    glia = bank["meta"]["Region"].isna().to_numpy()
    base = run["base"]
    pred = base.copy()
    take = (run["score"] >= threshold) & (run["alt"] != base)
    pred[take] = run["alt"][take]
    rows = [C.metric_row("iteration21_pool", base, base, truth, glia),
            C.metric_row(tag, pred, base, truth, glia, run["score"])]
    rows += _controls(base, run["alt"], truth, glia, run["score"], run["z"],
                      int(take.sum()), seed)
    frame = pd.DataFrame(rows)
    frame.insert(0, "partition", seed)
    frame["runtime_sec"] = run["runtime"]
    frame["device"] = "cpu (MPS unavailable; XGBoost hist)"
    return frame, run


def screen() -> tuple[str, float, pd.DataFrame]:
    rows = []
    cache = {}
    for tag, config in CONFIGS.items():
        bank = C.load_experts(SCREEN_SEED)
        run = crossfit(bank, config)
        cache[tag] = run
        truth = bank["y"]
        glia = bank["meta"]["Region"].isna().to_numpy()
        for threshold in THRESHOLDS:
            pred = run["base"].copy()
            take = run["score"] >= threshold
            pred[take] = run["alt"][take]
            row = C.metric_row(tag, pred, run["base"], truth, glia, run["score"])
            row.update(partition=SCREEN_SEED, threshold=threshold,
                       runtime_sec=run["runtime"], features=run["features"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(C.OUT / "screen_grid.csv", index=False)

    # Predeclared safety selection: require positive net, at least 20 changes, and
    # prefer the largest lower-bound-like score net/sqrt(changes), not raw accuracy.
    eligible = frame[(frame.net > 0) & (frame.changed >= 20)].copy()
    if eligible.empty:
        raise RuntimeError("no screen candidate has positive net utility")
    eligible["safety_score"] = eligible.net / np.sqrt(eligible.changed)
    best = eligible.sort_values(["safety_score", "net"], ascending=False).iloc[0]
    return str(best.candidate), float(best.threshold), frame


def main() -> None:
    t0 = time.time()
    tag, threshold, screen_frame = screen()
    print(f"frozen survivor from partition {SCREEN_SEED}: {tag}, threshold={threshold:.2f}")
    print(screen_frame.sort_values(["net", "changed"], ascending=False).head(12).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    all_rows = []
    for seed in CONFIRM_SEEDS:
        frame, _ = evaluate(seed, tag, CONFIGS[tag], threshold)
        all_rows.append(frame)
        route = frame[frame.candidate == tag].iloc[0]
        print(f"confirm {seed}: acc={route.accuracy:.4f} net={route.net:+d} "
              f"wins/losses={route.wins}/{route.losses} p={route.mcnemar_p:.4g} "
              f"coverage={route.coverage:.3f}", flush=True)
    confirm = pd.concat(all_rows, ignore_index=True)
    confirm.to_csv(C.OUT / "confirmation.csv", index=False)
    route = confirm[confirm.candidate == tag]
    base = confirm[confirm.candidate == "iteration21_pool"]
    gains = 100 * (route.accuracy.to_numpy() - base.accuracy.to_numpy())
    confirmed = bool((gains > 0).all() and gains.mean() >= 0.15 and route.net.sum() >= 15)
    freeze = {
        "screen_partition": SCREEN_SEED,
        "confirmation_partitions": list(CONFIRM_SEEDS),
        "candidate": tag,
        "config": CONFIGS[tag],
        "threshold": threshold,
        "confirmation_gain_pt": gains.tolist(),
        "mean_confirmation_gain_pt": float(gains.mean()),
        "worst_confirmation_gain_pt": float(gains.min()),
        "confirmed": confirmed,
        "test_scoring_authorized": confirmed,
        "test_truth_read": False,
        "device": "cpu",
        "device_reason": "torch MPS backend unavailable; XGBoost histogram router",
        "runtime_sec": time.time() - t0,
    }
    (C.OUT / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
