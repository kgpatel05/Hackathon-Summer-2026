"""Cell-disjoint empirical-Bayes correction rules for stable class confusions.

Instead of learning a high-capacity per-cell ranker, this mechanism asks a much lower
variance question: when a specific independent expert repeatedly proposes class B while
the frozen pool calls class A, is that directional correction reliably beneficial?

Rules are fitted on a fixed 60% design cell set using partition-18 OOF opinions,
selected using partition-41 opinions on those design cells, and confirmed exactly once
on the disjoint remaining 40% of cells with partitions 59 and 83.  Confirmation labels
therefore never influence a rule, source, support threshold, or posterior threshold.
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


SOURCES = ("pool2", "plurality", "atlas_plurality", "tree_plurality",
           "etaug4_0.25_3", "atlaslam_lin", "atlasftlam")
MIN_SUPPORT = (3, 5, 8, 12)
POSTERIOR_RATE = (0.60, 0.65, 0.70, 0.75)
SPLIT_SEED = 20260821


def alternatives(bank: dict, z: np.ndarray) -> dict[str, np.ndarray]:
    classes = bank["classes"]
    n = len(z)
    rows = np.arange(n)
    base_i = z.argmax(1)
    order = np.argsort(-z, axis=1)
    out = {"pool2": classes[order[:, 1]]}
    pred = np.where(bank["allow"][None], bank["probs"], -1).argmax(2)

    def plurality(idx: list[int]) -> np.ndarray:
        votes = np.zeros_like(z)
        for m in idx:
            votes[rows, pred[m]] += 1
        votes[rows, base_i] = -1
        votes[~bank["allow"]] = -1
        alt_i = votes.argmax(1)
        # If only the incumbent is allowed, abstain explicitly.
        alt_i[bank["allow"].sum(1) == 1] = base_i[bank["allow"].sum(1) == 1]
        return classes[alt_i]

    out["plurality"] = plurality(list(range(len(bank["names"]))))
    out["atlas_plurality"] = plurality([
        i for i, s in enumerate(bank["names"]) if s.startswith("atlas")])
    out["tree_plurality"] = plurality([
        i for i, s in enumerate(bank["names"])
        if s.startswith("et") or s in ("rf", "xgb", "xgbaug")])
    for name in ("etaug4_0.25_3", "atlaslam_lin", "atlasftlam"):
        m = bank["names"].index(name)
        alt_i = pred[m].copy()
        alt_i[~bank["allow"][rows, alt_i]] = base_i[~bank["allow"][rows, alt_i]]
        out[name] = classes[alt_i]
    return out


def fit_rules(base: np.ndarray, alt: np.ndarray, truth: np.ndarray, rows: np.ndarray,
              min_support: int, min_rate: float) -> set[tuple[str, str]]:
    table: dict[tuple[str, str], list[int]] = {}
    for i in rows:
        if alt[i] == base[i]:
            continue
        win = alt[i] == truth[i] and base[i] != truth[i]
        loss = base[i] == truth[i] and alt[i] != truth[i]
        if not (win or loss):
            continue
        key = (str(base[i]), str(alt[i]))
        table.setdefault(key, [0, 0])
        table[key][0 if win else 1] += 1
    # Beta(2,2) shrinkage prevents one-off directional coincidences from routing.
    return {key for key, (w, l) in table.items()
            if w + l >= min_support and (w + 2) / (w + l + 4) >= min_rate and w > l}


def apply_rules(base: np.ndarray, alt: np.ndarray, rules: set[tuple[str, str]],
                rows: np.ndarray | None = None) -> np.ndarray:
    pred = base.copy()
    take = np.array([(str(a), str(b)) in rules and a != b for a, b in zip(base, alt)])
    if rows is not None:
        keep = np.zeros(len(base), bool); keep[rows] = True; take &= keep
    pred[take] = alt[take]
    return pred


def score(name: str, pred: np.ndarray, base: np.ndarray, truth: np.ndarray,
          glia: np.ndarray, rows: np.ndarray) -> dict:
    row = C.metric_row(name, pred[rows], base[rows], truth[rows], glia[rows])
    return row


def main() -> None:
    t0 = time.time()
    train_bank = C.load_experts(18)
    select_bank = C.load_experts(41)
    y = train_bank["y"]
    idx = np.arange(len(y))
    design, confirm = train_test_split(idx, test_size=0.40, random_state=SPLIT_SEED,
                                       stratify=y)
    z18, z41 = C.pool_logits(train_bank), C.pool_logits(select_bank)
    base18 = train_bank["classes"][z18.argmax(1)]
    base41 = select_bank["classes"][z41.argmax(1)]
    alt18, alt41 = alternatives(train_bank, z18), alternatives(select_bank, z41)
    glia = train_bank["meta"]["Region"].isna().to_numpy()

    rows = []
    rule_bank = {}
    for source in SOURCES:
        for support in MIN_SUPPORT:
            for rate in POSTERIOR_RATE:
                rules = fit_rules(base18, alt18[source], y, design, support, rate)
                pred = apply_rules(base41, alt41[source], rules)
                r = score(source, pred, base41, y, glia, design)
                r.update(source=source, min_support=support, posterior_rate=rate,
                         n_rules=len(rules), partition=41, split="design_selection")
                rows.append(r)
                rule_bank[(source, support, rate)] = rules
    grid = pd.DataFrame(rows)
    grid.to_csv(C.OUT / "pairrule_screen.csv", index=False)

    eligible = grid[(grid.net > 0) & (grid.changed >= 5)].copy()
    if eligible.empty:
        best = grid.sort_values(["net", "wins", "changed"], ascending=False).iloc[0]
    else:
        eligible["safety"] = eligible.net / np.sqrt(eligible.changed)
        best = eligible.sort_values(["safety", "net"], ascending=False).iloc[0]
    key = (str(best.source), int(best.min_support), float(best.posterior_rate))
    frozen_rules = rule_bank[key]
    print("screen winner:")
    print(best.to_string())
    print(f"rules ({len(frozen_rules)}): {sorted(frozen_rules)}")

    confirm_rows = []
    for seed in (59, 83):
        bank = C.load_experts(seed)
        z = C.pool_logits(bank)
        base = bank["classes"][z.argmax(1)]
        alt = alternatives(bank, z)[key[0]]
        pred = apply_rules(base, alt, frozen_rules)
        r0 = score("iteration21_pool", base, base, y, glia, confirm)
        r1 = score("pairrule_router", pred, base, y, glia, confirm)
        changed = int((pred[confirm] != base[confirm]).sum())
        eligible = confirm[alt[confirm] != base[confirm]]
        rng = np.random.default_rng(22000 + seed)
        random_idx = rng.choice(eligible, min(changed, len(eligible)), replace=False)
        pr = base.copy(); pr[random_idx] = alt[random_idx]
        rr = score("random_same_coverage", pr, base, y, glia, confirm)
        order = np.argsort(-z, axis=1)
        margin = z[np.arange(len(z)), order[:, 0]] - z[np.arange(len(z)), order[:, 1]]
        low = eligible[np.argsort(margin[eligible])[:changed]]
        pm = base.copy(); pm[low] = alt[low]
        rm = score("low_margin_same_coverage", pm, base, y, glia, confirm)
        for r in (r0, r1, rr, rm):
            r.update(partition=seed, split="cell_disjoint_confirmation",
                     source=key[0], min_support=key[1], posterior_rate=key[2],
                     n_rules=len(frozen_rules), runtime_sec=time.time() - t0,
                     device="cpu (lookup rule; MPS not applicable)")
        confirm_rows.extend([r0, r1, rr, rm])
        print(f"confirm {seed}: pool={r0['accuracy']:.4f} router={r1['accuracy']:.4f} "
              f"net={r1['net']:+d} wins/losses={r1['wins']}/{r1['losses']} "
              f"p={r1['mcnemar_p']:.4g} coverage={r1['coverage']:.3f}")
    cf = pd.DataFrame(confirm_rows)
    cf.to_csv(C.OUT / "pairrule_confirmation.csv", index=False)
    routed = cf[cf.candidate == "pairrule_router"].reset_index(drop=True)
    base = cf[cf.candidate == "iteration21_pool"].reset_index(drop=True)
    gains = 100 * (routed.accuracy - base.accuracy)
    confirmed_ok = bool((gains > 0).all() and gains.mean() >= 0.10
                        and routed.net.sum() >= 4)
    freeze = {
        "mechanism": "cell-disjoint empirical-Bayes directional pair rules",
        "design_cells": int(len(design)), "confirmation_cells": int(len(confirm)),
        "fit_partition": 18, "selection_partition": 41,
        "confirmation_partitions": [59, 83], "source": key[0],
        "min_support": key[1], "posterior_rate": key[2],
        "rules": [list(v) for v in sorted(frozen_rules)],
        "confirmation_gain_pt": gains.tolist(),
        "mean_confirmation_gain_pt": float(gains.mean()),
        "worst_confirmation_gain_pt": float(gains.min()),
        "confirmed": confirmed_ok, "test_scoring_authorized": confirmed_ok,
        "test_truth_read": False, "runtime_sec": time.time() - t0,
        "device": "cpu", "device_reason": "rule lookup; MPS not applicable",
    }
    freeze_path = C.OUT / "pairrule_freeze.json"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text())
        if existing.get("test_truth_read"):
            for field in ("test_truth_read", "test_result_file", "prediction_file",
                          "test_accuracy", "test_gain_pt", "test_net",
                          "base_reconstruction_mismatch"):
                if field in existing:
                    freeze[field] = existing[field]
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
