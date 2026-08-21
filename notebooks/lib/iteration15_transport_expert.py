"""Iteration 15c - prior-deconvolved fused-transport expert.

Appending the 61 fused-transport columns to ExtraTrees gained +0.26 point in its
screen, but a matched shuffled-label block gained +0.18 and the real candidate
was non-significant.  This experiment avoids the feature-width effect entirely.

For each section, the balanced transport posterior is divided by its section
mean.  The ratio is local evidence: 1 means the cell receives the section's
ordinary class mixture; >1 means its fused spatial/molecular catchment is
enriched for that class.  The incumbent posterior is corrected once:

    p_new(class|cell) proportional to
        p_incumbent(class|cell) * local_enrichment ** 0.25

The enrichment is shrunk halfway to one before exponentiation.  The exponent
0.25 and shrinkage 0.5 are frozen before evaluating this script; there is no
weight search.  The matched null uses the identical transport plans with donor
labels permuted within section.  No model is fitted to transport labels and no
test truth is read.

Screen: fold partition 421, five incumbent seeds.  Advance only for >0.30 point,
paired exact McNemar p<0.05, and >0.20 point over the null.  Confirmation, if
eligible, uses partition 443 and twenty incumbent seeds and requires >0.20 point
with p<0.05.  This script never writes a submission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as OT


OUT = Path("outputs/iteration15")
TRANSPORT_CACHE = OUT / "atlas_fused_transport.npz"
SCREEN_PATH = OUT / "transport_expert_screen.csv"
SCREEN_PARTITION = 421
CONFIRM_PARTITION = 443
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
EVIDENCE_EXPONENT = 0.25
EVIDENCE_WEIGHT = 0.50
EPSILON = 1e-5


def enrichment(block: np.ndarray, meta_all: pd.DataFrame,
               n_classes: int) -> np.ndarray:
    """Prior-deconvolve a transport posterior and shrink the ratio to one."""
    posterior = block[:, :n_classes].astype(np.float64) + EPSILON
    posterior /= posterior.sum(1, keepdims=True)
    output = np.ones_like(posterior)
    sections = meta_all["Section_ID"].astype(str).to_numpy()
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        prior = posterior[rows].mean(0)
        ratio = posterior[rows] / np.maximum(prior[None, :], EPSILON)
        output[rows] = (1.0 - EVIDENCE_WEIGHT) + EVIDENCE_WEIGHT * ratio
    return output.astype(np.float32)


def correct(probabilities: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    adjusted = probabilities * np.power(np.maximum(evidence, EPSILON), EVIDENCE_EXPONENT)
    adjusted /= np.maximum(adjusted.sum(1, keepdims=True), EPSILON)
    return adjusted.astype(np.float32)


def evaluate(mode: str, x: np.ndarray, y: np.ndarray, meta_train: pd.DataFrame,
             meta_all: pd.DataFrame, classes: list[str], real: np.ndarray,
             null: np.ndarray) -> pd.DataFrame:
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    base = OT.oof_probabilities(x, y, meta_train, classes, partition, seeds)
    real_evidence = enrichment(real, meta_all, len(classes))[:len(y)]
    null_evidence = enrichment(null, meta_all, len(classes))[:len(y)]
    variants = {
        "incumbent_694": base,
        "+ prior-deconvolved fused transport": correct(base, real_evidence),
        "+ shuffled transport expert (null)": correct(base, null_evidence),
    }
    class_array = np.asarray(classes)
    glia = meta_train["Region"].isna().to_numpy()
    base_correct = class_array[base.argmax(1)] == y
    rows = []
    for name, probabilities in variants.items():
        is_correct = class_array[probabilities.argmax(1)] == y
        if name == "incumbent_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, table = M.paired_mcnemar(is_correct, base_correct)
            wins, losses = table[0][1], table[1][0]
        rows.append({
            "mode": mode, "partition": partition, "config": name,
            "accuracy": is_correct.mean(),
            "gain_pt": 100 * (is_correct.mean() - base_correct.mean()),
            "glia_accuracy": is_correct[glia].mean(),
            "neuron_accuracy": is_correct[~glia].mean(),
            "changed_predictions": int(np.sum(probabilities.argmax(1) != base.argmax(1))),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"transport_expert_{mode}.csv", index=False)
    np.savez_compressed(
        OUT / f"transport_expert_{mode}_oof.npz",
        base=base, real=variants["+ prior-deconvolved fused transport"],
        null=variants["+ shuffled transport expert (null)"], truth=y,
        classes=class_array, partition=partition,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    return result


def passes(result: pd.DataFrame, confirmation: bool) -> bool:
    rows = result.set_index("config")
    real = rows.loc["+ prior-deconvolved fused transport"]
    if confirmation:
        return bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
    null = rows.loc["+ shuffled transport expert (null)"]
    return bool(real.gain_pt > 0.30 and
                real.gain_pt - null.gain_pt > 0.20 and
                real.mcnemar_p < 0.05)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"screen", "confirm", "run"}:
        raise SystemExit("mode must be screen, confirm, or run")
    counts_train, meta_train, _, meta_test = F.load_challenge()
    del counts_train
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    cached = np.load(TRANSPORT_CACHE, allow_pickle=True)
    if list(cached["classes"].astype(str)) != classes:
        raise ValueError("fused transport class order mismatch")
    real, null = cached["real"], cached["null"]
    x, _ = OT.load_incumbent()

    if mode in {"screen", "run"}:
        screen = evaluate("screen", x, y, meta_train, meta_all, classes, real, null)
        advanced = passes(screen, confirmation=False)
        print("SCREEN VERDICT: " + ("ADVANCE" if advanced else "REJECT"), flush=True)
        with open(OUT / "transport_expert_decision.json", "w", encoding="utf-8") as handle:
            json.dump({"screen_passed": advanced, "confirm_passed": None}, handle, indent=2)
        if not advanced:
            return
    if mode == "confirm":
        if not SCREEN_PATH.exists():
            raise SystemExit("confirmation requires the saved screen result")
        if not passes(pd.read_csv(SCREEN_PATH), confirmation=False):
            raise SystemExit("screen gate failed; confirmation is intentionally blocked")
    confirmation = evaluate(
        "confirm", x, y, meta_train, meta_all, classes, real, null
    )
    adopted = passes(confirmation, confirmation=True)
    print("CONFIRM VERDICT: " + ("ADOPT CANDIDATE" if adopted else "REJECT"), flush=True)
    with open(OUT / "transport_expert_decision.json", "w", encoding="utf-8") as handle:
        json.dump({"screen_passed": True, "confirm_passed": adopted}, handle, indent=2)


if __name__ == "__main__":
    main()
