"""Iteration 15d - mechanism-selective transport arbitration.

The generic prior-deconvolved transport expert changed 214 screen predictions
and gained only +0.16 point.  Its errors were mechanistically heterogeneous:
the full-atlas spatial context is informative for anatomically segregated glial
subtypes, but not for transcript-defined neuronal subtypes.  This router limits
the already-frozen transport correction to three unordered pairs whose spatial
separation was documented before Iteration 15 (SCORECARD sections 9 and 11):

* astrocyte_1 / astrocyte_2 (atlas-neighbour AUC 0.954);
* meninges_1 / meninges_2 (13--27x local enrichment); and
* oligodendrocyte_progenitor_2 / oligodendrocyte_2 (AUC 0.806).

It is deliberately not a general pairwise model.  A call changes only when the
incumbent and the fixed fused-transport expert choose opposite members of one
of those pairs.  The expert uses the fixed 0.25 evidence exponent and 0.5
shrinkage from iteration15_transport_expert.py.  The matched null routes an
identical within-section label-permuted transport expert.

This candidate was fixed after the generic expert diagnostic and before its
fresh screen: partition 463, five ExtraTrees seeds.  Advance only for >0.30
point, >0.20 point over null, and paired exact McNemar p<0.05.  Confirmation is
partition 487 with twenty seeds and requires >0.20 point and p<0.05.  No test
truth is read and no submission is written.
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
import iteration15_transport_expert as E


OUT = Path("outputs/iteration15")
TRANSPORT_CACHE = OUT / "atlas_fused_transport.npz"
SCREEN_PATH = OUT / "spatial_pair_router_screen.csv"
SCREEN_PARTITION = 463
CONFIRM_PARTITION = 487
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
ANATOMICAL_PAIRS = frozenset({
    frozenset(("astrocyte_1", "astrocyte_2")),
    frozenset(("meninges_1", "meninges_2")),
    frozenset(("oligodendrocyte_progenitor_2", "oligodendrocyte_2")),
})


def route(base: np.ndarray, expert: np.ndarray,
          classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Use expert probabilities only for a pre-specified anatomical pair."""
    base_label = np.asarray(classes)[base.argmax(1)]
    expert_label = np.asarray(classes)[expert.argmax(1)]
    use = np.asarray([
        b != e and frozenset((b, e)) in ANATOMICAL_PAIRS
        for b, e in zip(base_label, expert_label)
    ])
    output = base.copy()
    output[use] = expert[use]
    return output, use


def evaluate(mode: str, x: np.ndarray, y: np.ndarray, meta_train: pd.DataFrame,
             meta_all: pd.DataFrame, classes: list[str], real: np.ndarray,
             null: np.ndarray) -> pd.DataFrame:
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    base = OT.oof_probabilities(x, y, meta_train, classes, partition, seeds)
    real_expert = E.correct(
        base, E.enrichment(real, meta_all, len(classes))[:len(y)]
    )
    null_expert = E.correct(
        base, E.enrichment(null, meta_all, len(classes))[:len(y)]
    )
    real_routed, real_use = route(base, real_expert, classes)
    null_routed, null_use = route(base, null_expert, classes)
    variants = {
        "incumbent_694": (base, np.zeros(len(y), bool)),
        "+ anatomical-pair transport router": (real_routed, real_use),
        "+ shuffled transport router (null)": (null_routed, null_use),
    }
    class_array = np.asarray(classes)
    glia = meta_train["Region"].isna().to_numpy()
    base_correct = class_array[base.argmax(1)] == y
    rows = []
    for name, (probabilities, used) in variants.items():
        correct = class_array[probabilities.argmax(1)] == y
        if name == "incumbent_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, table = M.paired_mcnemar(correct, base_correct)
            wins, losses = table[0][1], table[1][0]
        rows.append({
            "mode": mode, "partition": partition, "config": name,
            "accuracy": correct.mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "routed_cells": int(used.sum()),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"spatial_pair_router_{mode}.csv", index=False)
    np.savez_compressed(
        OUT / f"spatial_pair_router_{mode}_oof.npz",
        base=base, real=real_routed, null=null_routed, truth=y,
        classes=class_array, partition=partition,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    return result


def passes(result: pd.DataFrame, confirmation: bool) -> bool:
    rows = result.set_index("config")
    real = rows.loc["+ anatomical-pair transport router"]
    if confirmation:
        return bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
    null = rows.loc["+ shuffled transport router (null)"]
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
    x, _ = OT.load_incumbent()

    if mode in {"screen", "run"}:
        screen = evaluate(
            "screen", x, y, meta_train, meta_all, classes,
            cached["real"], cached["null"],
        )
        advanced = passes(screen, confirmation=False)
        print("SCREEN VERDICT: " + ("ADVANCE" if advanced else "REJECT"), flush=True)
        with open(OUT / "spatial_pair_router_decision.json", "w", encoding="utf-8") as handle:
            json.dump({"screen_passed": advanced, "confirm_passed": None}, handle, indent=2)
        if not advanced:
            return
    if mode == "confirm":
        if not SCREEN_PATH.exists():
            raise SystemExit("confirmation requires the saved screen result")
        if not passes(pd.read_csv(SCREEN_PATH), confirmation=False):
            raise SystemExit("screen gate failed; confirmation is intentionally blocked")
    confirmation = evaluate(
        "confirm", x, y, meta_train, meta_all, classes,
        cached["real"], cached["null"],
    )
    adopted = passes(confirmation, confirmation=True)
    print("CONFIRM VERDICT: " + ("ADOPT CANDIDATE" if adopted else "REJECT"), flush=True)
    with open(OUT / "spatial_pair_router_decision.json", "w", encoding="utf-8") as handle:
        json.dump({"screen_passed": True, "confirm_passed": adopted}, handle, indent=2)


if __name__ == "__main__":
    main()
