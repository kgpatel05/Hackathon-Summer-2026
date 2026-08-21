"""Iteration 15 - equal-capacity atlas optimal transport.

Motivation
----------
The 10,000 challenge cells are a sparse subsample of the 146,621-cell parent
atlas.  The adopted atlas-composition feature independently takes the ten
nearest non-challenge atlas cells for every query.  That operation has no
capacity constraint: one atlas cell can support many challenge cells, while
large parts of a tissue section may support none.

This experiment models the subsampling mechanism jointly.  Within each tissue
section it solves an entropy-regularised balanced optimal-transport problem
from every non-challenge atlas cell to every challenge cell.  Source cells have
uniform mass and challenge cells have uniform capacity, so each challenge cell
receives one equal-mass spatial catchment.  The feature is the public atlas
cell-type distribution transported into that catchment.

The construction uses only Section_ID and center_x/center_y for challenge
cells, plus public annotations on atlas cells after removing all 10,000
challenge cells.  It never imports a challenge-cell label from the atlas,
never reads recovered test truth, and never reads a withheld gene.

Null and decision rule
----------------------
The matched null applies the *same transport plan* after permuting atlas labels
within each section.  It therefore preserves transport geometry, feature
width, section composition and class marginals while destroying local biology.

SCREEN (partition 271, five folds, five ExtraTrees seeds): advance only if the
real block gains >0.30 point over the exact 694-feature incumbent, beats the
matched null by >0.20 point, and has paired exact McNemar p<0.05.

CONFIRM (partition 307, five folds, twenty ExtraTrees seeds): adopt only if the
frozen block again gains >0.20 point and p<0.05.  Confirmation is not run after
a failed screen.  Nothing in this file writes or replaces prediction.csv.

Usage:
    python3 notebooks/lib/iteration15_optimal_transport.py build
    python3 notebooks/lib/iteration15_optimal_transport.py screen
    python3 notebooks/lib/iteration15_optimal_transport.py confirm
    python3 notebooks/lib/iteration15_optimal_transport.py run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M


OUT = Path("outputs/iteration15")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "atlas_optimal_transport.npz"
SCREEN_RESULT = OUT / "optimal_transport_screen.csv"
BASE_CACHE = Path("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz")
COMP_CACHE = Path("outputs/iteration9/atlas_composition_cache.npz")
NICHE_CACHE = Path("outputs/iteration8/atlas_niche.npz")
ATLAS_ET_CACHE = Path("outputs/iteration9/atlas_et_block.npz")

ALPHA = 0.45
ENTROPY = 0.50
MAX_SINKHORN_ITER = 1000
SINKHORN_TOL = 2e-7
SCREEN_PARTITION = 271
CONFIRM_PARTITION = 307
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))
MASK_COLS = ("Region", "Excitatory_vs_Inhibitory", "Segment")


def sinkhorn_plan(cost: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Balanced entropic transport with uniform source mass and target capacity.

    Costs are section-normalised before this function is called.  Clipping the
    Gibbs kernel away from zero keeps ordinary Sinkhorn scaling stable and is
    substantially faster than a log-domain implementation for 108 small plans.
    """
    n_source, n_target = cost.shape
    a = np.full(n_source, 1.0 / n_source, dtype=np.float64)
    b = np.full(n_target, 1.0 / n_target, dtype=np.float64)
    kernel = np.exp(-np.minimum(cost, 20.0) / ENTROPY)
    kernel = np.maximum(kernel, 1e-18)
    u = np.ones(n_source, dtype=np.float64)
    v = np.ones(n_target, dtype=np.float64)
    error = np.inf

    for iteration in range(1, MAX_SINKHORN_ITER + 1):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
        if iteration % 10 == 0 or iteration == MAX_SINKHORN_ITER:
            plan = (u[:, None] * kernel) * v[None, :]
            row_error = np.max(np.abs(plan.sum(1) - a))
            col_error = np.max(np.abs(plan.sum(0) - b))
            error = float(max(row_error, col_error))
            if error < SINKHORN_TOL:
                break
    plan = (u[:, None] * kernel) * v[None, :]
    return plan, iteration, error


def build_transport(meta_all: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build real and within-section label-permuted transport posteriors."""
    _, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    class_index = {label: i for i, label in enumerate(classes)}
    other = len(classes)
    donor_codes = np.asarray([class_index.get(label, other) for label in labels[donors]])
    donor_sections = sections[donors]
    query_sections = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    real = np.zeros((len(meta_all), other + 1), dtype=np.float32)
    null = np.zeros_like(real)
    rng = np.random.default_rng(20260820)
    diagnostics: list[dict] = []

    for number, section in enumerate(np.unique(query_sections), 1):
        qrows = np.flatnonzero(query_sections == section)
        local = np.flatnonzero(donor_sections == section)
        if len(qrows) == 0 or len(local) == 0:
            continue
        source_rows = donors[local]
        source_xy = np.column_stack([ax[source_rows], ay[source_rows]])

        squared_distance = cdist(source_xy, query_xy[qrows], metric="sqeuclidean")
        # The median distance from an atlas cell to its nearest retained query is
        # the natural scale of this particular section's subsampling density.
        scale = float(np.median(np.min(squared_distance, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            positive = squared_distance[squared_distance > 0]
            scale = float(np.median(positive)) if len(positive) else 1.0
        cost = squared_distance / max(scale, 1e-12)
        plan, iterations, marginal_error = sinkhorn_plan(cost)
        target_mass = np.maximum(plan.sum(axis=0), 1e-300)

        codes = donor_codes[local]
        permuted_codes = codes[rng.permutation(len(codes))]
        for code in range(other + 1):
            real[qrows, code] = plan[codes == code].sum(axis=0) / target_mass
            null[qrows, code] = plan[permuted_codes == code].sum(axis=0) / target_mass

        diagnostics.append({
            "section": section,
            "atlas_donors": len(local),
            "challenge_targets": len(qrows),
            "donors_per_target": len(local) / len(qrows),
            "distance_scale_sq": scale,
            "sinkhorn_iterations": iterations,
            "max_marginal_error": marginal_error,
        })
        if number % 12 == 0 or number == len(np.unique(query_sections)):
            print(f"transport sections {number:3d}/108", flush=True)

    if not np.allclose(real.sum(1), 1.0, atol=2e-5):
        raise AssertionError("real transport posteriors do not sum to one")
    if not np.allclose(null.sum(1), 1.0, atol=2e-5):
        raise AssertionError("null transport posteriors do not sum to one")
    return real, null, pd.DataFrame(diagnostics)


def load_or_build(meta_all: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        if list(cached["classes"].astype(str)) != classes:
            raise ValueError("cached class order does not match challenge labels")
        print(f"loaded {CACHE}: {cached['real'].shape}", flush=True)
        cached_iterations = int(cached["max_sinkhorn_iter"]) if "max_sinkhorn_iter" in cached else -1
        cached_tolerance = float(cached["sinkhorn_tol"]) if "sinkhorn_tol" in cached else np.inf
        if cached_iterations == MAX_SINKHORN_ITER and cached_tolerance == SINKHORN_TOL:
            return cached["real"], cached["null"]
        print("transport solver settings changed; rebuilding cache", flush=True)
    started = time.time()
    real, null, diagnostics = build_transport(meta_all, classes)
    np.savez_compressed(CACHE, real=real, null=null, classes=np.asarray(classes),
                        entropy=ENTROPY, max_sinkhorn_iter=MAX_SINKHORN_ITER,
                        sinkhorn_tol=SINKHORN_TOL)
    diagnostics.to_csv(OUT / "transport_diagnostics.csv", index=False)
    print(f"built {CACHE}: {real.shape} in {time.time()-started:.1f}s", flush=True)
    return real, null


def block_offsets() -> dict[str, tuple[int, int]]:
    """Column ranges of the adopted stack, derived from the caches rather than assumed.

    The BASE block widens whenever the metadata one-hot encoder sees new Mouse_ID or
    Section_ID levels, which happens as soon as the validation cohort replaces the test
    cohort.  Every consumer that slices the stack must therefore compute its offsets
    instead of hard-coding the released-cohort layout, or it will silently read the wrong
    columns.
    """
    base = np.load(BASE_CACHE, allow_pickle=True)
    widths = [("BASE", base["BASE_TR"].shape[1]), ("EXT", base["EXT_TR"].shape[1]),
              ("SPA", base["SPA_TR"].shape[1]), ("NIC", base["NIC_TR"].shape[1]),
              ("COMP", np.load(COMP_CACHE, allow_pickle=True)["k10"].shape[1]),
              ("ANIC", np.load(NICHE_CACHE, allow_pickle=True)["k50"].shape[1]),
              ("ATL", base["ATL_TR"].shape[1])]
    et = np.load(ATLAS_ET_CACHE, allow_pickle=True)
    widths += [("ATL_ET", et["ATL_ET_TR"].shape[1]), ("COARSE", et["COARSE_TR"].shape[1])]
    out, start = {}, 0
    for name, w in widths:
        out[name] = (start, start + w)
        start += w
    out["TOTAL"] = (0, start)
    return out


def load_incumbent() -> tuple[np.ndarray, np.ndarray]:
    """Load the exact adopted 694 columns; do not silently reconstruct variants.

    The train/test split point is taken from the released files rather than hard-coded,
    so the re-run required after the test set is replaced by the validation set works
    even if the validation cohort is not 5,000 cells.
    """
    base = np.load(BASE_CACHE, allow_pickle=True)
    comp = np.load(COMP_CACHE, allow_pickle=True)["k10"]
    niche = np.load(NICHE_CACHE, allow_pickle=True)["k50"]
    atlas_et = np.load(ATLAS_ET_CACHE, allow_pickle=True)
    n_tr = len(base["BASE_TR"])
    n_te = len(base["BASE_TE"])
    if len(comp) != n_tr + n_te or len(niche) != n_tr + n_te:
        raise ValueError(
            f"stale feature caches: base has {n_tr}+{n_te} rows but the atlas "
            f"composition/niche caches have {len(comp)}/{len(niche)}. Delete "
            f"outputs/ derived caches and rebuild (run_prediction.py does this).")
    train = np.hstack([
        base["BASE_TR"], base["EXT_TR"], base["SPA_TR"], base["NIC_TR"],
        comp[:n_tr], niche[:n_tr], base["ATL_TR"],
        atlas_et["ATL_ET_TR"], atlas_et["COARSE_TR"],
    ]).astype(np.float32)
    test = np.hstack([
        base["BASE_TE"], base["EXT_TE"], base["SPA_TE"], base["NIC_TE"],
        comp[n_tr:], niche[n_tr:], base["ATL_TE"],
        atlas_et["ATL_ET_TE"], atlas_et["COARSE_TE"],
    ]).astype(np.float32)
    # The width is 694 for the released cohort, but the metadata one-hot encoder widens
    # when the validation cohort introduces new Mouse_ID / Section_ID levels.  Only the
    # train/test agreement is structural.
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"incumbent train/test widths differ: {train.shape}, {test.shape}")
    return train, test


def compatibility_mask(meta_fit: pd.DataFrame, y_fit: np.ndarray,
                       meta_eval: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Fold-scoped hard constraints learned only from released training labels."""
    allow = np.ones((len(meta_eval), len(classes)), dtype=bool)
    for column in MASK_COLS:
        fit_values = meta_fit[column].astype(str).to_numpy()
        known = set(fit_values)
        allowed_values = [set(fit_values[y_fit == cls]) for cls in classes]
        for row, value in enumerate(meta_eval[column].astype(str).to_numpy()):
            if value in known:
                allow[row] &= np.asarray([value in values for values in allowed_values])
    allow[~allow.any(axis=1)] = True
    return allow


def oof_probabilities(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                      classes: list[str], partition: int,
                      seeds: tuple[int, ...]) -> np.ndarray:
    """One fold-isolated probability vector per challenge-training cell."""
    output = np.zeros((len(y), len(classes)), dtype=np.float32)
    folds = StratifiedKFold(5, shuffle=True, random_state=partition)
    for fold, (fit, valid) in enumerate(folds.split(x, y), 1):
        probabilities = np.zeros((len(valid), len(classes)), dtype=np.float32)
        for seed in seeds:
            model = ExtraTreesClassifier(random_state=seed, **M.ET_KWARGS).fit(x[fit], y[fit])
            probabilities += M.align_proba(model, x[valid], classes)
        probabilities /= len(seeds)
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[fit]), classes), ALPHA
        )
        allow = compatibility_mask(meta.iloc[fit], y[fit], meta.iloc[valid], classes)
        probabilities = np.where(allow, probabilities, 0.0)
        probabilities /= np.maximum(probabilities.sum(1, keepdims=True), 1e-12)
        output[valid] = probabilities
        print(f"    fold {fold}/5", flush=True)
    return output


def evaluate(mode: str, x_base: np.ndarray, real: np.ndarray, null: np.ndarray,
             y: np.ndarray, meta: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    configs = {
        "incumbent_694": x_base,
        "+ equal-capacity transport": np.hstack([x_base, real]).astype(np.float32),
        "+ shuffled-label transport (null)": np.hstack([x_base, null]).astype(np.float32),
    }
    class_array = np.asarray(classes)
    glia = meta["Region"].isna().to_numpy()
    probabilities: dict[str, np.ndarray] = {}
    print(f"mode={mode} partition={partition} estimator_seeds={len(seeds)}", flush=True)
    for name, features in configs.items():
        started = time.time()
        print(f"  {name}: {features.shape[1]} features", flush=True)
        probabilities[name] = oof_probabilities(
            features, y, meta, classes, partition, seeds
        )
        print(f"  completed in {time.time()-started:.1f}s", flush=True)

    base_correct = class_array[probabilities["incumbent_694"].argmax(1)] == y
    rows = []
    for name, probs in probabilities.items():
        correct = class_array[probs.argmax(1)] == y
        if name == "incumbent_694":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, table = M.paired_mcnemar(correct, base_correct)
            wins, losses = table[0][1], table[1][0]
        rows.append({
            "mode": mode,
            "partition": partition,
            "config": name,
            "accuracy": correct.mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "wins": wins,
            "losses": losses,
            "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"optimal_transport_{mode}.csv", index=False)
    np.savez_compressed(
        OUT / f"optimal_transport_{mode}_oof.npz",
        **{name.replace(" ", "_"): probs for name, probs in probabilities.items()},
        truth=y, classes=class_array, partition=partition,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    return result


def passed_screen(result: pd.DataFrame) -> bool:
    by_name = result.set_index("config")
    real = by_name.loc["+ equal-capacity transport"]
    null = by_name.loc["+ shuffled-label transport (null)"]
    return bool(real.gain_pt > 0.30 and
                real.gain_pt - null.gain_pt > 0.20 and
                real.mcnemar_p < 0.05)


def passed_confirm(result: pd.DataFrame) -> bool:
    row = result.set_index("config").loc["+ equal-capacity transport"]
    return bool(row.gain_pt > 0.20 and row.mcnemar_p < 0.05)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"build", "screen", "confirm", "run"}:
        raise SystemExit("mode must be build, screen, confirm, or run")
    counts_train, meta_train, _, meta_test = F.load_challenge()
    del counts_train
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    real, null = load_or_build(meta_all, classes)
    if mode == "build":
        return
    x_train, _ = load_incumbent()
    screen = None
    if mode in {"screen", "run"}:
        screen = evaluate("screen", x_train, real[:len(y)], null[:len(y)],
                          y, meta_train, classes)
        verdict = "ADVANCE" if passed_screen(screen) else "REJECT"
        print(f"SCREEN VERDICT: {verdict}", flush=True)
        with open(OUT / "decision.json", "w", encoding="utf-8") as handle:
            json.dump({"screen_passed": passed_screen(screen),
                       "confirm_passed": None}, handle, indent=2)
        if not passed_screen(screen):
            return
    if mode == "confirm" and not SCREEN_RESULT.exists():
        raise SystemExit("confirmation requires the saved screen result")
    if mode == "confirm":
        screen = pd.read_csv(SCREEN_RESULT)
        if not passed_screen(screen):
            raise SystemExit("screen gate failed; confirmation is intentionally blocked")
    confirm = evaluate("confirm", x_train, real[:len(y)], null[:len(y)],
                       y, meta_train, classes)
    verdict = "ADOPT CANDIDATE" if passed_confirm(confirm) else "REJECT"
    print(f"CONFIRM VERDICT: {verdict}", flush=True)
    with open(OUT / "decision.json", "w", encoding="utf-8") as handle:
        json.dump({"screen_passed": True,
                   "confirm_passed": passed_confirm(confirm)}, handle, indent=2)


if __name__ == "__main__":
    main()
