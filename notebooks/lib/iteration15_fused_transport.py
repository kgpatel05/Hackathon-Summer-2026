"""Iteration 15b - fused spatial/molecular atlas optimal transport.

The spatial-only equal-capacity transport screen was neutral (0.8032 ->
0.8036, 28 wins / 30 losses, p=0.896), showing that capacity constraints alone
mostly reproduce the adopted nearest-atlas composition block.  This frozen
follow-up tests the missing part of prototype alignment: donor/query similarity
in the released molecular space.

For each tissue section, the transport cost is the equal-weight mean of:

* squared spatial distance, normalised by the median atlas-to-nearest-query
  distance in that section; and
* cosine distance in a 20-dimensional donor-fitted truncated-SVD embedding of
  200-gene log-CPM, normalised in the same way.

The balanced entropic plan again gives every external atlas donor uniform mass
and every challenge cell equal capacity.  The output is its transported
61-class atlas posterior.  The SVD is fitted only on non-challenge atlas cells.
No challenge label, recovered test label, or withheld gene enters the feature.

This is one fixed candidate, not a weight sweep.  Its matched null keeps the
identical fused transport plan and permutes donor labels within Section_ID.

Pre-registered after the spatial-only result and before this candidate ran:

* screen: five-fold partition 353, five ExtraTrees seeds;
* advance only for >0.30 point over the 694-feature incumbent, >0.20 point over
  the null, and paired exact McNemar p<0.05;
* confirmation: partition 379, twenty ExtraTrees seeds, requiring >0.20 point
  and p<0.05.

Nothing here reads held-out truth or writes a submission.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration15_optimal_transport as OT


OUT = Path("outputs/iteration15")
CACHE = OUT / "atlas_fused_transport.npz"
SCREEN_PATH = OUT / "fused_transport_screen.csv"
N_COMPONENTS = 20
SCREEN_PARTITION = 353
CONFIRM_PARTITION = 379
SCREEN_SEEDS = tuple(range(5))
CONFIRM_SEEDS = tuple(range(20))


def sparse_log_cpm(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Sparse equivalent of the project's released-panel log-CPM transform."""
    output = matrix.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(output.sum(axis=1)).ravel()
    scale = np.divide(100.0, totals, out=np.ones_like(totals), where=totals > 0)
    output = sparse.diags(scale) @ output
    output.data = np.log1p(output.data)
    return output.tocsr()


def molecular_embedding(counts_train: pd.DataFrame, counts_test: pd.DataFrame,
                        donors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit released-gene SVD on external donors and transform donors + queries."""
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        atlas_genes = [gene.decode() for gene in handle["var/_index"][:]]
        lookup = {gene: i for i, gene in enumerate(atlas_genes)}
        columns = np.asarray([lookup[gene] for gene in counts_train.columns])
        full = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(handle["obs/_index"]), len(atlas_genes)),
        )
    donor_expression = sparse_log_cpm(full[donors][:, columns])
    query_expression = sparse_log_cpm(sparse.csr_matrix(np.vstack([
        counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32),
    ])))
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=20260820)
    donor_embedding = svd.fit_transform(donor_expression).astype(np.float32)
    query_embedding = svd.transform(query_expression).astype(np.float32)

    # Cosine distance avoids letting the norm of an already depth-normalised,
    # very sparse cell dominate its molecular match.
    donor_embedding /= np.maximum(np.linalg.norm(donor_embedding, axis=1, keepdims=True), 1e-8)
    query_embedding /= np.maximum(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-8)
    np.savez_compressed(
        OUT / "fused_transport_svd_audit.npz",
        explained_variance_ratio=svd.explained_variance_ratio_,
        singular_values=svd.singular_values_,
    )
    return donor_embedding, query_embedding


def build_fused(counts_train: pd.DataFrame, counts_test: pd.DataFrame,
                meta_all: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    _, labels, sections, ax, ay, donors = F._atlas_neighbour_setup(meta_all)
    donor_embedding, query_embedding = molecular_embedding(counts_train, counts_test, donors)
    class_index = {label: i for i, label in enumerate(classes)}
    other = len(classes)
    donor_codes = np.asarray([class_index.get(label, other) for label in labels[donors]])
    donor_sections = sections[donors]
    query_sections = meta_all["Section_ID"].astype(str).to_numpy()
    query_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    real = np.zeros((len(meta_all), other + 1), np.float32)
    null = np.zeros_like(real)
    rng = np.random.default_rng(20260820)
    diagnostics = []

    unique_sections = np.unique(query_sections)
    for number, section in enumerate(unique_sections, 1):
        qrows = np.flatnonzero(query_sections == section)
        local = np.flatnonzero(donor_sections == section)
        source_rows = donors[local]
        spatial = cdist(
            np.column_stack([ax[source_rows], ay[source_rows]]), query_xy[qrows],
            metric="sqeuclidean",
        )
        molecular = cdist(
            donor_embedding[local], query_embedding[qrows], metric="sqeuclidean"
        )

        spatial_scale = max(float(np.median(np.min(spatial, axis=1))), 1e-12)
        molecular_scale = max(float(np.median(np.min(molecular, axis=1))), 1e-12)
        fused_cost = 0.5 * spatial / spatial_scale + 0.5 * molecular / molecular_scale
        plan, iterations, marginal_error = OT.sinkhorn_plan(fused_cost)
        target_mass = np.maximum(plan.sum(0), 1e-300)
        codes = donor_codes[local]
        permuted = codes[rng.permutation(len(codes))]
        for code in range(other + 1):
            real[qrows, code] = plan[codes == code].sum(0) / target_mass
            null[qrows, code] = plan[permuted == code].sum(0) / target_mass
        diagnostics.append({
            "section": section,
            "atlas_donors": len(local),
            "challenge_targets": len(qrows),
            "spatial_scale_sq": spatial_scale,
            "molecular_scale_sq": molecular_scale,
            "sinkhorn_iterations": iterations,
            "max_marginal_error": marginal_error,
        })
        if number % 12 == 0 or number == len(unique_sections):
            print(f"fused transport sections {number:3d}/108", flush=True)

    if not np.allclose(real.sum(1), 1.0, atol=2e-5):
        raise AssertionError("fused transport posteriors do not sum to one")
    if not np.allclose(null.sum(1), 1.0, atol=2e-5):
        raise AssertionError("fused null posteriors do not sum to one")
    pd.DataFrame(diagnostics).to_csv(OUT / "fused_transport_diagnostics.csv", index=False)
    np.savez_compressed(CACHE, real=real, null=null, classes=np.asarray(classes),
                        n_components=N_COMPONENTS, spatial_weight=0.5,
                        molecular_weight=0.5, entropy=OT.ENTROPY,
                        max_sinkhorn_iter=OT.MAX_SINKHORN_ITER,
                        sinkhorn_tol=OT.SINKHORN_TOL)
    return real, null


def load_or_build(counts_train: pd.DataFrame, counts_test: pd.DataFrame,
                  meta_all: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        if list(cached["classes"].astype(str)) != classes:
            raise ValueError("cached class order does not match challenge labels")
        print(f"loaded {CACHE}: {cached['real'].shape}", flush=True)
        cached_iterations = int(cached["max_sinkhorn_iter"]) if "max_sinkhorn_iter" in cached else -1
        cached_tolerance = float(cached["sinkhorn_tol"]) if "sinkhorn_tol" in cached else np.inf
        if (cached_iterations == OT.MAX_SINKHORN_ITER and
                cached_tolerance == OT.SINKHORN_TOL):
            return cached["real"], cached["null"]
        print("transport solver settings changed; rebuilding cache", flush=True)
    started = time.time()
    real, null = build_fused(counts_train, counts_test, meta_all, classes)
    print(f"built {CACHE}: {real.shape} in {time.time()-started:.1f}s", flush=True)
    return real, null


def evaluate(mode: str, x_base: np.ndarray, real: np.ndarray, null: np.ndarray,
             y: np.ndarray, meta: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    partition = SCREEN_PARTITION if mode == "screen" else CONFIRM_PARTITION
    seeds = SCREEN_SEEDS if mode == "screen" else CONFIRM_SEEDS
    configs = {
        "incumbent_694": x_base,
        "+ fused spatial-molecular transport": np.hstack([x_base, real]).astype(np.float32),
        "+ shuffled-label fused transport (null)": np.hstack([x_base, null]).astype(np.float32),
    }
    class_array = np.asarray(classes)
    glia = meta["Region"].isna().to_numpy()
    probabilities = {}
    print(f"mode={mode} partition={partition} estimator_seeds={len(seeds)}", flush=True)
    for name, features in configs.items():
        started = time.time()
        print(f"  {name}: {features.shape[1]} features", flush=True)
        probabilities[name] = OT.oof_probabilities(
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
            "mode": mode, "partition": partition, "config": name,
            "accuracy": correct.mean(),
            "gain_pt": 100 * (correct.mean() - base_correct.mean()),
            "glia_accuracy": correct[glia].mean(),
            "neuron_accuracy": correct[~glia].mean(),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"fused_transport_{mode}.csv", index=False)
    np.savez_compressed(
        OUT / f"fused_transport_{mode}_oof.npz",
        **{name.replace(" ", "_"): probs for name, probs in probabilities.items()},
        truth=y, classes=class_array, partition=partition,
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    return result


def passes(result: pd.DataFrame, confirmation: bool) -> bool:
    by_name = result.set_index("config")
    real = by_name.loc["+ fused spatial-molecular transport"]
    if confirmation:
        return bool(real.gain_pt > 0.20 and real.mcnemar_p < 0.05)
    null = by_name.loc["+ shuffled-label fused transport (null)"]
    return bool(real.gain_pt > 0.30 and
                real.gain_pt - null.gain_pt > 0.20 and
                real.mcnemar_p < 0.05)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"build", "screen", "confirm", "run"}:
        raise SystemExit("mode must be build, screen, confirm, or run")
    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    real, null = load_or_build(counts_train, counts_test, meta_all, classes)
    if mode == "build":
        return
    x_train, _ = OT.load_incumbent()
    if mode in {"screen", "run"}:
        screen = evaluate("screen", x_train, real[:len(y)], null[:len(y)],
                          y, meta_train, classes)
        advanced = passes(screen, confirmation=False)
        print("SCREEN VERDICT: " + ("ADVANCE" if advanced else "REJECT"), flush=True)
        with open(OUT / "fused_transport_decision.json", "w", encoding="utf-8") as handle:
            json.dump({"screen_passed": advanced, "confirm_passed": None}, handle, indent=2)
        if not advanced:
            return
    if mode == "confirm":
        if not SCREEN_PATH.exists():
            raise SystemExit("confirmation requires the saved screen result")
        if not passes(pd.read_csv(SCREEN_PATH), confirmation=False):
            raise SystemExit("screen gate failed; confirmation is intentionally blocked")
    confirm = evaluate("confirm", x_train, real[:len(y)], null[:len(y)],
                       y, meta_train, classes)
    adopted = passes(confirm, confirmation=True)
    print("CONFIRM VERDICT: " + ("ADOPT CANDIDATE" if adopted else "REJECT"), flush=True)
    with open(OUT / "fused_transport_decision.json", "w", encoding="utf-8") as handle:
        json.dump({"screen_passed": True, "confirm_passed": adopted}, handle, indent=2)


if __name__ == "__main__":
    main()
