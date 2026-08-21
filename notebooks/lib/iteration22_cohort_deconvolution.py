"""Iteration 22 cohort track: empirical-Bayes soft-confusion deconvolution.

This is deliberately different from the earlier quota and section-prior experiments.
Those methods assume that the unlabelled cohort should copy a labelled composition.
Here the target composition is inferred from the target cohort's own *soft predictions*:

    mean_target(p_hat) = A.T @ pi_target,

where A[k, j] is the cross-fitted mean probability assigned to class j when the
truth is class k.  A non-negative ridge solve estimates ``pi_target``.  The estimate
is shrunk toward the matching labelled mouse/section composition and only changes the
relative posterior within the glial branch, where nearly all remaining errors live.

Honesty protocol
----------------
The 5,000 released training cells are split once into a 2,500-cell development cohort
and a 2,500-cell untouched confirmation cohort.  Candidate settings are selected using
an internal 1,250/1,250 split of development.  Confirmation fits every parameter on the
2,500 development cells and scores only the untouched 2,500.  Expert predictions are
already fold-isolated, and log-pool weights are re-fit on the labelled source rows.

The matched null permutes source labels within (mouse, branch), preserving mouse class
composition, class imbalance, prediction matrix, feature set, and target cohort.  It
destroys only the prediction-to-truth confusion relationship used for deconvolution.
Recovered test labels are loaded only if the frozen survivor passes confirmation.

The calculation is small constrained linear algebra and L-BFGS; Torch would add device
transfer overhead, so CPU is the appropriate device.  No prediction artifact is written.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS
import iteration5_models as M


OUT = Path("outputs/iteration22/cohort")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-9
MASTER_SEED = 2201
INNER_SEED = 2237
NULL_SEED = 2297
SCREEN_VIEWS = (18, 41)
CONFIRM_VIEWS = (59, 83)
EXCLUDED = {"rank", "atlaslam_proto", "etaug4_0.08", "xgbaug4"}


@dataclass(frozen=True)
class Config:
    name: str
    group: str
    confusion: str
    ridge: float
    source_tau: float
    confusion_tau: float
    beta: float


# A small, pre-named mechanism grid.  It varies only the granularity and shrinkage of
# one estimator; it is not a generic hyperparameter search over the confirmation set.
CONFIGS = (
    Config("global_soft_r1", "global", "soft", 1.0, 40.0, 4.0, 0.50),
    Config("mouse_soft_r1", "mouse", "soft", 1.0, 40.0, 4.0, 0.50),
    Config("mouse_soft_r03", "mouse", "soft", 0.3, 40.0, 4.0, 0.50),
    Config("mouse_hard_r1", "mouse", "hard", 1.0, 40.0, 4.0, 0.50),
    Config("section_soft_r3", "section", "soft", 3.0, 60.0, 6.0, 0.35),
    Config("section_soft_r1", "section", "soft", 1.0, 60.0, 6.0, 0.35),
)


def available_experts() -> list[str]:
    common = set(SS.part(SCREEN_VIEWS[0])[0])
    for seed in SCREEN_VIEWS[1:] + CONFIRM_VIEWS:
        common &= set(SS.part(seed)[0])
    return sorted(common - EXCLUDED)


def role_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_rows = np.arange(n)
    development, confirmation = next(
        KFold(2, shuffle=True, random_state=MASTER_SEED).split(all_rows)
    )
    source_local, screen_local = next(
        KFold(2, shuffle=True, random_state=INNER_SEED).split(development)
    )
    source = development[source_local]
    screen = development[screen_local]
    return source, screen, development, confirmation


def pooled_probabilities(
    fit_rows: np.ndarray,
    eval_rows: np.ndarray,
    views: tuple[int, ...],
    names: list[str],
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit pool weights on fit_rows, then average frozen-view probabilities."""
    outputs = []
    classes = SS.part(views[0])[3]
    y = SS.part(views[0])[2]
    prior = (
        pd.Series(y[fit_rows]).value_counts(normalize=True).reindex(classes)
        .fillna(EPS).to_numpy()
    )
    log_prior = np.log(prior)
    glia = meta["Region"].isna().to_numpy()
    for seed in views:
        lgd, allow, yy, cls = SS.part(seed)
        assert np.array_equal(cls, classes) and np.array_equal(yy, y)
        logs = np.stack([lgd[name] for name in names])
        z = np.full((len(eval_rows), len(classes)), -1e9, dtype=np.float64)
        for branch_mask in (glia, ~glia):
            fit_branch = fit_rows[branch_mask[fit_rows]]
            eval_branch_local = np.flatnonzero(branch_mask[eval_rows])
            if not len(eval_branch_local):
                continue
            eval_branch = eval_rows[eval_branch_local]
            w, a = LP.fit(
                logs, y, classes, log_prior, allow, rows=fit_branch, l2=1e-3
            )
            z[eval_branch_local] = LP.apply(
                logs[:, eval_branch], w, a, log_prior, allow[eval_branch]
            )
        outputs.append(softmax(z, axis=1))
    probabilities = np.mean(outputs, axis=0)
    probabilities /= np.maximum(probabilities.sum(1, keepdims=True), EPS)
    return probabilities.astype(np.float64), classes


def permute_within_mouse_branch(
    labels: np.ndarray, meta: pd.DataFrame, seed: int
) -> np.ndarray:
    out = labels.copy()
    mouse = meta["Mouse_ID"].astype(str).to_numpy()
    glia = meta["Region"].isna().to_numpy()
    rng = np.random.default_rng(seed)
    for m in np.unique(mouse):
        for branch in (False, True):
            rows = np.flatnonzero((mouse == m) & (glia == branch))
            out[rows] = out[rows][rng.permutation(len(rows))]
    return out


def confusion_matrix(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_idx: np.ndarray,
    classes: np.ndarray,
    mode: str,
    tau: float,
) -> np.ndarray:
    """Rows are truth, columns are predicted probability/hard call."""
    local = probabilities[:, class_idx]
    local /= np.maximum(local.sum(1, keepdims=True), EPS)
    k = len(class_idx)
    result = np.zeros((k, k), dtype=np.float64)
    global_to_local = {int(g): j for j, g in enumerate(class_idx)}
    class_to_global = {str(c): j for j, c in enumerate(classes)}
    if mode == "hard":
        observed = np.eye(k)[local.argmax(1)]
    else:
        observed = local
    for label, global_j in class_to_global.items():
        if global_j not in global_to_local:
            continue
        row = global_to_local[global_j]
        take = labels == label
        n = int(take.sum())
        # A weak identity pseudo-count stabilises rare classes without asserting that
        # the model is perfect.  It is applied identically to the real and null arms.
        result[row] = (observed[take].sum(0) + tau * np.eye(k)[row]) / (n + tau)
    result /= np.maximum(result.sum(1, keepdims=True), EPS)
    return result


def solve_prior(
    matrix: np.ndarray, observed: np.ndarray, prior: np.ndarray, ridge: float
) -> np.ndarray:
    """Non-negative ridge BBSE solve, followed by simplex normalisation."""
    k = len(prior)
    design = np.vstack(
        [matrix.T, np.sqrt(ridge) * np.eye(k), 5.0 * np.ones((1, k))]
    )
    target = np.concatenate(
        [observed, np.sqrt(ridge) * prior, np.array([5.0])]
    )
    estimate, _ = nnls(design, target)
    if estimate.sum() <= 0:
        return prior
    return estimate / estimate.sum()


def group_keys(meta: pd.DataFrame, group: str) -> np.ndarray:
    if group == "global":
        return np.repeat("all", len(meta)).astype(object)
    if group == "mouse":
        return meta["Mouse_ID"].astype(str).to_numpy()
    if group == "section":
        return meta["Section_ID"].astype(str).to_numpy()
    raise ValueError(group)


def adjust(
    fit_prob: np.ndarray,
    fit_y: np.ndarray,
    fit_meta: pd.DataFrame,
    eval_prob: np.ndarray,
    eval_meta: pd.DataFrame,
    classes: np.ndarray,
    cfg: Config,
    *,
    null: bool = False,
) -> np.ndarray:
    """Adjust glial posteriors; neurons are an exact no-op."""
    source_y = (
        permute_within_mouse_branch(fit_y, fit_meta, NULL_SEED)
        if null else fit_y
    )
    out = eval_prob.copy()
    fit_glia = fit_meta["Region"].isna().to_numpy()
    eval_glia = eval_meta["Region"].isna().to_numpy()
    glia_labels = np.unique(fit_y[fit_glia])
    class_lookup = {str(c): j for j, c in enumerate(classes)}
    class_idx = np.asarray([class_lookup[str(c)] for c in glia_labels], dtype=int)

    global_counts = (
        pd.Series(source_y[fit_glia]).value_counts().reindex(glia_labels)
        .fillna(0).to_numpy(float) + 0.5
    )
    global_prior = global_counts / global_counts.sum()
    matrix = confusion_matrix(
        fit_prob[fit_glia], source_y[fit_glia], class_idx, classes,
        cfg.confusion, cfg.confusion_tau,
    )
    fit_key = group_keys(fit_meta, cfg.group)
    eval_key = group_keys(eval_meta, cfg.group)

    for key in np.unique(eval_key[eval_glia]):
        erows = np.flatnonzero(eval_glia & (eval_key == key))
        frows = np.flatnonzero(fit_glia & (fit_key == key))
        if not len(erows):
            continue
        counts = (
            pd.Series(source_y[frows]).value_counts().reindex(glia_labels)
            .fillna(0).to_numpy(float)
        )
        source_prior = (
            counts + cfg.source_tau * global_prior
        ) / (counts.sum() + cfg.source_tau)
        local = eval_prob[erows][:, class_idx]
        local /= np.maximum(local.sum(1, keepdims=True), EPS)
        if cfg.confusion == "hard":
            observed = np.bincount(local.argmax(1), minlength=len(class_idx)).astype(float)
            observed /= observed.sum()
        else:
            observed = local.mean(0)
        target_prior = solve_prior(matrix, observed, source_prior, cfg.ridge)

        # Correct relative to the cohort marginal the model currently expresses, not
        # relative to the global training prior.  This avoids double-counting mouse and
        # section metadata already present in the experts.
        current = np.maximum(observed, 1e-5)
        ratio = np.power(np.maximum(target_prior, 1e-5) / current, cfg.beta)
        adjusted = local * ratio[None, :]
        adjusted /= np.maximum(adjusted.sum(1, keepdims=True), EPS)
        branch_mass = eval_prob[erows][:, class_idx].sum(1, keepdims=True)
        out[np.ix_(erows, class_idx)] = adjusted * branch_mass
    out /= np.maximum(out.sum(1, keepdims=True), EPS)
    return out


def metrics(
    name: str,
    probabilities: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    classes: np.ndarray,
    glia: np.ndarray,
    stage: str,
) -> dict:
    pred = classes[probabilities.argmax(1)]
    base_pred = classes[baseline.argmax(1)]
    ok = pred == truth
    base_ok = base_pred == truth
    if name == "baseline":
        p_value, wins, losses = 1.0, 0, 0
    else:
        p_value, table = M.paired_mcnemar(ok, base_ok)
        wins, losses = int(table[0][1]), int(table[1][0])
    return {
        "stage": stage,
        "config": name,
        "accuracy": float(ok.mean()),
        "gain_pt": float(100 * (ok.mean() - base_ok.mean())),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "kappa": float(cohen_kappa_score(truth, pred)),
        "glia_accuracy": float(ok[glia].mean()),
        "neuron_accuracy": float(ok[~glia].mean()),
        "changed": int((pred != base_pred).sum()),
        "wins": wins,
        "losses": losses,
        "mcnemar_p": float(p_value),
    }


def evaluate_stage(
    fit_rows: np.ndarray,
    eval_rows: np.ndarray,
    views: tuple[int, ...],
    configs: tuple[Config, ...],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    data = B.load_all()
    y, meta = data["y"], data["meta_train"]
    names = available_experts()
    # Pool probabilities for both sides are fit from the same labelled source.  Source
    # expert outputs remain fold-isolated, so their confusion is not in-sample.
    fit_prob, classes = pooled_probabilities(fit_rows, fit_rows, views, names, meta)
    eval_prob, _ = pooled_probabilities(fit_rows, eval_rows, views, names, meta)
    results = [
        metrics(
            "baseline", eval_prob, eval_prob, y[eval_rows], classes,
            meta.iloc[eval_rows]["Region"].isna().to_numpy(), stage,
        )
    ]
    matrices = {"baseline": eval_prob}
    for cfg in configs:
        real = adjust(
            fit_prob, y[fit_rows], meta.iloc[fit_rows], eval_prob,
            meta.iloc[eval_rows], classes, cfg,
        )
        null = adjust(
            fit_prob, y[fit_rows], meta.iloc[fit_rows], eval_prob,
            meta.iloc[eval_rows], classes, cfg, null=True,
        )
        matrices[cfg.name] = real
        matrices[cfg.name + "__null"] = null
        glia = meta.iloc[eval_rows]["Region"].isna().to_numpy()
        results.append(metrics(cfg.name, real, eval_prob, y[eval_rows], classes, glia, stage))
        results.append(metrics(cfg.name + "__null", null, eval_prob, y[eval_rows], classes, glia, stage))
    return pd.DataFrame(results), matrices, classes


def write_readme(
    screen: pd.DataFrame,
    confirm: pd.DataFrame | None,
    selected: Config,
    passed: bool,
    test_metrics: dict | None,
) -> None:
    s = screen.set_index("config")
    lines = [
        "# Iteration 22 cohort track — soft-confusion deconvolution",
        "",
        "This track tested empirical-Bayes estimation of the unlabelled cohort's class",
        "composition from its own soft predictions. It is not a quota copied from train:",
        "the source labels estimate the classifier confusion operator and provide only a",
        "shrinkage prior. All selection used released training labels in cell-disjoint",
        "development/confirmation cohorts.",
        "",
        "## Screen",
        "",
        "| candidate | accuracy | gain (pt) | balanced | kappa | wins/losses | p | null gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in CONFIGS:
        r, n = s.loc[cfg.name], s.loc[cfg.name + "__null"]
        lines.append(
            f"| {cfg.name} | {r.accuracy:.4f} | {r.gain_pt:+.2f} | "
            f"{r.balanced_accuracy:.4f} | {r.kappa:.4f} | "
            f"{int(r.wins)}/{int(r.losses)} | {r.mcnemar_p:.4g} | {n.gain_pt:+.2f} |"
        )
    lines += ["", f"Frozen survivor: `{selected.name}`.", ""]
    if confirm is not None:
        c = confirm.set_index("config")
        r, n = c.loc[selected.name], c.loc[selected.name + "__null"]
        lines += [
            "## Untouched confirmation",
            "",
            f"Baseline {c.loc['baseline'].accuracy:.4f}; candidate {r.accuracy:.4f} "
            f"({r.gain_pt:+.2f} pt), balanced {r.balanced_accuracy:.4f}, kappa "
            f"{r.kappa:.4f}, {int(r.wins)} wins / {int(r.losses)} losses, "
            f"McNemar p={r.mcnemar_p:.4g}. Matched-null gain {n.gain_pt:+.2f} pt.",
            "",
            "Verdict: **" + ("passed" if passed else "rejected") + "**.",
            "",
        ]
    if test_metrics is not None:
        lines += [
            "## One-way test thermometer",
            "",
            f"Accuracy {test_metrics['accuracy']:.4f}, balanced accuracy "
            f"{test_metrics['balanced_accuracy']:.4f}, kappa "
            f"{test_metrics['kappa']:.4f}, {test_metrics['wins']} wins / "
            f"{test_metrics['losses']} losses versus the 0.8120 production calls.",
            "",
        ]
    lines += [
        "## Device",
        "",
        "CPU. The method is SciPy NNLS plus small L-BFGS pooling problems; it contains no",
        "dense neural training and would be slower after moving these small arrays to MPS.",
        "",
        "No production prediction, root README, or shared scorecard was modified.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def production_test(selected: Config) -> dict:
    """One-way thermometer, called only after confirmation passes."""
    from evaluate import load_truth

    data = B.load_all()
    y, classes, meta = data["y"], data["classes"], data["meta_train"]
    names = available_experts()
    # Honest cross-fitted source probabilities: each row is pooled with weights fitted
    # on other cells.  Averaging four OOF views stabilises the confusion estimate.
    source = np.zeros((len(y), len(classes)), dtype=float)
    for fit, val in KFold(5, shuffle=True, random_state=2269).split(y):
        p, _ = pooled_probabilities(fit, val, SCREEN_VIEWS + CONFIRM_VIEWS, names, meta)
        source[val] = p

    manifest = json.loads((B.OUT / "freeze_manifest.json").read_text())
    used = manifest["experts"]
    expert_test = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    logs = np.stack([np.log(np.maximum(expert_test[n], EPS)) for n in used])
    allow = expert_test["allow"]
    prior = (
        pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    )
    lp = np.log(prior)
    test_prob = np.zeros((len(data["meta_test"]), len(classes)), float)
    glia = data["meta_test"]["Region"].isna().to_numpy()
    for tag, mask in (("glia", glia), ("neuron", ~glia)):
        w = np.asarray(manifest["exponents"][tag])
        a = float(manifest["prior_exponent"][tag])
        test_prob[mask] = softmax(LP.apply(logs[:, mask], w, a, lp, allow[mask]), axis=1)
    candidate = adjust(
        source, y, meta, test_prob, data["meta_test"], classes, selected,
    )
    truth = load_truth().reindex(data["meta_test"].index.astype(str)).to_numpy()
    production = pd.read_csv(
        "outputs/iteration18/predictions/prediction_iteration18_it21.csv",
        dtype={"Cell_ID": str},
    ).set_index("Cell_ID").iloc[:, 0].reindex(data["meta_test"].index.astype(str)).to_numpy()
    pred = classes[candidate.argmax(1)]
    ok, base_ok = pred == truth, production == truth
    p_value, table = M.paired_mcnemar(ok, base_ok)
    return {
        "accuracy": float(ok.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "kappa": float(cohen_kappa_score(truth, pred)),
        "glia_accuracy": float(ok[glia].mean()),
        "neuron_accuracy": float(ok[~glia].mean()),
        "changed": int((pred != production).sum()),
        "wins": int(table[0][1]),
        "losses": int(table[1][0]),
        "mcnemar_p": float(p_value),
    }


def main() -> None:
    data = B.load_all()
    source, screen_rows, development, confirmation = role_split(len(data["y"]))
    np.savez_compressed(
        OUT / "frozen_roles.npz", source=source, screen=screen_rows,
        development=development, confirmation=confirmation,
    )
    print(
        f"device=cpu source={len(source)} screen={len(screen_rows)} "
        f"development={len(development)} confirmation={len(confirmation)} "
        f"experts={len(available_experts())}", flush=True,
    )
    screen, screen_matrices, classes = evaluate_stage(
        source, screen_rows, SCREEN_VIEWS, CONFIGS, "screen"
    )
    screen.to_csv(OUT / "screen.csv", index=False)
    np.savez_compressed(OUT / "screen_probabilities.npz", classes=classes, **screen_matrices)
    candidates = screen[~screen.config.str.endswith("__null") & (screen.config != "baseline")]
    # Stable tie-break: accuracy, then smaller beta/ridge departure is encoded by order.
    winner_name = str(candidates.sort_values(
        ["accuracy", "balanced_accuracy"], ascending=False, kind="stable"
    ).iloc[0].config)
    selected = next(cfg for cfg in CONFIGS if cfg.name == winner_name)
    (OUT / "frozen_config.json").write_text(json.dumps(asdict(selected), indent=2) + "\n")
    print("\nSCREEN\n" + screen.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nfrozen survivor: {selected.name}", flush=True)

    confirm, confirm_matrices, _ = evaluate_stage(
        development, confirmation, CONFIRM_VIEWS, (selected,), "confirm"
    )
    confirm.to_csv(OUT / "confirm.csv", index=False)
    np.savez_compressed(
        OUT / "confirm_probabilities.npz", classes=classes, **confirm_matrices
    )
    rows = confirm.set_index("config")
    real = rows.loc[selected.name]
    null = rows.loc[selected.name + "__null"]
    passed = bool(
        real.gain_pt > 0.20
        and real.mcnemar_p < 0.05
        and real.gain_pt - null.gain_pt > 0.15
    )
    decision = {
        "selected": asdict(selected),
        "confirmation_passed": passed,
        "gate": "gain>0.20pt, McNemar p<0.05, and >0.15pt over shuffled-confusion null",
        "test_truth_read": False,
        "device": "cpu",
    }
    test_result = None
    if passed:
        print("\nCONFIRMATION PASSED; reading the one-way test thermometer", flush=True)
        test_result = production_test(selected)
        decision["test_truth_read"] = True
        decision["test_metrics"] = test_result
        (OUT / "test_metrics.json").write_text(json.dumps(test_result, indent=2) + "\n")
    else:
        print("\nCONFIRMATION FAILED; test labels remain unread by this track", flush=True)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    write_readme(screen, confirm, selected, passed, test_result)
    print("\nCONFIRM\n" + confirm.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nVERDICT: " + ("PASS" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
