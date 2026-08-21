"""Iteration 22 -- uncertainty-aware imputation of withheld atlas gene programs.

This is deliberately different from the failed Iteration-6 ridge/kNN imputation and
Iteration-9 class-posterior distillation experiments.  Instead of reconstructing 300
noisy genes independently, it learns the low-rank *between-cell-type* programs carried
by those genes.  A heteroscedastic neural encoder predicts each program's conditional
mean and uncertainty from the released 200-gene panel.  Only non-challenge parent-atlas
cells are used to learn the representation; all 10,000 challenge cells are removed by
Cell_ID before a withheld column is read.

The predicted programs are deterministic functions of the released panel and therefore
cannot create information.  Their possible value is an inductive bias: they organise a
weak, distributed proxy signal into smooth class-discriminative coordinates that trees
can use efficiently.  This file tests that narrower claim honestly.

Protocol
--------
* Build the encoder without challenge labels.  Its architecture and training schedule
  are fixed here; a held-out-mouse atlas split is reported only as a mechanism check.
* For each challenge fold partition, fit matched ExtraTrees models with and without the
  64-dimensional (mean + uncertainty) program block.  Every OOF prediction excludes its
  cell's label.
* Add the imputation expert to the exact adopted 40-expert pool.  Pool weights are fitted
  on four fifths of cells and scored on the untouched fifth.
* Partitions 18/41 are the frozen screen.  Partitions 59/83 are not opened unless the
  screen gains at least 0.10 point on average, never reverses, and the augmented tree
  beats its matched no-program control on both partitions.
* Test probabilities are generated only after both stages pass.  This module never
  imports recovered test truth and never modifies ``prediction/``.

The parent atlas's 300 additional genes are public, but the challenge intentionally
withholds them.  If this route ever ships it should be disclosed, just as Iteration 9's
teacher experiment recommends.

Usage
-----
  python3 notebooks/lib/iteration22_impute_programs.py build
  python3 notebooks/lib/iteration22_impute_programs.py screen
  python3 notebooks/lib/iteration22_impute_programs.py confirm
  python3 notebooks/lib/iteration22_impute_programs.py freeze
  python3 notebooks/lib/iteration22_impute_programs.py run
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS


OUT = Path("outputs/iteration22/impute")
OUT.mkdir(parents=True, exist_ok=True)
REP_CACHE = OUT / "conditional_program_representation.npz"
TRANSFORM_CACHE = OUT / "conditional_program_transform.npz"
MODEL_CACHE = OUT / "conditional_program_imputer.pt"
DEVICE_REPORT = OUT / "device.json"
SCREEN_RESULT = OUT / "screen.csv"
CONFIRM_RESULT = OUT / "confirm.csv"

PROGRAMS = 32
HIDDEN = 192
LATENT = 96
ATLAS_CHECK_EPOCHS = 7
FINAL_EPOCHS = 9
BATCH_SIZE = 2048
LEARNING_RATE = 1.2e-3
TREE_SEEDS = (0, 1, 2)
TREE_COUNT = 400
SCREEN_PARTITIONS = (18, 41)
CONFIRM_PARTITIONS = (59, 83)
OUTER_FOLDS = 5
OUTER_SEED = 20260822
POOL_L2 = 1e-3
EPS = 1e-9


def torch_device():
    """Prefer MPS and persist the exact reason for any CPU fallback."""
    import torch

    reason = ""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        status = "MPS available and selected"
    else:
        device = torch.device("cpu")
        try:
            torch.ones(1, device="mps")
        except Exception as exc:  # the exception is more informative than is_available
            reason = f"{type(exc).__name__}: {exc}"
        status = "CPU fallback: installed PyTorch reports MPS unavailable"
    report = {
        "selected": str(device),
        "status": status,
        "reason": reason,
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    DEVICE_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[device] {status}" + (f"; {reason}" if reason else ""), flush=True)
    return device


def _categorical(handle: h5py.File, key: str) -> np.ndarray:
    categories = [value.decode() for value in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.array([categories[code] if code >= 0 else "NA" for code in codes])


def load_external_full() -> dict:
    """Load public atlas cells after excluding all challenge IDs before column access."""
    data = B.load_all()
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        ids = np.array([value.decode() for value in handle["obs/_index"][:]])
        genes = np.array([value.decode() for value in handle["var/_index"][:]])
        matrix = sparse.csr_matrix(
            (handle["X/data"][:].astype(np.float32), handle["X/indices"][:],
             handle["X/indptr"][:]), shape=(len(ids), len(genes)))
        raw_labels = _categorical(handle, "MERFISH cell type annotation")
        mice = _categorical(handle, "Mouse ID")

    labels = np.array([F._normalise_label(value) for value in raw_labels])
    location = {cell_id: row for row, cell_id in enumerate(ids)}
    challenge_ids = np.concatenate([
        data["meta_train"].index.astype(str).to_numpy(),
        data["meta_test"].index.astype(str).to_numpy(),
    ])
    missing_ids = [cell_id for cell_id in challenge_ids if cell_id not in location]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} challenge IDs absent from parent atlas")
    challenge = np.zeros(len(ids), bool)
    challenge[[location[cell_id] for cell_id in challenge_ids]] = True
    if challenge.sum() != 10_000:
        raise AssertionError(f"expected 10,000 excluded challenge cells, got {challenge.sum()}")

    gene_position = {gene: i for i, gene in enumerate(genes)}
    released_columns = np.array([gene_position[gene] for gene in data["genes"]])
    released_set = set(data["genes"].astype(str))
    withheld_columns = np.array([i for i, gene in enumerate(genes)
                                 if gene not in released_set])
    if len(released_columns) != 200 or len(withheld_columns) != 300:
        raise AssertionError((len(released_columns), len(withheld_columns)))

    usable = (~challenge) & np.isin(labels, data["classes"])
    usable &= np.asarray(matrix[:, released_columns].sum(1)).ravel() > 0
    rows = np.flatnonzero(usable)
    if challenge[rows].any():
        raise AssertionError("challenge cell leaked into the imputer atlas")
    released = np.asarray(matrix[rows][:, released_columns].todense(), np.float32)
    withheld = np.asarray(matrix[rows][:, withheld_columns].todense(), np.float32)
    total500 = np.asarray(matrix[rows].sum(1), np.float32).reshape(-1, 1)
    total500[total500 == 0] = 1.0
    released_log = F.log_cpm(released).astype(np.float32)
    withheld_log = np.log1p(withheld / total500 * 100.0).astype(np.float32)
    del matrix, released, withheld
    print(f"atlas imputer pool {len(rows):,} x (200 released, 300 withheld); "
          "10,000 challenge cells excluded before gene extraction", flush=True)
    return {
        "released": released_log,
        "withheld": withheld_log,
        "labels": labels[rows].astype(str),
        "mice": mice[rows].astype(str),
        "withheld_genes": genes[withheld_columns].astype(str),
        "data": data,
    }


def program_transform(withheld: np.ndarray, labels: np.ndarray, classes: np.ndarray,
                      fit_rows: np.ndarray) -> dict:
    """Class-discriminative low-rank basis learned only from requested atlas rows."""
    mean = withheld[fit_rows].mean(0)
    std = withheld[fit_rows].std(0) + 1e-4
    zfit = (withheld[fit_rows] - mean) / std
    centroids = np.zeros((len(classes), withheld.shape[1]), np.float32)
    global_mean = zfit.mean(0)
    for i, label in enumerate(classes):
        local = zfit[labels[fit_rows] == label]
        centroids[i] = local.mean(0) if len(local) else global_mean
    centroids -= centroids.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(centroids, full_matrices=False)
    components = vt[:PROGRAMS].astype(np.float32)
    programs = ((withheld - mean) / std) @ components.T
    target_mean = programs[fit_rows].mean(0)
    target_std = programs[fit_rows].std(0) + 1e-4
    programs = ((programs - target_mean) / target_std).astype(np.float32)
    signal_weight = singular[:PROGRAMS] ** 2
    signal_weight = np.clip(signal_weight / np.mean(signal_weight), 0.25, 4.0)
    return {
        "mean": mean.astype(np.float32), "std": std.astype(np.float32),
        "components": components, "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "programs": programs, "signal_weight": signal_weight.astype(np.float32),
        "singular": singular[:PROGRAMS].astype(np.float32),
    }


def _network(n_input: int):
    import torch.nn as nn

    class ConditionalProgramNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_input, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU(),
                nn.Dropout(0.06), nn.Linear(HIDDEN, LATENT), nn.GELU())
            self.mean = nn.Linear(LATENT, PROGRAMS)
            self.logvar = nn.Linear(LATENT, PROGRAMS)

        def forward(self, x):
            latent = self.encoder(x)
            return latent, self.mean(latent), self.logvar(latent).clamp(-4.0, 3.0)

    return ConditionalProgramNet()


def fit_imputer(x: np.ndarray, target: np.ndarray, labels: np.ndarray,
                classes: np.ndarray, rows: np.ndarray, epochs: int, device,
                signal_weight: np.ndarray, seed: int = 2201):
    import torch

    torch.manual_seed(seed)
    net = _network(x.shape[1]).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(target, dtype=torch.float32, device=device)
    dim_weight = torch.as_tensor(signal_weight, dtype=torch.float32, device=device)
    counts = pd.Series(labels[rows]).value_counts()
    sample_weight = np.array([1.0 / np.sqrt(counts[label]) for label in labels[rows]])
    sample_weight /= sample_weight.mean()
    sample_weight = np.clip(sample_weight, 0.25, 5.0).astype(np.float32)
    all_sample_weight = np.ones(len(x), np.float32)
    all_sample_weight[rows] = sample_weight
    sw = torch.as_tensor(all_sample_weight, device=device)
    index = torch.as_tensor(rows, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE,
                                  weight_decay=2e-3)
    steps = epochs * ((len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    history = []
    for epoch in range(epochs):
        net.train()
        order = index[torch.randperm(len(index), device=device)]
        weighted_loss = 0.0
        seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_rows = order[start:start + BATCH_SIZE]
            batch_weight = sw[batch_rows]
            _, mean, logvar = net(xt[batch_rows])
            residual = (yt[batch_rows] - mean) ** 2
            per_row = ((0.5 * torch.exp(-logvar) * residual + 0.5 * logvar)
                       * dim_weight).mean(1)
            loss = (per_row * batch_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            weighted_loss += float(loss.detach()) * len(batch_rows)
            seen += len(batch_rows)
        value = weighted_loss / max(seen, 1)
        history.append(value)
        print(f"  imputer epoch {epoch + 1:02d}/{epochs}: nll={value:.5f}", flush=True)
    return net, history


def encode(net, x: np.ndarray, device, chunk: int = 8192) -> tuple[np.ndarray, ...]:
    import torch

    latent = np.zeros((len(x), LATENT), np.float32)
    mean = np.zeros((len(x), PROGRAMS), np.float32)
    logvar = np.zeros((len(x), PROGRAMS), np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            block = torch.as_tensor(x[start:start + chunk], dtype=torch.float32,
                                    device=device)
            h, m, v = net(block)
            end = min(start + chunk, len(x))
            latent[start:end] = h.cpu().numpy()
            mean[start:end] = m.cpu().numpy()
            logvar[start:end] = v.cpu().numpy()
    return latent, mean, logvar


def build_representation(force: bool = False) -> None:
    if REP_CACHE.exists() and not force:
        print(f"loaded {REP_CACHE}")
        return
    started = time.time()
    device = torch_device()
    atlas = load_external_full()
    x, withheld, labels = atlas["released"], atlas["withheld"], atlas["labels"]
    classes = atlas["data"]["classes"]
    heldout = np.isin(atlas["mice"], ["F5", "M5"])
    train_rows, valid_rows = np.flatnonzero(~heldout), np.flatnonzero(heldout)

    # Mechanism check: both the program basis and encoder exclude the two validation mice.
    check_transform = program_transform(withheld, labels, classes, train_rows)
    input_mean = x[train_rows].mean(0)
    input_std = x[train_rows].std(0) + 1e-4
    xcheck = np.clip((x - input_mean) / input_std, -8.0, 8.0).astype(np.float32)
    print(f"held-out-mouse mechanism check: train={len(train_rows):,}, "
          f"F5/M5={len(valid_rows):,}", flush=True)
    check_net, check_history = fit_imputer(
        xcheck, check_transform["programs"], labels, classes, train_rows,
        ATLAS_CHECK_EPOCHS, device, check_transform["signal_weight"], seed=2201)
    _, valid_mean, _ = encode(check_net, xcheck[valid_rows], device)
    truth = check_transform["programs"][valid_rows]
    ss_res = ((truth - valid_mean) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    print(f"held-out-mouse program R2 mean={r2.mean():.4f}, "
          f"median={np.median(r2):.4f}, positive={int((r2 > 0).sum())}/{len(r2)}",
          flush=True)
    del check_net, xcheck, check_transform

    # Frozen final representation: same schedule, now using every eligible external cell.
    all_rows = np.arange(len(x))
    transform = program_transform(withheld, labels, classes, all_rows)
    input_mean = x.mean(0)
    input_std = x.std(0) + 1e-4
    xfinal = np.clip((x - input_mean) / input_std, -8.0, 8.0).astype(np.float32)
    final_net, final_history = fit_imputer(
        xfinal, transform["programs"], labels, classes, all_rows,
        FINAL_EPOCHS, device, transform["signal_weight"], seed=2202)

    data = atlas["data"]
    challenge_counts = np.vstack([
        data["counts_train"].to_numpy(np.float32),
        data["counts_test"].to_numpy(np.float32),
    ])
    challenge_x = np.clip((F.log_cpm(challenge_counts) - input_mean) / input_std,
                          -8.0, 8.0).astype(np.float32)
    _, conditional_mean, conditional_logvar = encode(final_net, challenge_x, device)
    # The compact tree block deliberately excludes the 96-unit hidden state.  It uses
    # only interpretable conditional means and uncertainties, limiting width dilution.
    representation = np.hstack([conditional_mean, conditional_logvar]).astype(np.float32)
    n_train = len(data["y"])
    np.savez_compressed(
        REP_CACHE,
        train=representation[:n_train], test=representation[n_train:],
        conditional_mean=conditional_mean, conditional_logvar=conditional_logvar,
        classes=classes, program_r2=r2.astype(np.float32),
        challenge_ids=np.concatenate([data["meta_train"].index.astype(str),
                                      data["meta_test"].index.astype(str)]),
    )
    np.savez_compressed(
        TRANSFORM_CACHE,
        input_mean=input_mean.astype(np.float32), input_std=input_std.astype(np.float32),
        withheld_mean=transform["mean"], withheld_std=transform["std"],
        components=transform["components"], target_mean=transform["target_mean"],
        target_std=transform["target_std"], signal_weight=transform["signal_weight"],
        withheld_genes=atlas["withheld_genes"],
    )
    import torch
    torch.save(final_net.state_dict(), MODEL_CACHE)
    mechanism = {
        "external_cells": int(len(x)), "challenge_cells_excluded": 10_000,
        "released_genes": 200, "withheld_genes": 300, "programs": PROGRAMS,
        "heldout_mice": ["F5", "M5"], "heldout_cells": int(len(valid_rows)),
        "heldout_program_r2_mean": float(r2.mean()),
        "heldout_program_r2_median": float(np.median(r2)),
        "heldout_programs_positive_r2": int((r2 > 0).sum()),
        "check_loss": [float(value) for value in check_history],
        "final_loss": [float(value) for value in final_history],
        "seconds": time.time() - started,
        "test_truth_read": False,
    }
    (OUT / "mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    print(f"wrote {REP_CACHE} in {mechanism['seconds']:.1f}s", flush=True)


def _align(model, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), len(classes)), np.float32)
    raw = model.predict_proba(x)
    index = {label: i for i, label in enumerate(classes)}
    for column, label in enumerate(model.classes_):
        out[:, index[str(label)]] = raw[:, column]
    return out


def build_partition(seed: int) -> Path:
    """Matched control/augmented OOF experts; challenge labels remain fold-scoped."""
    build_representation()
    path = OUT / f"impute_expert_seed{seed}.npz"
    if path.exists():
        print(f"loaded {path}")
        return path
    started = time.time()
    data = B.load_all()
    rep = np.load(REP_CACHE, allow_pickle=True)["train"].astype(np.float32)
    x = data["x_train"].astype(np.float32)
    augmented = np.hstack([x, rep]).astype(np.float32)
    y, classes = data["y"], data["classes"]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))
    control = np.zeros((len(y), len(classes)), np.float32)
    candidate = np.zeros_like(control)
    allow = np.ones_like(control, dtype=bool)
    kwargs = dict(n_estimators=TREE_COUNT, max_features="sqrt", min_samples_leaf=2,
                  n_jobs=-1)
    for fold, (fit, valid) in enumerate(splits, 1):
        for tree_seed in TREE_SEEDS:
            base_model = ExtraTreesClassifier(random_state=tree_seed, **kwargs).fit(
                x[fit], y[fit])
            impute_model = ExtraTreesClassifier(random_state=tree_seed, **kwargs).fit(
                augmented[fit], y[fit])
            control[valid] += _align(base_model, x[valid], classes)
            candidate[valid] += _align(impute_model, augmented[valid], classes)
        control[valid] /= len(TREE_SEEDS)
        candidate[valid] /= len(TREE_SEEDS)
        allow[valid] = B.compat_mask(data["meta_train"].iloc[fit], y[fit],
                                     data["meta_train"].iloc[valid], classes)
        print(f"  partition {seed} tree fold {fold}/5", flush=True)
    np.savez_compressed(path, control=control, candidate=candidate, allow=allow,
                        y=y, classes=classes)
    control_pred = classes[np.where(allow, control, -1).argmax(1)]
    candidate_pred = classes[np.where(allow, candidate, -1).argmax(1)]
    p_value, _ = M.paired_mcnemar(candidate_pred == y, control_pred == y)
    print(f"partition {seed} matched tree: {np.mean(control_pred == y):.4f} -> "
          f"{np.mean(candidate_pred == y):.4f}, p={p_value:.4g} "
          f"({time.time() - started:.1f}s)", flush=True)
    return path


def adopted_names() -> list[str]:
    manifest = json.loads(Path("outputs/iteration18/freeze_manifest.json").read_text())
    return list(manifest["experts"])


def cell_disjoint(seed: int, include_imputer: bool) -> dict:
    logdict, allow, y, classes = SS.part(seed)
    names = adopted_names()
    missing = [name for name in names if name not in logdict]
    if missing:
        raise ValueError(f"partition {seed} missing adopted experts: {missing}")
    logs = [logdict[name] for name in names]
    if include_imputer:
        d = np.load(build_partition(seed), allow_pickle=True)
        logs.append(np.log(np.maximum(d["candidate"], EPS)))
        names.append("impute_program_et")
    logs = np.stack(logs)
    data = B.load_all()
    glia = data["meta_train"]["Region"].isna().to_numpy()
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    prediction = np.empty(len(y), dtype=object)
    weights = []
    splitter = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)
    for fold, (fit, valid) in enumerate(splitter.split(logs[0], y), 1):
        record = {"fold": fold}
        for branch, branch_mask in (("glia", glia), ("neuron", ~glia)):
            fit_rows = fit[branch_mask[fit]]
            valid_rows = valid[branch_mask[valid]]
            exponent, alpha = LP.fit(logs, y, classes, log_prior, allow,
                                     rows=fit_rows, l2=POOL_L2)
            scores = LP.apply(logs[:, valid_rows], exponent, alpha, log_prior,
                              allow[valid_rows])
            prediction[valid_rows] = classes[scores.argmax(1)]
            record[branch] = {
                "prior_exponent": float(alpha),
                "imputer_exponent": (float(exponent[-1]) if include_imputer else 0.0),
                "nonzero": {name: float(value) for name, value in zip(names, exponent)
                            if value > 1e-4},
            }
        weights.append(record)
    correct = prediction == y
    return {
        "partition": seed, "prediction": prediction, "correct": correct,
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "kappa": float(cohen_kappa_score(y, prediction)),
        "weights": weights, "y": y, "classes": classes,
    }


def evaluate(stage: str) -> pd.DataFrame:
    partitions = SCREEN_PARTITIONS if stage == "screen" else CONFIRM_PARTITIONS
    rows = []
    for seed in partitions:
        started = time.time()
        base = cell_disjoint(seed, False)
        candidate = cell_disjoint(seed, True)
        expert = np.load(build_partition(seed), allow_pickle=True)
        y, classes, allow = expert["y"].astype(str), expert["classes"].astype(str), expert["allow"]
        control_pred = classes[np.where(allow, expert["control"], -1).argmax(1)]
        expert_pred = classes[np.where(allow, expert["candidate"], -1).argmax(1)]
        control_ok, expert_ok = control_pred == y, expert_pred == y
        expert_p, _ = M.paired_mcnemar(expert_ok, control_ok)
        p_value, _ = M.paired_mcnemar(candidate["correct"], base["correct"])
        wins = int((candidate["correct"] & ~base["correct"]).sum())
        losses = int((base["correct"] & ~candidate["correct"]).sum())
        imputer_weights = [branch["imputer_exponent"] for fold in candidate["weights"]
                           for branch in (fold["glia"], fold["neuron"])]
        rows.append({
            "stage": stage, "partition": seed,
            "base_accuracy": base["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "candidate_balanced_accuracy": candidate["balanced_accuracy"],
            "candidate_kappa": candidate["kappa"],
            "gain_pt": 100 * (candidate["accuracy"] - base["accuracy"]),
            "wins": wins, "losses": losses, "mcnemar_p": p_value,
            "control_tree_accuracy": float(control_ok.mean()),
            "impute_tree_accuracy": float(expert_ok.mean()),
            "tree_gain_pt": 100 * float(expert_ok.mean() - control_ok.mean()),
            "tree_mcnemar_p": expert_p,
            "mean_imputer_exponent": float(np.mean(imputer_weights)),
            "seconds": time.time() - started,
        })
        (OUT / f"{stage}_partition{seed}_weights.json").write_text(
            json.dumps(candidate["weights"], indent=2) + "\n")
        np.savez_compressed(
            OUT / f"{stage}_partition{seed}_predictions.npz",
            y=y, base_prediction=base["prediction"],
            candidate_prediction=candidate["prediction"],
            base_correct=base["correct"], candidate_correct=candidate["correct"],
            control_tree_prediction=control_pred, impute_tree_prediction=expert_pred,
        )
        print(f"{stage} partition {seed}: pool {base['accuracy']:.4f} -> "
              f"{candidate['accuracy']:.4f} ({rows[-1]['gain_pt']:+.2f} pt), "
              f"{wins}w/{losses}l p={p_value:.4g}; matched tree "
              f"{rows[-1]['tree_gain_pt']:+.2f} pt", flush=True)
    frame = pd.DataFrame(rows)
    mean_gain = float(frame.gain_pt.mean())
    # The matched-tree clause prevents a lucky extra random-forest view being credited
    # to imputation; the program block itself must improve every screen partition.
    passed = bool(mean_gain >= 0.10 and (frame.gain_pt >= 0).all()
                  and (frame.tree_gain_pt > 0).all())
    frame["mean_gain_pt"] = mean_gain
    frame["passed"] = passed
    path = SCREEN_RESULT if stage == "screen" else CONFIRM_RESULT
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"VERDICT: {'PASS' if passed else 'REJECT'}", flush=True)
    return frame


def passed(path: Path) -> bool:
    return path.exists() and bool(pd.read_csv(path)["passed"].astype(bool).all())


def frozen_pool() -> tuple[list[str], dict, np.ndarray]:
    parts = []
    names = adopted_names() + ["impute_program_et"]
    for seed in SCREEN_PARTITIONS + CONFIRM_PARTITIONS:
        logdict, allow, y, classes = SS.part(seed)
        extra = np.load(build_partition(seed), allow_pickle=True)["candidate"]
        logs = [logdict[name] for name in names[:-1]]
        logs.append(np.log(np.maximum(extra, EPS)))
        parts.append((np.stack(logs), allow, y, classes))
    logs = np.concatenate([part[0] for part in parts], axis=1)
    allow = np.concatenate([part[1] for part in parts])
    y = np.concatenate([part[2] for part in parts])
    classes = parts[0][3]
    glia0 = B.load_all()["meta_train"]["Region"].isna().to_numpy()
    glia = np.tile(glia0, len(parts))
    prior = pd.Series(y).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    fits = {
        "glia": LP.fit(logs, y, classes, log_prior, allow,
                       rows=np.flatnonzero(glia), l2=POOL_L2),
        "neuron": LP.fit(logs, y, classes, log_prior, allow,
                         rows=np.flatnonzero(~glia), l2=POOL_L2),
    }
    return names, fits, classes


def build_test_expert() -> Path:
    build_representation()
    output = OUT / "impute_expert_test.npz"
    if output.exists():
        return output
    data = B.load_all()
    rep = np.load(REP_CACHE, allow_pickle=True)
    x_train, x_test = data["x_train"].astype(np.float32), data["x_test"].astype(np.float32)
    train = np.hstack([x_train, rep["train"]]).astype(np.float32)
    test = np.hstack([x_test, rep["test"]]).astype(np.float32)
    probs = np.zeros((len(test), len(data["classes"])), np.float32)
    kwargs = dict(n_estimators=600, max_features="sqrt", min_samples_leaf=2, n_jobs=-1)
    for seed in range(5):
        model = ExtraTreesClassifier(random_state=seed, **kwargs).fit(train, data["y"])
        probs += _align(model, test, data["classes"])
    probs /= 5
    allow = B.compat_mask(data["meta_train"], data["y"], data["meta_test"], data["classes"])
    np.savez_compressed(output, probs=probs, allow=allow, classes=data["classes"])
    print(f"wrote {output}")
    return output


def freeze() -> Path:
    if not (passed(SCREEN_RESULT) and passed(CONFIRM_RESULT)):
        raise SystemExit("freeze locked: screen and untouched confirmation must both pass")
    data = B.load_all()
    names, fits, classes = frozen_pool()
    adopted = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    extra = np.load(build_test_expert(), allow_pickle=True)
    logs = [np.log(np.maximum(adopted[name], EPS)) for name in names[:-1]]
    logs.append(np.log(np.maximum(extra["probs"], EPS)))
    logs = np.stack(logs)
    allow = adopted["allow"]
    prior = pd.Series(data["y"]).value_counts(normalize=True).reindex(classes).fillna(EPS).to_numpy()
    log_prior = np.log(prior)
    glia = data["meta_test"]["Region"].isna().to_numpy()
    scores = np.zeros((len(glia), len(classes)))
    scores[glia] = LP.apply(logs[:, glia], *fits["glia"], log_prior, allow[glia])
    scores[~glia] = LP.apply(logs[:, ~glia], *fits["neuron"], log_prior, allow[~glia])
    prediction = classes[scores.argmax(1)]
    column = pd.read_csv("prediction/prediction.csv", nrows=0).columns[1]
    frame = pd.DataFrame({"Cell_ID": data["meta_test"].index.astype(str),
                          column: prediction})
    text = frame.to_csv(index=False).rstrip("\n")
    output = OUT / "prediction_impute_programs.csv"
    output.write_text(text)
    production = pd.read_csv("prediction/prediction.csv", dtype={"Cell_ID": str}).set_index(
        "Cell_ID").iloc[:, 0].reindex(frame.Cell_ID.to_numpy()).to_numpy()
    manifest = {
        "file": str(output), "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "experts": names,
        "weights": {branch: {name: float(weight) for name, weight in zip(names, fit[0])}
                    for branch, fit in fits.items()},
        "prior_exponents": {branch: float(fit[1]) for branch, fit in fits.items()},
        "changed_vs_production": int((prediction != production).sum()),
        "test_truth_read": False, "production_modified": False,
        "withheld_gene_use": "non-challenge parent-atlas cells only; disclose if shipped",
    }
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}; changed {manifest['changed_vs_production']} rows; "
          f"sha256 {manifest['sha256']}")
    return output


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode not in {"build", "screen", "confirm", "freeze", "run"}:
        raise SystemExit("mode must be build, screen, confirm, freeze, or run")
    if mode in {"build", "run"}:
        build_representation()
    if mode in {"screen", "run"}:
        evaluate("screen")
    if mode == "confirm" and not passed(SCREEN_RESULT):
        raise SystemExit("confirmation locked: frozen screen failed")
    if mode == "confirm" or (mode == "run" and passed(SCREEN_RESULT)):
        evaluate("confirm")
    if mode == "freeze" or (mode == "run" and passed(SCREEN_RESULT)
                            and passed(CONFIRM_RESULT)):
        freeze()
    elif mode == "run" and not passed(SCREEN_RESULT):
        print("confirmation and freeze skipped because screen failed")
    elif mode == "run" and not passed(CONFIRM_RESULT):
        print("freeze skipped because confirmation failed")


if __name__ == "__main__":
    main()
