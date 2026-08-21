"""Iteration 22 MPS track: count-split denoising in Hellinger geometry.

Biological mechanism
--------------------
MERFISH counts are a shallow sample from a cell's latent transcript composition.  The
median challenge cell has only about 21 observed molecules, so ordinary masking removes
whole gene coordinates rather than reproducing the assay's sampling noise.  This model
draws two binomial thinnings of each cell, maps their empirical-Bayes compositions to the
Hellinger sphere, and asks one encoder to classify both views, reconstruct the unthinned
composition, and agree across thinnings.  A deterministic Hellinger MLP is the matched
ablation.  The denoised latent is also offered to ExtraTrees as a fold-scoped feature.

Protocol
--------
* Only released training labels and released 200-gene counts are read.
* Every neural and tree prediction used for scoring is outer-fold OOF.
* Screen: partition 18, nested pool split seed 2201.  Freeze exactly one arm.
* Confirmation: fresh outer partition 83, nested pool split seed 2297.
* A candidate is test-eligible only for >= +0.15 point, wins > losses, exact paired
  p < 0.05, and no nested-fold loss worse than -0.10 point on confirmation.
* This module cannot read recovered test truth or write a submission.

Apple Silicon uses ``torch.device('mps')`` whenever available.  CPU is an explicit,
reported fallback for restricted runners where the MPS device is hidden.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as TF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration18_subsets as SS
import iteration5_features as F
import iteration5_models as M


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iteration22/mps"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "outputs/iteration18/freeze_manifest.json"
EPS = 1e-8
SCREEN_PARTITION = 18
CONFIRM_PARTITION = 83
SCREEN_NESTED_SEED = 2201
CONFIRM_NESTED_SEED = 2297
META_CATEGORICAL = (
    "Datasets", "Region", "Excitatory_vs_Inhibitory", "Segment", "Gender",
    "Mouse_ID", "AP_position", "Section_ID",
)


@dataclass(frozen=True)
class Config:
    latent: int = 48
    hidden: int = 192
    batch_size: int = 384
    epochs: int = 52
    patience: int = 8
    lr: float = 1.8e-3
    weight_decay: float = 3e-3
    thinning_keep: float = 0.78
    dirichlet_strength: float = 5.0
    consistency_weight: float = 0.15
    reconstruction_weight: float = 1.0
    latent_agreement_weight: float = 0.10
    label_smoothing: float = 0.025
    outer_folds: int = 5
    tree_estimators: int = 450
    nested_folds: int = 5


CFG = Config()


def device() -> torch.device:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    return torch.device("cpu")


DEVICE = device()


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(seed)


def empirical_bayes_hellinger(
    counts: np.ndarray, gene_prior: np.ndarray, strength: float,
) -> np.ndarray:
    counts = np.asarray(counts, np.float32)
    total = counts.sum(1, keepdims=True)
    posterior = (counts + strength * gene_prior[None, :]) / np.maximum(
        total + strength, EPS
    )
    return np.sqrt(np.maximum(posterior, 0)).astype(np.float32)


def count_design(
    counts: np.ndarray, gene_prior: np.ndarray, strength: float,
) -> np.ndarray:
    h = empirical_bayes_hellinger(counts, gene_prior, strength)
    # Soft presence retains molecule multiplicity without letting large genes dominate.
    presence = counts / (counts + 1.0)
    return np.hstack([h, presence]).astype(np.float32)


def meta_design(
    meta_fit: pd.DataFrame, meta_eval: pd.DataFrame, counts_fit: np.ndarray,
    counts_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_cat = meta_fit.loc[:, META_CATEGORICAL].astype(str).fillna("missing")
    eval_cat = meta_eval.loc[:, META_CATEGORICAL].astype(str).fillna("missing")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    fit_onehot = enc.fit_transform(fit_cat)
    eval_onehot = enc.transform(eval_cat)

    def continuous(frame: pd.DataFrame, counts: np.ndarray) -> np.ndarray:
        section = frame["Section_ID"].astype(str)
        xy = frame[["center_x", "center_y"]].astype(float).copy()
        # Section-relative position prevents absolute slide coordinates becoming a batch ID.
        for col in ("center_x", "center_y"):
            mean = xy[col].groupby(section).transform("mean")
            std = xy[col].groupby(section).transform("std").fillna(1.0).clip(lower=1.0)
            xy[col] = (xy[col] - mean) / std
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0).to_numpy()
        total = counts.sum(1)
        detected = (counts > 0).sum(1)
        return np.column_stack([
            np.log1p(np.maximum(volume, 0)), np.log1p(total), detected,
            total / np.maximum(volume, 1.0), xy.to_numpy(),
        ]).astype(np.float32)

    fit_cont = continuous(meta_fit, counts_fit)
    eval_cont = continuous(meta_eval, counts_eval)
    sc = StandardScaler().fit(fit_cont)
    return (
        np.hstack([fit_onehot, sc.transform(fit_cont)]).astype(np.float32),
        np.hstack([eval_onehot, sc.transform(eval_cont)]).astype(np.float32),
    )


class CountSplitNet(nn.Module):
    def __init__(self, count_dim: int, meta_dim: int, n_classes: int) -> None:
        super().__init__()
        self.count_encoder = nn.Sequential(
            nn.Linear(count_dim, CFG.hidden), nn.LayerNorm(CFG.hidden), nn.GELU(),
            nn.Dropout(0.12), nn.Linear(CFG.hidden, CFG.latent),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.08),
        )
        self.fusion = nn.Sequential(
            nn.Linear(CFG.latent + 64, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.12),
        )
        self.classifier = nn.Linear(128, n_classes)
        self.decoder = nn.Sequential(
            nn.Linear(CFG.latent, 128), nn.GELU(), nn.Linear(128, count_dim // 2),
        )

    def forward(self, count_x: torch.Tensor, meta_x: torch.Tensor):
        latent = self.count_encoder(count_x)
        fused = self.fusion(torch.cat([latent, self.meta_encoder(meta_x)], 1))
        return self.classifier(fused), latent, self.decoder(latent)


def _numpy_thinning(
    counts: np.ndarray, rows: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    # Sampling on CPU avoids an unsupported aten::binomial fallback on older MPS builds;
    # all matrix multiplications, gradients and parameter updates remain on MPS.
    return rng.binomial(counts[rows].astype(np.int64), CFG.thinning_keep).astype(np.float32)


def fit_network(
    counts_fit: np.ndarray, meta_fit: pd.DataFrame, y_fit: np.ndarray,
    counts_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
    seed: int, denoise: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    seed_all(seed)
    ci = {c: i for i, c in enumerate(classes)}
    y_code = np.array([ci[v] for v in y_fit], np.int64)
    inner, monitor = train_test_split(
        np.arange(len(y_fit)), test_size=0.12, random_state=seed,
        stratify=y_code,
    )
    summed = counts_fit[inner].sum(0, dtype=np.float64) + 0.5
    gene_prior = (summed / summed.sum()).astype(np.float32)
    cx_fit = count_design(counts_fit, gene_prior, CFG.dirichlet_strength)
    cx_eval = count_design(counts_eval, gene_prior, CFG.dirichlet_strength)
    mx_fit, mx_eval = meta_design(meta_fit, meta_eval, counts_fit, counts_eval)
    target_h = empirical_bayes_hellinger(
        counts_fit, gene_prior, CFG.dirichlet_strength
    )

    net = CountSplitNet(cx_fit.shape[1], mx_fit.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epochs)
    freq = np.bincount(y_code[inner], minlength=len(classes)).astype(np.float32)
    class_weight = np.sqrt(freq.sum() / np.maximum(freq, 1.0))
    class_weight /= class_weight.mean()
    class_weight = np.clip(class_weight, 0.45, 3.0)
    class_weight_t = torch.tensor(class_weight, device=DEVICE)
    mx_fit_t = torch.tensor(mx_fit, device=DEVICE)
    cx_fit_t = torch.tensor(cx_fit, device=DEVICE)
    target_h_t = torch.tensor(target_h, device=DEVICE)
    y_t = torch.tensor(y_code, device=DEVICE)
    rng = np.random.default_rng(seed + 991)
    best_acc, best_state, bad, best_epoch = -1.0, None, 0, 0
    t0 = time.time()

    for epoch in range(CFG.epochs):
        net.train()
        order = rng.permutation(inner)
        for start in range(0, len(order), CFG.batch_size):
            rows = order[start:start + CFG.batch_size]
            if len(rows) < 4:
                continue
            rt = torch.tensor(rows, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            full_logits, full_z, full_recon = net(cx_fit_t[rt], mx_fit_t[rt])
            loss = TF.cross_entropy(
                full_logits, y_t[rt], weight=class_weight_t,
                label_smoothing=CFG.label_smoothing,
            )
            if denoise:
                thin1 = _numpy_thinning(counts_fit, rows, rng)
                thin2 = _numpy_thinning(counts_fit, rows, rng)
                v1 = torch.tensor(
                    count_design(thin1, gene_prior, CFG.dirichlet_strength),
                    device=DEVICE,
                )
                v2 = torch.tensor(
                    count_design(thin2, gene_prior, CFG.dirichlet_strength),
                    device=DEVICE,
                )
                l1, z1, r1 = net(v1, mx_fit_t[rt])
                l2, z2, r2 = net(v2, mx_fit_t[rt])
                supervised = 0.5 * (
                    TF.cross_entropy(l1, y_t[rt], weight=class_weight_t,
                                     label_smoothing=CFG.label_smoothing)
                    + TF.cross_entropy(l2, y_t[rt], weight=class_weight_t,
                                       label_smoothing=CFG.label_smoothing)
                )
                p1, p2 = TF.log_softmax(l1, 1), TF.log_softmax(l2, 1)
                q1, q2 = p1.exp(), p2.exp()
                js = 0.5 * (
                    TF.kl_div(p1, q2.detach(), reduction="batchmean")
                    + TF.kl_div(p2, q1.detach(), reduction="batchmean")
                )
                recon = TF.mse_loss(r1, target_h_t[rt]) + TF.mse_loss(
                    r2, target_h_t[rt]
                )
                agree = 1.0 - TF.cosine_similarity(z1, z2, dim=1).mean()
                loss = (0.5 * loss + 0.5 * supervised
                        + CFG.consistency_weight * js
                        + CFG.reconstruction_weight * recon
                        + CFG.latent_agreement_weight * agree)
            else:
                loss = loss + 0.25 * TF.mse_loss(full_recon, target_h_t[rt])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            mon = torch.tensor(monitor, device=DEVICE)
            acc = float((net(cx_fit_t[mon], mx_fit_t[mon])[0].argmax(1)
                         == y_t[mon]).float().mean().cpu())
        if acc > best_acc + 1e-6:
            best_acc, best_epoch, bad = acc, epoch + 1, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= CFG.patience:
                break

    if best_state is None:
        raise RuntimeError("inner monitor never produced a model state")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p_eval, z_eval, _ = net(
            torch.tensor(cx_eval, device=DEVICE), torch.tensor(mx_eval, device=DEVICE)
        )
        _, z_fit, _ = net(cx_fit_t, mx_fit_t)
        p_eval = TF.softmax(p_eval, 1).cpu().numpy().astype(np.float32)
        z_fit = z_fit.cpu().numpy().astype(np.float32)
        z_eval = z_eval.cpu().numpy().astype(np.float32)
    return p_eval, z_fit, z_eval, {
        "best_inner_accuracy": best_acc,
        "best_epoch": best_epoch,
        "seconds": time.time() - t0,
        "denoise": denoise,
    }


def latent_trees(
    x_fit: np.ndarray, latent_fit: np.ndarray, y_fit: np.ndarray,
    x_eval: np.ndarray, latent_eval: np.ndarray, classes: np.ndarray, seed: int,
) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=CFG.tree_estimators, max_features=0.08, min_samples_leaf=1,
        class_weight=None, n_jobs=-1, random_state=seed,
    )
    model.fit(np.hstack([x_fit, latent_fit]), y_fit)
    raw = model.predict_proba(np.hstack([x_eval, latent_eval]))
    out = np.zeros((len(x_eval), len(classes)), np.float32)
    pos = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, pos[str(c)]] = raw[:, j]
    prior = M.prior_vector(pd.Series(y_fit), list(classes))
    return M.correct_prior(out, prior, 0.45).astype(np.float32)


def build_oof(partition: int, arms: set[str]) -> tuple[dict[str, np.ndarray], dict]:
    data = B.load_all()
    counts = data["counts_train"].to_numpy(np.float32)
    y, classes, x = data["y"], data["classes"], data["x_train"]
    meta = data["meta_train"]
    requested_models = set()
    if "plain" in arms:
        requested_models.add("plain")
    if {"denoise", "denoise_latent_et"} & arms:
        requested_models.add("denoise")
    probabilities = {
        name: np.zeros((len(y), len(classes)), np.float32) for name in arms
    }
    diagnostics: dict[str, list] = {name: [] for name in requested_models}
    folds = StratifiedKFold(
        CFG.outer_folds, shuffle=True, random_state=partition
    )
    t0 = time.time()
    for fold, (fit, val) in enumerate(folds.split(counts, y), 1):
        print(f"outer fold {fold}/{CFG.outer_folds}", flush=True)
        if "plain" in requested_models:
            p, _, _, diag = fit_network(
                counts[fit], meta.iloc[fit], y[fit], counts[val], meta.iloc[val],
                classes, seed=partition * 100 + fold, denoise=False,
            )
            probabilities["plain"][val] = p
            diagnostics["plain"].append(diag)
            print(f"  plain inner={diag['best_inner_accuracy']:.4f} "
                  f"epoch={diag['best_epoch']} {diag['seconds']:.1f}s", flush=True)
        if "denoise" in requested_models:
            p, z_fit, z_val, diag = fit_network(
                counts[fit], meta.iloc[fit], y[fit], counts[val], meta.iloc[val],
                classes, seed=partition * 1000 + fold, denoise=True,
            )
            if "denoise" in probabilities:
                probabilities["denoise"][val] = p
            if "denoise_latent_et" in probabilities:
                probabilities["denoise_latent_et"][val] = latent_trees(
                    x[fit], z_fit, y[fit], x[val], z_val, classes,
                    seed=partition * 10 + fold,
                )
            diagnostics["denoise"].append(diag)
            print(f"  denoise inner={diag['best_inner_accuracy']:.4f} "
                  f"epoch={diag['best_epoch']} {diag['seconds']:.1f}s", flush=True)
    diagnostics["device"] = str(DEVICE)
    diagnostics["mps_built"] = bool(torch.backends.mps.is_built())
    diagnostics["mps_available"] = bool(torch.backends.mps.is_available())
    diagnostics["total_seconds"] = time.time() - t0
    path = OUT / f"countsplit_oof_partition{partition}.npz"
    np.savez_compressed(path, y=y, classes=classes, **probabilities)
    (OUT / f"runtime_partition{partition}.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    return probabilities, diagnostics


def _pool_predict(
    logs: np.ndarray, allow: np.ndarray, y: np.ndarray, classes: np.ndarray,
    glia: np.ndarray, nested_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.empty(len(y), object)
    fold_index = np.full(len(y), -1, np.int16)
    folds = StratifiedKFold(
        CFG.nested_folds, shuffle=True, random_state=nested_seed
    )
    for fold, (fit, val) in enumerate(folds.split(logs[0], y)):
        prior = pd.Series(y[fit]).value_counts(normalize=True).reindex(classes).fillna(
            EPS
        ).to_numpy()
        log_prior = np.log(prior)
        for mask in (glia, ~glia):
            rr = fit[mask[fit]]
            vv = val[mask[val]]
            w, a = LP.fit(logs, y, classes, log_prior, allow, rows=rr, l2=1e-3)
            z = LP.apply(logs[:, vv], w, a, log_prior, allow[vv])
            pred[vv] = classes[z.argmax(1)]
            fold_index[vv] = fold
    if np.any(fold_index < 0):
        raise AssertionError("nested predictions contain unfilled rows")
    return pred.astype(str), fold_index


def metric_row(
    name: str, pred: np.ndarray, y: np.ndarray, base_pred: np.ndarray,
    glia: np.ndarray, fold_index: np.ndarray,
) -> dict:
    ok, base_ok = pred == y, base_pred == y
    p, _ = M.paired_mcnemar(ok, base_ok)
    wins = int((ok & ~base_ok).sum())
    losses = int((base_ok & ~ok).sum())
    folds = []
    for f in np.unique(fold_index):
        rows = fold_index == f
        folds.append(100 * (ok[rows].mean() - base_ok[rows].mean()))
    return {
        "config": name,
        "accuracy": float(ok.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "glia_accuracy": float(ok[glia].mean()),
        "neuron_accuracy": float(ok[~glia].mean()),
        "gain_pt": float(100 * (ok.mean() - base_ok.mean())),
        "wins": wins, "losses": losses, "p_value": float(p),
        "worst_nested_fold_gain_pt": float(min(folds)),
        "fold_gains_pt": " ".join(f"{v:+.3f}" for v in folds),
    }


def evaluate_partition(
    partition: int, nested_seed: int, probabilities: dict[str, np.ndarray], stage: str,
) -> pd.DataFrame:
    data = B.load_all()
    y, classes = data["y"], data["classes"]
    glia = data["meta_train"]["Region"].isna().to_numpy()
    log_dict, allow, y_cache, classes_cache = SS.part(partition)
    if not np.array_equal(y, y_cache) or not np.array_equal(classes, classes_cache):
        raise ValueError("expert partition does not align with challenge training rows")
    adopted = json.loads(MANIFEST.read_text())["experts"]
    missing = [name for name in adopted if name not in log_dict]
    if missing:
        raise ValueError(f"adopted experts absent from partition {partition}: {missing}")
    base_logs = np.stack([log_dict[n] for n in adopted])
    base_pred, fold_index = _pool_predict(
        base_logs, allow, y, classes, glia, nested_seed
    )
    base_ok = base_pred == y
    rows = [{
        "config": "adopted_40_pool", "accuracy": float(base_ok.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, base_pred)),
        "kappa": float(cohen_kappa_score(y, base_pred)),
        "glia_accuracy": float(base_ok[glia].mean()),
        "neuron_accuracy": float(base_ok[~glia].mean()),
        "gain_pt": 0.0, "wins": 0, "losses": 0, "p_value": 1.0,
        "worst_nested_fold_gain_pt": 0.0,
        "fold_gains_pt": "+0.000 +0.000 +0.000 +0.000 +0.000",
    }]
    for name, probs in probabilities.items():
        masked = np.where(allow, probs, -1.0)
        stand_pred = classes[masked.argmax(1)]
        rows.append(metric_row(
            f"{name}_standalone", stand_pred, y, base_pred, glia, fold_index
        ))
        augmented = np.concatenate(
            [base_logs, np.log(np.maximum(probs, EPS))[None, :, :]], axis=0
        )
        pool_pred, pool_fold = _pool_predict(
            augmented, allow, y, classes, glia, nested_seed
        )
        rows.append(metric_row(
            f"adopted_plus_{name}", pool_pred, y, base_pred, glia, pool_fold
        ))
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    table.to_csv(OUT / f"{stage}_results.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def screen() -> None:
    arms = {"plain", "denoise", "denoise_latent_et"}
    probabilities, runtime = build_oof(SCREEN_PARTITION, arms)
    table = evaluate_partition(
        SCREEN_PARTITION, SCREEN_NESTED_SEED, probabilities, "screen"
    )
    pool = table[table.config.str.startswith("adopted_plus_")].sort_values(
        ["gain_pt", "balanced_accuracy"], ascending=False
    )
    chosen = pool.iloc[0]
    arm = str(chosen.config).removeprefix("adopted_plus_")
    freeze = {
        "stage": "screen_frozen", "selected_arm": arm,
        "screen_partition": SCREEN_PARTITION,
        "screen_nested_seed": SCREEN_NESTED_SEED,
        "screen_metrics": chosen.to_dict(), "config": asdict(CFG),
        "device": runtime["device"],
        "selection_rule": "highest train-only nested gain; one arm confirmed once",
        "confirmation_gate": {
            "gain_pt_min": 0.15, "paired_p_max": 0.05,
            "wins_must_exceed_losses": True,
            "worst_nested_fold_gain_pt_min": -0.10,
        },
        "test_truth_read": False, "production_modified": False,
    }
    (OUT / "freeze.json").write_text(json.dumps(freeze, indent=2))
    print(f"\nfrozen arm for fresh confirmation: {arm}")


def confirm() -> None:
    path = OUT / "freeze.json"
    if not path.exists():
        raise SystemExit("run screen before confirmation")
    freeze = json.loads(path.read_text())
    arm = freeze["selected_arm"]
    probabilities, runtime = build_oof(CONFIRM_PARTITION, {arm})
    table = evaluate_partition(
        CONFIRM_PARTITION, CONFIRM_NESTED_SEED, probabilities, "confirmation"
    )
    row = table.loc[table.config == f"adopted_plus_{arm}"].iloc[0]
    passed = bool(
        row.gain_pt >= 0.15 and row.p_value < 0.05
        and row.wins > row.losses and row.worst_nested_fold_gain_pt >= -0.10
    )
    freeze.update({
        "stage": "confirmation_complete", "confirmation_partition": CONFIRM_PARTITION,
        "confirmation_nested_seed": CONFIRM_NESTED_SEED,
        "confirmation_metrics": row.to_dict(), "confirmation_runtime": runtime,
        "eligible_for_one_test_score": passed,
        "test_truth_read": False, "production_modified": False,
    })
    path.write_text(json.dumps(freeze, indent=2))
    print("\nCONFIRMATION VERDICT: " + (
        "PASS — frozen candidate is eligible for one test score"
        if passed else "REJECT — do not test-score"
    ))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    print(
        f"device={DEVICE} mps_built={torch.backends.mps.is_built()} "
        f"mps_available={torch.backends.mps.is_available()} mode={mode}", flush=True
    )
    if mode == "screen":
        screen()
    elif mode == "confirm":
        confirm()
    else:
        raise SystemExit("usage: iteration22_mps_countsplit.py [screen|confirm]")


if __name__ == "__main__":
    main()
