"""Iteration 16 - ten frozen, genuinely orthogonal model families.

This is an experiment runner, not a submission tuner.  It never imports the recovered
test labels and never writes ``prediction/prediction.csv``.  Every configuration below
is fixed in this source before the separate test thermometer is read.  The common screen
is one stratified 80/20 split (seed 20260821); promising results still require a fresh
multi-fold confirmation before they can be considered for production.

Usage
-----
``python3 notebooks/lib/iteration16_novel_suite.py screen``
``python3 notebooks/lib/iteration16_novel_suite.py test``
``python3 notebooks/lib/iteration16_novel_suite.py all``
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as TF

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_models as M
import iteration16_common as C


DEVICE = torch.device(C.device_name())
DTYPE = torch.float32
EPS = 1e-8


@dataclass(frozen=True)
class Config:
    section_epochs: int = 35
    section_weight: float = 0.12
    supcon_epochs: int = 35
    supcon_weight: float = 0.12
    ecoc_bits: int = 24
    ecoc_trees: int = 160
    ecoc_weight: float = 0.15
    groupdro_epochs: int = 45
    groupdro_weight: float = 0.10
    grok_max_epochs: int = 120
    grok_weight: float = 0.10
    cotrain_weight: float = 0.15
    label_model_weight: float = 0.30
    neural_process_epochs: int = 55
    neural_process_weight: float = 0.15
    hyper_epochs: int = 50
    hyper_weight: float = 0.10
    tta_epochs: int = 45
    tta_weight: float = 0.10


CFG = Config()


def _tensor(x: np.ndarray, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def _numpy_proba(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.detach().cpu(), dim=1).numpy().astype(np.float32)


def _encoded(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(classes)}
    return np.asarray([lookup[str(label)] for label in y], dtype=np.int64)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights /= weights.mean()
    return _tensor(weights)


def _masked_candidate(
    p: np.ndarray, meta_fit: pd.DataFrame, y_fit: np.ndarray,
    meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    return C.mask_probabilities(p, meta_fit, y_fit, meta_eval, classes)


# --------------------------------------------------------------------------- 1
class SectionSetNet(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(d_in, 64), nn.LayerNorm(64), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            64, nhead=4, dim_feedforward=128, dropout=0.10,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        h = self.project(x)
        h = self.context(h, src_key_padding_mask=padding)
        return self.head(h)


def _pad_sections(x: np.ndarray, sections: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(sections.astype(str))
    rows = [np.flatnonzero(sections.astype(str) == section) for section in unique]
    width = max(map(len, rows))
    padded = np.zeros((len(rows), width, x.shape[1]), np.float32)
    padding = np.ones((len(rows), width), bool)
    origin = np.full((len(rows), width), -1, np.int64)
    for i, section_rows in enumerate(rows):
        padded[i, :len(section_rows)] = x[section_rows]
        padding[i, :len(section_rows)] = False
        origin[i, :len(section_rows)] = section_rows
    return padded, padding, origin


def section_set_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 64)
    all_x = np.vstack([fit_z, eval_z])
    all_sections = np.concatenate([
        meta_fit["Section_ID"].astype(str).to_numpy(),
        meta_eval["Section_ID"].astype(str).to_numpy(),
    ])
    padded, padding, origin = _pad_sections(all_x, all_sections)
    y_code = _encoded(y_fit, classes)
    all_y = np.full(len(all_x), -1, np.int64)
    all_y[:len(y_fit)] = y_code
    targets = np.full(origin.shape, -1, np.int64)
    valid = origin >= 0
    targets[valid] = all_y[origin[valid]]

    model = SectionSetNet(all_x.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    px, pm, py = _tensor(padded), _tensor(padding, torch.bool), _tensor(targets, torch.long)
    weights = _class_weights(y_code, len(classes))
    model.train()
    for _ in range(CFG.section_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(px, pm)
        loss = TF.cross_entropy(logits.reshape(-1, len(classes)), py.reshape(-1),
                                weight=weights, ignore_index=-1)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(px, pm).detach().cpu().numpy()
    flat = np.zeros((len(all_x), len(classes)), np.float32)
    flat[origin[valid]] = logits[valid]
    return _masked_candidate(
        softmax(flat[len(y_fit):], axis=1).astype(np.float32),
        meta_fit, y_fit, meta_eval, classes,
    )


# --------------------------------------------------------------------------- 2
class SupConNet(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(96, 48),
        )
        self.head = nn.Linear(48, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = TF.normalize(self.encoder(x), dim=1)
        return z, self.head(z)


def _supcon_loss(z: torch.Tensor, y: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
    similarity = z @ z.T / 0.10
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (y[:, None] == y[None, :]) & (mouse[:, None] != mouse[None, :]) & ~eye
    allowed = ~eye
    similarity = similarity - similarity.max(1, keepdim=True).values.detach()
    denominator = torch.logsumexp(similarity.masked_fill(~allowed, -1e9), dim=1)
    log_probability = similarity - denominator[:, None]
    counts = positives.sum(1)
    keep = counts > 0
    if not keep.any():
        return z.sum() * 0.0
    return -(log_probability * positives).sum(1)[keep].div(counts[keep]).mean()


def supervised_contrastive_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 96)
    y_code = _encoded(y_fit, classes)
    mouse_values = pd.factorize(meta_fit["Mouse_ID"].astype(str))[0].astype(np.int64)
    model = SupConNet(fit_z.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    weights = _class_weights(y_code, len(classes))
    rng = np.random.default_rng(C.SCREEN_SEED + 2)
    model.train()
    for _ in range(CFG.supcon_epochs):
        order = rng.permutation(len(y_fit))
        for start in range(0, len(order), 512):
            rows = order[start:start + 512]
            bx = _tensor(fit_z[rows]); by = _tensor(y_code[rows], torch.long)
            bm = _tensor(mouse_values[rows], torch.long)
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(bx)
            loss = TF.cross_entropy(logits, by, weight=weights)
            loss = loss + 0.15 * _supcon_loss(embedding, by, bm)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_embedding, _ = model(_tensor(fit_z))
        eval_embedding, _ = model(_tensor(eval_z))
        prototypes = []
        y_tensor = _tensor(y_code, torch.long)
        for cls in range(len(classes)):
            prototypes.append(TF.normalize(train_embedding[y_tensor == cls].mean(0), dim=0))
        prototypes = torch.stack(prototypes)
        logits = eval_embedding @ prototypes.T / 0.15
    return _masked_candidate(
        _numpy_proba(logits), meta_fit, y_fit, meta_eval, classes
    )


# --------------------------------------------------------------------------- 3
def _ecoc_code(n_classes: int, n_bits: int) -> np.ndarray:
    rng = np.random.default_rng(C.SCREEN_SEED + 3)
    code = [rng.integers(0, 2, n_bits, dtype=np.int8)]
    for _ in range(1, n_classes):
        candidates = rng.integers(0, 2, (512, n_bits), dtype=np.int8)
        distance = np.min(
            np.stack([np.sum(candidates != previous, axis=1) for previous in code]), axis=0
        )
        balance_penalty = np.abs(candidates.mean(1) - 0.5)
        choice = np.argmax(distance - 2.0 * balance_penalty)
        code.append(candidates[choice])
    result = np.asarray(code)
    if len(np.unique(result, axis=0)) != n_classes:
        raise AssertionError("ECOC construction produced duplicate class words")
    return result


def ecoc_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    code = _ecoc_code(len(classes), CFG.ecoc_bits)
    y_code = _encoded(y_fit, classes)
    bit_probability = np.zeros((len(x_eval), CFG.ecoc_bits), np.float32)
    for bit in range(CFG.ecoc_bits):
        target = code[y_code, bit]
        model = ExtraTreesClassifier(
            n_estimators=CFG.ecoc_trees, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=C.SCREEN_SEED + bit,
        ).fit(x_fit, target)
        bit_probability[:, bit] = model.predict_proba(x_eval)[:, 1]
    p = np.clip(bit_probability, 1e-5, 1 - 1e-5)
    log_likelihood = (
        code[None, :, :] * np.log(p[:, None, :])
        + (1 - code[None, :, :]) * np.log1p(-p[:, None, :])
    ).sum(2)
    return _masked_candidate(
        softmax(log_likelihood, axis=1).astype(np.float32),
        meta_fit, y_fit, meta_eval, classes,
    )


# --------------------------------------------------------------------------- 4
class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class GroupDRONet(nn.Module):
    def __init__(self, d_in: int, n_classes: int, n_mice: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.mouse = nn.Linear(64, n_mice)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.classifier(h), self.mouse(ReverseGradient.apply(h, 0.20))


def groupdro_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 96)
    y_code = _encoded(y_fit, classes)
    mouse_code, mouse_names = pd.factorize(meta_fit["Mouse_ID"].astype(str))
    group = y_code * len(mouse_names) + mouse_code
    unique, inverse = np.unique(group, return_inverse=True)
    model = GroupDRONet(fit_z.shape[1], len(classes), len(mouse_names)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-3)
    tx = _tensor(fit_z); ty = _tensor(y_code, torch.long)
    tm = _tensor(mouse_code, torch.long); tg = _tensor(inverse, torch.long)
    q = torch.ones(len(unique), device=DEVICE) / len(unique)
    for _ in range(CFG.groupdro_epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits, mouse_logits = model(tx)
        losses = TF.cross_entropy(logits, ty, reduction="none")
        group_sum = torch.zeros(len(unique), device=DEVICE).index_add(0, tg, losses)
        group_n = torch.zeros(len(unique), device=DEVICE).index_add(
            0, tg, torch.ones_like(losses)
        )
        group_loss = group_sum / group_n.clamp_min(1)
        with torch.no_grad():
            q *= torch.exp(0.06 * group_loss.detach().clamp(max=8))
            q /= q.sum()
        loss = (q * group_loss).sum() + 0.05 * TF.cross_entropy(mouse_logits, tm)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(_tensor(eval_z))
    return _masked_candidate(
        _numpy_proba(logits), meta_fit, y_fit, meta_eval, classes
    )


# --------------------------------------------------------------------------- 5
def _gene_tokens(counts: np.ndarray, top_k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, counts.shape[1])
    indices = np.argpartition(counts, -k, axis=1)[:, -k:]
    values = np.take_along_axis(counts, indices, axis=1)
    order = np.argsort(-values, axis=1)
    indices = np.take_along_axis(indices, order, axis=1).astype(np.int64)
    values = np.take_along_axis(values, order, axis=1)
    totals = np.maximum(counts.sum(1, keepdims=True), 1.0)
    values = np.log1p(values / totals * 1e4).astype(np.float32)
    return indices, values


class GeneTokenNet(nn.Module):
    def __init__(self, n_genes: int, n_classes: int) -> None:
        super().__init__()
        self.gene = nn.Embedding(n_genes, 48)
        self.value = nn.Sequential(nn.Linear(1, 48), nn.GELU(), nn.Linear(48, 48))
        layer = nn.TransformerEncoderLayer(
            48, nhead=4, dim_feedforward=96, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, n_classes))

    def forward(self, genes: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        h = self.gene(genes) + self.value(values[..., None])
        h = self.transformer(h)
        weights = torch.softmax(values, dim=1)[..., None]
        return self.head((h * weights).sum(1))


def _train_gene_token(
    counts: np.ndarray, y_code: np.ndarray, n_classes: int, epochs: int,
    monitor: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[GeneTokenNet, list[dict]]:
    ids, vals = _gene_tokens(counts)
    model = GeneTokenNet(counts.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    weights = _class_weights(y_code, n_classes)
    rng = np.random.default_rng(C.SCREEN_SEED + 5)
    gradient_ema: dict[int, torch.Tensor] = {}
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y_code))
        for start in range(0, len(y_code), 256):
            rows = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            logits = model(_tensor(ids[rows], torch.long), _tensor(vals[rows]))
            loss = TF.cross_entropy(logits, _tensor(y_code[rows], torch.long), weight=weights)
            loss.backward()
            # Grokfast: amplify the slowly varying component of each gradient.
            for index, parameter in enumerate(model.parameters()):
                if parameter.grad is None:
                    continue
                old = gradient_ema.get(index)
                ema = parameter.grad.detach().clone() if old is None else 0.98 * old + 0.02 * parameter.grad.detach()
                gradient_ema[index] = ema
                parameter.grad.add_(ema, alpha=1.0)
            optimizer.step()
        if monitor is not None and (epoch == 1 or epoch % 5 == 0):
            monitor_counts, monitor_y = monitor
            model.eval()
            with torch.no_grad():
                tr_logits = model(_tensor(ids, torch.long), _tensor(vals))
                mi, mv = _gene_tokens(monitor_counts)
                va_logits = model(_tensor(mi, torch.long), _tensor(mv))
            history.append({
                "epoch": epoch,
                "train_accuracy": float((tr_logits.argmax(1).cpu().numpy() == y_code).mean()),
                "monitor_accuracy": float((va_logits.argmax(1).cpu().numpy() == monitor_y).mean()),
            })
    return model, history


def gene_token_grokfast_candidate(
    counts_fit: np.ndarray, y_fit: np.ndarray, counts_eval: np.ndarray,
    meta_fit: pd.DataFrame, meta_eval: pd.DataFrame, classes: np.ndarray,
    selected_epoch: int | None,
) -> tuple[np.ndarray, int, dict]:
    y_code = _encoded(y_fit, classes)
    diagnostics: dict = {}
    if selected_epoch is None:
        inner, monitor = train_test_split(
            np.arange(len(y_fit)), test_size=0.20, random_state=C.SCREEN_SEED + 55,
            stratify=y_fit,
        )
        _, history = _train_gene_token(
            counts_fit[inner], y_code[inner], len(classes), CFG.grok_max_epochs,
            monitor=(counts_fit[monitor], y_code[monitor]),
        )
        frame = pd.DataFrame(history)
        saturated = frame.loc[frame.train_accuracy >= 0.98]
        eligible = frame.loc[frame.epoch >= int(saturated.epoch.min())] if len(saturated) else frame
        selected_epoch = int(eligible.loc[eligible.monitor_accuracy.idxmax(), "epoch"])
        before = float(frame.iloc[0].monitor_accuracy)
        best = float(eligible.monitor_accuracy.max())
        diagnostics = {
            "selected_epoch": selected_epoch,
            "train_saturated": bool(len(saturated)),
            "grokking_detected": bool(len(saturated) and best - before >= 0.01),
            "first_monitor_accuracy": before,
            "best_post_saturation_monitor_accuracy": best,
        }
        frame.to_csv(C.OUT / "grokfast_learning_curve.csv", index=False)
    model, _ = _train_gene_token(
        counts_fit, y_code, len(classes), selected_epoch, monitor=None
    )
    ids, vals = _gene_tokens(counts_eval)
    model.eval()
    with torch.no_grad():
        probabilities = _numpy_proba(model(_tensor(ids, torch.long), _tensor(vals)))
    probabilities = _masked_candidate(
        probabilities, meta_fit, y_fit, meta_eval, classes
    )
    return probabilities, selected_epoch, diagnostics


# --------------------------------------------------------------------------- 6
VIEW_SLICES = (slice(0, 371), slice(371, 560), slice(560, 694))


def _view_model(
    x_fit: np.ndarray, y_fit: np.ndarray, x_eval: np.ndarray,
    classes: np.ndarray, seed: int, sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=240, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    ).fit(x_fit, y_fit, sample_weight=sample_weight)
    probabilities = M.align_proba(model, x_eval, classes.tolist())
    return M.correct_prior(
        probabilities, M.prior_vector(pd.Series(y_fit[:len(sample_weight)] if sample_weight is not None else y_fit), classes.tolist()),
        C.ALPHA,
    ).astype(np.float32)


def multiview_cotraining_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    initial: list[np.ndarray] = []
    for view, block in enumerate(VIEW_SLICES):
        initial.append(_view_model(
            x_fit[:, block], y_fit, x_eval[:, block], classes,
            C.SCREEN_SEED + 600 + view,
        ))
    predictions = [p.argmax(1) for p in initial]
    confidences = [p.max(1) for p in initial]
    retrained: list[np.ndarray] = []
    pseudo_counts: list[int] = []
    for target in range(3):
        others = [i for i in range(3) if i != target]
        use = (
            (predictions[others[0]] == predictions[others[1]])
            & (confidences[others[0]] >= 0.75)
            & (confidences[others[1]] >= 0.75)
        )
        rows = np.flatnonzero(use)
        maximum = max(1, int(0.20 * len(x_eval)))
        if len(rows) > maximum:
            score = confidences[others[0]][rows] * confidences[others[1]][rows]
            rows = rows[np.argsort(score)[-maximum:]]
        pseudo = classes[predictions[others[0]][rows]]
        block = VIEW_SLICES[target]
        aug_x = np.vstack([x_fit[:, block], x_eval[rows, block]])
        aug_y = np.concatenate([y_fit, pseudo])
        weights = np.concatenate([np.ones(len(y_fit)), np.full(len(rows), 0.35)])
        model = ExtraTreesClassifier(
            n_estimators=240, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1,
            random_state=C.SCREEN_SEED + 630 + target,
        ).fit(aug_x, aug_y, sample_weight=weights)
        probabilities = M.align_proba(model, x_eval[:, block], classes.tolist())
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y_fit), classes.tolist()), C.ALPHA
        )
        retrained.append(probabilities.astype(np.float32))
        pseudo_counts.append(len(rows))
    output = C.normalize_probabilities(np.mean(retrained, axis=0))
    return _masked_candidate(output, meta_fit, y_fit, meta_eval, classes), {
        "pseudo_labels_by_view": pseudo_counts,
        "mean_pseudo_labels": float(np.mean(pseudo_counts)),
    }


# --------------------------------------------------------------------------- 8
class NeuralProcessClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.x_encoder = nn.Sequential(nn.Linear(d_in, 64), nn.LayerNorm(64), nn.GELU())
        self.label = nn.Embedding(n_classes, 16)
        self.context = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 64))
        self.decoder = nn.Sequential(
            nn.Linear(128, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, n_classes)
        )

    def representations(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.x_encoder(x)
        c = self.context(torch.cat([h, self.label(y)], dim=1))
        return h, c

    def decode(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([h, context], dim=1))


def _aggregate_context(
    values: torch.Tensor, sections: torch.Tensor, support: torch.Tensor, n_sections: int,
) -> torch.Tensor:
    sums = torch.zeros((n_sections, values.shape[1]), device=values.device)
    counts = torch.zeros(n_sections, device=values.device)
    sums.index_add_(0, sections[support], values[support])
    counts.index_add_(
        0, sections[support],
        torch.ones_like(sections[support], dtype=values.dtype),
    )
    context = sums / counts[:, None].clamp_min(1)
    global_context = values[support].mean(0)
    context[counts == 0] = global_context
    return context


def neural_process_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 64)
    all_sections = np.concatenate([
        meta_fit["Section_ID"].astype(str).to_numpy(),
        meta_eval["Section_ID"].astype(str).to_numpy(),
    ])
    section_code, section_names = pd.factorize(all_sections)
    fit_sections = section_code[:len(y_fit)].astype(np.int64)
    eval_sections = section_code[len(y_fit):].astype(np.int64)
    y_code = _encoded(y_fit, classes)
    model = NeuralProcessClassifier(fit_z.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    weights = _class_weights(y_code, len(classes))
    tx = _tensor(fit_z); ty = _tensor(y_code, torch.long)
    ts = _tensor(fit_sections, torch.long)
    rng = np.random.default_rng(C.SCREEN_SEED + 8)
    for _ in range(CFG.neural_process_epochs):
        support_np = rng.random(len(y_fit)) < 0.50
        for section in np.unique(fit_sections):
            rows = np.flatnonzero(fit_sections == section)
            if len(rows) > 1:
                support_np[rows[0]] = True
                support_np[rows[-1]] = False
        support = _tensor(support_np, torch.bool)
        query = ~support
        model.train(); optimizer.zero_grad(set_to_none=True)
        h, c = model.representations(tx, ty)
        context = _aggregate_context(c, ts, support, len(section_names))
        logits = model.decode(h, context[ts])
        loss = TF.cross_entropy(logits[query], ty[query], weight=weights)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        h_fit, c_fit = model.representations(tx, ty)
        full_support = torch.ones(len(y_fit), dtype=torch.bool, device=DEVICE)
        context = _aggregate_context(c_fit, ts, full_support, len(section_names))
        h_eval = model.x_encoder(_tensor(eval_z))
        logits = model.decode(h_eval, context[_tensor(eval_sections, torch.long)])
    return _masked_candidate(
        _numpy_proba(logits), meta_fit, y_fit, meta_eval, classes
    )


# --------------------------------------------------------------------------- 9
class SectionHyperNet(nn.Module):
    def __init__(self, d_in: int, n_classes: int, rank: int = 4) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.rank = rank
        self.encoder = nn.Sequential(nn.Linear(d_in, 64), nn.LayerNorm(64), nn.GELU())
        self.global_head = nn.Linear(64, n_classes)
        self.low_rank_b = nn.Parameter(torch.randn(rank, 64) * 0.05)
        self.hyper = nn.Sequential(
            nn.Linear(d_in, 32), nn.GELU(), nn.Linear(32, n_classes * rank)
        )

    def forward(
        self, x: torch.Tensor, section: torch.Tensor, summaries: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x)
        code = h @ self.low_rank_b.T
        a = self.hyper(summaries).view(-1, self.n_classes, self.rank)
        correction = torch.einsum("nr,ncr->nc", code, a[section])
        return self.global_head(h) + 0.10 * correction


def hypernetwork_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> np.ndarray:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 64)
    all_z = np.vstack([fit_z, eval_z])
    all_sections = np.concatenate([
        meta_fit["Section_ID"].astype(str).to_numpy(),
        meta_eval["Section_ID"].astype(str).to_numpy(),
    ])
    section_code, section_names = pd.factorize(all_sections)
    summaries = np.zeros((len(section_names), all_z.shape[1]), np.float32)
    for section in range(len(section_names)):
        summaries[section] = all_z[section_code == section].mean(0)
    y_code = _encoded(y_fit, classes)
    model = SectionHyperNet(fit_z.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-3)
    weights = _class_weights(y_code, len(classes))
    tx = _tensor(fit_z); ty = _tensor(y_code, torch.long)
    ts = _tensor(section_code[:len(y_fit)], torch.long); summary = _tensor(summaries)
    for _ in range(CFG.hyper_epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(tx, ts, summary)
        loss = TF.cross_entropy(logits, ty, weight=weights)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(
            _tensor(eval_z), _tensor(section_code[len(y_fit):], torch.long), summary
        )
    return _masked_candidate(
        _numpy_proba(logits), meta_fit, y_fit, meta_eval, classes
    )


# -------------------------------------------------------------------------- 10
class TTANet(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.first = nn.Linear(d_in, 96)
        self.norm = nn.LayerNorm(96)
        self.classifier = nn.Linear(96, n_classes)
        self.decoder = nn.Sequential(nn.Linear(96, 96), nn.GELU(), nn.Linear(96, d_in))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = TF.gelu(self.norm(self.first(x)))
        return self.classifier(h), self.decoder(h)


def test_time_adaptation_candidate(
    x_fit: np.ndarray, y_fit: np.ndarray, meta_fit: pd.DataFrame,
    x_eval: np.ndarray, meta_eval: pd.DataFrame, classes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    fit_z, eval_z, _, _ = C.reduce_features(x_fit, x_eval, 64)
    y_code = _encoded(y_fit, classes)
    model = TTANet(fit_z.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    weights = _class_weights(y_code, len(classes))
    tx = _tensor(fit_z); ty = _tensor(y_code, torch.long)
    generator = torch.Generator(device="cpu").manual_seed(C.SCREEN_SEED + 10)
    for _ in range(CFG.tta_epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        mask = torch.rand(tx.shape, generator=generator).to(DEVICE) < 0.15
        logits, reconstruction = model(tx.masked_fill(mask, 0.0))
        reconstruct_loss = ((reconstruction - tx) ** 2)[mask].mean()
        loss = TF.cross_entropy(logits, ty, weight=weights) + 0.10 * reconstruct_loss
        loss.backward(); optimizer.step()
    source = copy.deepcopy(model).eval()
    anchor_rows = np.random.default_rng(C.SCREEN_SEED + 100).choice(
        len(fit_z), size=min(256, len(fit_z)), replace=False
    )
    anchor = _tensor(fit_z[anchor_rows])
    with torch.no_grad():
        anchor_logits, _ = source(anchor)
        anchor_distribution = torch.softmax(anchor_logits, dim=1)
    sections = meta_eval["Section_ID"].astype(str).to_numpy()
    output = np.zeros((len(x_eval), len(classes)), np.float32)
    rollback_sections = 0
    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        adapted = copy.deepcopy(source).train()
        for parameter in adapted.parameters():
            parameter.requires_grad_(False)
        adapted.norm.weight.requires_grad_(True)
        adapted.norm.bias.requires_grad_(True)
        adaptation_optimizer = torch.optim.Adam(
            [adapted.norm.weight, adapted.norm.bias], lr=5e-3
        )
        target = _tensor(eval_z[rows])
        for _ in range(8):
            adaptation_optimizer.zero_grad(set_to_none=True)
            mask = torch.rand(target.shape, generator=generator).to(DEVICE) < 0.20
            _, reconstruction = adapted(target.masked_fill(mask, 0.0))
            anchor_new, _ = adapted(anchor)
            reconstruction_loss = ((reconstruction - target) ** 2)[mask].mean()
            anchor_loss = TF.kl_div(
                torch.log_softmax(anchor_new, dim=1), anchor_distribution,
                reduction="batchmean",
            )
            (reconstruction_loss + 0.05 * anchor_loss).backward()
            adaptation_optimizer.step()
        adapted.eval()
        with torch.no_grad():
            anchor_new, _ = adapted(anchor)
            drift = TF.kl_div(
                torch.log_softmax(anchor_new, dim=1), anchor_distribution,
                reduction="batchmean",
            ).item()
            chosen = source if drift > 0.05 or not np.isfinite(drift) else adapted
            rollback_sections += int(chosen is source)
            logits, _ = chosen(target)
            output[rows] = _numpy_proba(logits)
    return _masked_candidate(output, meta_fit, y_fit, meta_eval, classes), {
        "sections": int(len(np.unique(sections))),
        "rollback_sections": int(rollback_sections),
    }


# --------------------------------------------------------------------------- 7
def _fit_label_parameters(
    sources: np.ndarray, truth: np.ndarray, classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a correlation-tempered Dawid-Skene-style annotator model."""
    # sources: annotator x cell x class
    votes = sources.argmax(2)
    y_code = _encoded(truth, classes)
    n_sources, _, n_classes = sources.shape
    confusion = np.ones((n_sources, n_classes, n_classes), np.float64)
    for source in range(n_sources):
        for true in range(n_classes):
            counts = np.bincount(votes[source, y_code == true], minlength=n_classes)
            confusion[source, true] += counts
        confusion[source] /= confusion[source].sum(1, keepdims=True)
    errors = (votes != y_code[None, :]).astype(float)
    correlation = np.nan_to_num(np.corrcoef(errors), nan=0.0)
    redundancy = 1.0 + np.maximum(np.abs(correlation) - 0.50, 0.0).sum(1)
    accuracy = (votes == y_code[None, :]).mean(1)
    weights = np.maximum(accuracy - 1.0 / n_classes, 0.01) / redundancy
    weights /= weights.mean()
    prior = np.bincount(y_code, minlength=n_classes).astype(float) + 1.0
    prior /= prior.sum()
    return confusion, weights, prior


def _apply_label_parameters(
    sources: np.ndarray, parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    confusion, weights, prior = parameters
    votes = sources.argmax(2)
    log_p = np.broadcast_to(np.log(prior)[None, :], (sources.shape[1], len(prior))).copy()
    for source in range(sources.shape[0]):
        # confusion[source, true class, observed vote]
        log_p += weights[source] * np.log(confusion[source, :, votes[source]] + EPS)
    return softmax(log_p, axis=1).astype(np.float32)


def label_model_screen(
    source_probabilities: dict[str, np.ndarray], truth: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, dict]:
    from sklearn.model_selection import StratifiedKFold

    names = list(source_probabilities)
    sources = np.stack([source_probabilities[name] for name in names])
    output = np.zeros((len(truth), len(classes)), np.float32)
    folds = StratifiedKFold(5, shuffle=True, random_state=C.SCREEN_SEED + 7)
    for fit, valid in folds.split(np.arange(len(truth)), truth):
        parameters = _fit_label_parameters(sources[:, fit], truth[fit], classes)
        output[valid] = _apply_label_parameters(sources[:, valid], parameters)
    parameters = _fit_label_parameters(sources, truth, classes)
    np.savez_compressed(
        C.OUT / "label_model_parameters.npz",
        confusion=parameters[0], weights=parameters[1], prior=parameters[2],
        source_names=np.asarray(names), classes=classes,
    )
    return output, {"sources": names, "weights": dict(zip(names, parameters[1].tolist()))}


def label_model_test(source_probabilities: dict[str, np.ndarray]) -> np.ndarray:
    cached = np.load(C.OUT / "label_model_parameters.npz", allow_pickle=True)
    names = cached["source_names"].astype(str).tolist()
    if set(names) != set(source_probabilities):
        raise ValueError("label-model sources do not match frozen screen")
    sources = np.stack([source_probabilities[name] for name in names])
    parameters = (cached["confusion"], cached["weights"], cached["prior"])
    return _apply_label_parameters(sources, parameters)


# ---------------------------------------------------------------------- runner
WEIGHTS = {
    "section_set": CFG.section_weight,
    "supcon_prototypes": CFG.supcon_weight,
    "ecoc": CFG.ecoc_weight,
    "groupdro": CFG.groupdro_weight,
    "gene_token_grokfast": CFG.grok_weight,
    "multiview_cotraining": CFG.cotrain_weight,
    "weak_label_model": CFG.label_model_weight,
    "neural_process": CFG.neural_process_weight,
    "hypernetwork": CFG.hyper_weight,
    "test_time_adaptation": CFG.tta_weight,
}


def _clean_device() -> None:
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def _call_candidate(
    name: str, function, failures: dict, diagnostics: dict,
    base: np.ndarray,
) -> np.ndarray:
    C.seed_everything(C.SCREEN_SEED + len(diagnostics) * 101)
    started = time.time()
    try:
        result = function()
        if isinstance(result, tuple):
            probabilities, detail = result
            diagnostics[name] = detail
        else:
            probabilities = result
            diagnostics[name] = {}
        probabilities = C.normalize_probabilities(probabilities)
        if probabilities.shape != base.shape or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid probabilities: {probabilities.shape}")
        diagnostics[name]["seconds"] = round(time.time() - started, 3)
        print(f"[{name}] complete in {time.time()-started:.1f}s", flush=True)
        return probabilities
    except Exception as error:  # keep the suite auditable even when one moonshot fails
        failures[name] = f"{type(error).__name__}: {error}"
        diagnostics[name] = {"seconds": round(time.time() - started, 3), "failed": True}
        print(f"[{name}] FAILED: {failures[name]}", flush=True)
        return base.copy()
    finally:
        _clean_device()


def run_stage(stage: str, data: dict) -> None:
    if stage == "screen":
        fit = data["fit"]
        eval_rows = data["valid"]
        x_fit, x_eval = data["x_train"][fit], data["x_train"][eval_rows]
        counts_fit = data["counts_train"][fit]
        counts_eval = data["counts_train"][eval_rows]
        y_fit, eval_truth = data["y"][fit], data["y"][eval_rows]
        meta_fit = data["meta_train"].iloc[fit]
        meta_eval = data["meta_train"].iloc[eval_rows]
        seeds = C.SCREEN_ET_SEEDS
        grok_epoch = None
    elif stage == "test":
        fit = np.arange(len(data["y"]))
        eval_rows = np.arange(len(data["x_test"]))
        x_fit, x_eval = data["x_train"], data["x_test"]
        counts_fit, counts_eval = data["counts_train"], data["counts_test"]
        y_fit, eval_truth = data["y"], None
        meta_fit, meta_eval = data["meta_train"], data["meta_test"]
        seeds = C.TEST_ET_SEEDS
        state_path = C.OUT / "frozen_state.json"
        if not state_path.exists():
            raise SystemExit("test stage requires a completed screen to freeze Grokfast epoch")
        grok_epoch = int(json.loads(state_path.read_text())["grokfast_selected_epoch"])
    else:
        raise ValueError(stage)

    classes = data["classes"]
    failures: dict[str, str] = {}
    diagnostics: dict[str, dict] = {}
    print(
        f"stage={stage} device={DEVICE} fit={len(x_fit)} eval={len(x_eval)} "
        f"features={x_fit.shape[1]} classes={len(classes)}",
        flush=True,
    )
    started = time.time()
    base = C.fit_incumbent(
        x_fit, y_fit, meta_fit, x_eval, meta_eval, classes, seeds
    )
    print(f"[incumbent] complete in {time.time()-started:.1f}s", flush=True)

    raw: dict[str, np.ndarray] = {}
    raw["section_set"] = _call_candidate(
        "section_set",
        lambda: section_set_candidate(x_fit, y_fit, meta_fit, x_eval, meta_eval, classes),
        failures, diagnostics, base,
    )
    raw["supcon_prototypes"] = _call_candidate(
        "supcon_prototypes",
        lambda: supervised_contrastive_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )
    raw["ecoc"] = _call_candidate(
        "ecoc", lambda: ecoc_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )
    raw["groupdro"] = _call_candidate(
        "groupdro", lambda: groupdro_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )

    grok_result: dict = {}
    def run_grokfast():
        probabilities, epoch, detail = gene_token_grokfast_candidate(
            counts_fit, y_fit, counts_eval, meta_fit, meta_eval, classes, grok_epoch
        )
        grok_result["epoch"] = epoch
        return probabilities, detail
    raw["gene_token_grokfast"] = _call_candidate(
        "gene_token_grokfast", run_grokfast, failures, diagnostics, base
    )
    if stage == "screen" and "epoch" not in grok_result:
        grok_result["epoch"] = 60

    raw["multiview_cotraining"] = _call_candidate(
        "multiview_cotraining", lambda: multiview_cotraining_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )
    raw["neural_process"] = _call_candidate(
        "neural_process", lambda: neural_process_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )
    raw["hypernetwork"] = _call_candidate(
        "hypernetwork", lambda: hypernetwork_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )
    raw["test_time_adaptation"] = _call_candidate(
        "test_time_adaptation", lambda: test_time_adaptation_candidate(
            x_fit, y_fit, meta_fit, x_eval, meta_eval, classes
        ), failures, diagnostics, base,
    )

    final = {name: C.blend(base, probabilities, WEIGHTS[name])
             for name, probabilities in raw.items()}
    label_sources = {"incumbent": base, **final}
    if stage == "screen":
        label_raw = _call_candidate(
            "weak_label_model",
            lambda: label_model_screen(label_sources, eval_truth, classes),
            failures, diagnostics, base,
        )
    else:
        label_raw = _call_candidate(
            "weak_label_model",
            lambda: label_model_test(label_sources),
            failures, diagnostics, base,
        )
    raw["weak_label_model"] = label_raw
    final["weak_label_model"] = C.blend(base, label_raw, WEIGHTS["weak_label_model"])

    payload = {"incumbent": base, **{f"raw__{k}": v for k, v in raw.items()},
               **{f"final__{k}": v for k, v in final.items()},
               "classes": classes, "eval_rows": eval_rows}
    np.savez_compressed(C.OUT / f"{stage}_probabilities.npz", **payload)

    if stage == "screen":
        glia = meta_eval["Region"].isna().to_numpy()
        base_correct = classes[base.argmax(1)] == eval_truth
        rows = [C.metric_row("incumbent", base, eval_truth, classes, glia)]
        rows[0]["gain_pt"] = 0.0
        standalone_rows = []
        for name in WEIGHTS:
            row = C.metric_row(name, final[name], eval_truth, classes, glia, base_correct)
            row["gain_pt"] = 100 * (row["accuracy"] - rows[0]["accuracy"])
            row["changed_vs_incumbent"] = int(
                np.sum(final[name].argmax(1) != base.argmax(1))
            )
            rows.append(row)
            standalone = C.metric_row(name, raw[name], eval_truth, classes, glia)
            standalone["blend_weight"] = WEIGHTS[name]
            standalone_rows.append(standalone)
        table = pd.DataFrame(rows)
        table.to_csv(C.OUT / "screen_results.csv", index=False)
        pd.DataFrame(standalone_rows).to_csv(C.OUT / "screen_standalone.csv", index=False)
        C.save_json(C.OUT / "frozen_state.json", {
            "screen_seed": C.SCREEN_SEED,
            "grokfast_selected_epoch": int(grok_result["epoch"]),
            "device": str(DEVICE),
            "weights": WEIGHTS,
            "config": CFG.__dict__,
            "test_truth_read": False,
        })
        print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    else:
        production = pd.read_csv(C.PRODUCTION, dtype={"Cell_ID": str})
        production_pred = production.iloc[:, 1].astype(str).to_numpy()
        base_pred = classes[base.argmax(1)]
        manifest_rows = []
        for name in WEIGHTS:
            path = C.write_candidate(name, final[name], meta_eval, classes)
            pred = classes[final[name].argmax(1)]
            manifest_rows.append({
                "candidate": name,
                "file": str(path),
                "blend_weight": WEIGHTS[name],
                "changed_vs_production": int(np.sum(pred != production_pred)),
                "distinct_labels": int(len(np.unique(pred))),
                "training_failed": name in failures,
            })
        pd.DataFrame(manifest_rows).to_csv(C.OUT / "test_manifest.csv", index=False)
        C.save_json(C.OUT / "test_freeze.json", {
            "device": str(DEVICE),
            "incumbent_reproduced_exactly": bool(np.array_equal(base_pred, production_pred)),
            "incumbent_disagreements": int(np.sum(base_pred != production_pred)),
            "candidate_count": len(manifest_rows),
            "test_truth_read": False,
            "production_modified": False,
        })
        print("\n" + pd.DataFrame(manifest_rows).to_string(index=False))

    C.save_json(C.OUT / f"{stage}_diagnostics.json", {
        "device": str(DEVICE),
        "seconds": round(time.time() - started, 3),
        "failures": failures,
        "methods": diagnostics,
        "test_truth_read": False,
    })


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in {"screen", "test", "all"}:
        raise SystemExit("mode must be screen, test, or all")
    C.seed_everything()
    data = C.load_data()
    if mode in {"screen", "all"}:
        run_stage("screen", data)
    if mode in {"test", "all"}:
        run_stage("test", data)
    print(f"completed mode={mode}; recovered test truth was never imported", flush=True)


if __name__ == "__main__":
    main()
