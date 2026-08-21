"""Iteration 10 - semi-supervised, batch-conditioned ZINB CVAE gate.

This is the defensible core of Gemini's sanitized generative-count suggestion.  Spatial
prediction smoothing is intentionally omitted because this repository measured only
9.4% neighbor homophily and already rejected graph smoothing.  The model instead:

* models raw 200-gene counts with a zero-inflated negative-binomial decoder;
* learns a 32-D latent code from 20k non-challenge atlas cells, challenge train cells,
  and label-free challenge validation/query counts;
* conditions the decoder on atlas-versus-challenge domain;
* uses challenge labels at weight 1.0 and external atlas labels at weight 0.15.

Compute gate (not adoption): frozen stratified 80/20 split seed 487, 120 epochs, one
neural seed on MPS, and the five-seed ExtraTrees incumbent.  Advance only if standalone
accuracy >=0.78 and the fixed 80/20 blend improves ExtraTrees by >0.30 points.  No hidden
test label or withheld gene is read.

Usage:
    python3 notebooks/lib/iteration10_zinb_cvae.py gate
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import iteration9_atlas_model as I9A
from iteration10_spatial_fields import current_stack

OUT = Path("outputs/iteration10")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 487
EPOCHS = 120
ATLAS_SAMPLE = 20_000
ALPHA = 0.45
BLEND_WEIGHT = 0.20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ZINBCVAE(nn.Module):
    def __init__(self, n_genes: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes + 1, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.z_mean = nn.Linear(128, 32)
        self.z_logvar = nn.Linear(128, 32)
        self.domain = nn.Embedding(2, 8)
        self.decoder = nn.Sequential(
            nn.Linear(40, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(),
        )
        self.mean_logits = nn.Linear(256, n_genes)
        self.dropout_logits = nn.Linear(256, n_genes)
        self.log_dispersion = nn.Parameter(torch.zeros(n_genes))
        self.classifier = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, n_classes)
        )

    def forward(self, scaled_log_counts, library_size, domain):
        enc = self.encoder(torch.cat([scaled_log_counts, torch.log1p(library_size)], dim=1))
        mean = self.z_mean(enc)
        logvar = self.z_logvar(enc).clamp(-8, 8)
        if self.training:
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            z = mean
        hidden = self.decoder(torch.cat([z, self.domain(domain)], dim=1))
        proportions = torch.softmax(self.mean_logits(hidden), dim=1)
        mu = proportions * library_size.clamp_min(1.0)
        dropout = self.dropout_logits(hidden)
        theta = torch.nn.functional.softplus(self.log_dispersion) + 1e-4
        return mean, logvar, mu, theta, dropout, self.classifier(mean)


def zinb_nll(counts, mu, theta, dropout_logits):
    theta = theta[None, :]
    log_nb = (
        torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
        + theta * (torch.log(theta) - torch.log(theta + mu + 1e-8))
        + counts * (torch.log(mu + 1e-8) - torch.log(theta + mu + 1e-8))
    )
    log_nonzero = torch.nn.functional.logsigmoid(-dropout_logits) + log_nb
    log_zero = torch.logaddexp(
        torch.nn.functional.logsigmoid(dropout_logits),
        torch.nn.functional.logsigmoid(-dropout_logits) + log_nb,
    )
    return -torch.where(counts == 0, log_zero, log_nonzero).mean()


def main() -> None:
    if DEVICE.type != "mps":
        raise RuntimeError("MPS unavailable; run outside the sandbox on Apple Silicon")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    counts_train, meta_train, counts_test, meta_test = F.load_challenge()
    y_text = meta_train[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y_text))
    class_array = np.asarray(classes)
    class_index = {name: i for i, name in enumerate(classes)}
    y_codes = np.asarray([class_index[name] for name in y_text], np.int64)
    train, valid = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    ).split(counts_train, y_codes))

    expression, atlas_labels, _, challenge = I9A.load_atlas(
        list(counts_train.columns), meta_train, meta_test
    )
    usable = (~challenge) & np.isin(atlas_labels, classes) & (expression.sum(1) > 0)
    atlas_rows = np.flatnonzero(usable)
    atlas_rows, _ = train_test_split(
        atlas_rows, train_size=ATLAS_SAMPLE, random_state=SEED,
        stratify=atlas_labels[atlas_rows],
    )
    atlas_counts = expression[atlas_rows].astype(np.float32)
    atlas_y = np.asarray([class_index[name] for name in atlas_labels[atlas_rows]], np.int64)

    challenge_counts = np.vstack([
        counts_train.to_numpy(np.float32), counts_test.to_numpy(np.float32)
    ])
    # Atlas labelled + challenge training labelled + validation/test reconstruction-only.
    counts = np.vstack([atlas_counts, challenge_counts]).astype(np.float32)
    labels = np.full(len(counts), -1, np.int64)
    weights = np.zeros(len(counts), np.float32)
    labels[:len(atlas_counts)] = atlas_y
    weights[:len(atlas_counts)] = 0.15
    offset = len(atlas_counts)
    labels[offset + train] = y_codes[train]
    weights[offset + train] = 1.0
    domain = np.concatenate([
        np.zeros(len(atlas_counts), np.int64), np.ones(len(challenge_counts), np.int64)
    ])
    library = counts.sum(1, keepdims=True).astype(np.float32)
    log_counts = np.log1p(counts)
    mean = log_counts.mean(0, keepdims=True)
    std = log_counts.std(0, keepdims=True) + 1e-4
    scaled = ((log_counts - mean) / std).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(counts), torch.from_numpy(scaled), torch.from_numpy(library),
        torch.from_numpy(domain), torch.from_numpy(labels), torch.from_numpy(weights),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    model = ZINBCVAE(200, len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"device={DEVICE} atlas={len(atlas_counts)} total={len(counts)} "
          f"labelled={(labels >= 0).sum()} split={len(train)}/{len(valid)}", flush=True)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for raw, x, lib, batch, target, weight in loader:
            raw, x, lib = raw.to(DEVICE), x.to(DEVICE), lib.to(DEVICE)
            batch, target, weight = batch.to(DEVICE), target.to(DEVICE), weight.to(DEVICE)
            z_mean, logvar, mu, theta, dropout, logits = model(x, lib, batch)
            reconstruction = zinb_nll(raw, mu, theta, dropout)
            kl = -0.5 * (1 + logvar - z_mean.square() - logvar.exp()).mean()
            supervised = target >= 0
            ce = torch.nn.functional.cross_entropy(
                logits[supervised], target[supervised], reduction="none"
            )
            classification = (ce * weight[supervised]).sum() / weight[supervised].sum()
            loss = reconstruction + 0.05 * kl + 2.0 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        if epoch in {0, 29, 59, 89, EPOCHS - 1}:
            print(f"epoch {epoch+1:3d}/{EPOCHS} loss={running/len(loader):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    model.eval()
    valid_rows = offset + valid
    with torch.no_grad():
        _, _, _, _, _, logits = model(
            torch.from_numpy(scaled[valid_rows]).to(DEVICE),
            torch.from_numpy(library[valid_rows]).to(DEVICE),
            torch.ones(len(valid_rows), dtype=torch.long, device=DEVICE),
        )
        neural = torch.softmax(logits, dim=1).cpu().numpy()

    meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])
    stack = current_stack(meta_all, classes, list(counts_train.columns))
    et = M.fit_extra_trees(
        stack[train], pd.Series(y_text[train]), classes, stack[valid], seeds=tuple(range(5))
    )
    et = M.correct_prior(et, M.prior_vector(pd.Series(y_text[train]), classes), ALPHA)
    blend = (1 - BLEND_WEIGHT) * et + BLEND_WEIGHT * neural
    truth = y_text[valid]
    et_ok = class_array[et.argmax(1)] == truth
    neural_ok = class_array[neural.argmax(1)] == truth
    blend_ok = class_array[blend.argmax(1)] == truth
    p_value, _ = M.paired_mcnemar(blend_ok, et_ok)
    wins = int((blend_ok & ~et_ok).sum())
    losses = int((et_ok & ~blend_ok).sum())
    print(f"ExtraTrees={et_ok.mean():.4f}", flush=True)
    print(f"ZINB-CVAE={neural_ok.mean():.4f}", flush=True)
    print(f"0.80 ET + 0.20 ZINB={blend_ok.mean():.4f} "
          f"gain={100*(blend_ok.mean()-et_ok.mean()):+.2f}pt "
          f"{wins}w/{losses}l p={p_value:.5g}", flush=True)
    passed = neural_ok.mean() >= 0.78 and blend_ok.mean() - et_ok.mean() > 0.003
    print("VERDICT: " + ("ADVANCE TO FULL CV" if passed else "REJECT"), flush=True)
    np.savez_compressed(OUT / "zinb_cvae_gate.npz", valid=valid, et=et, zinb=neural,
                        y=truth, classes=class_array)


if __name__ == "__main__":
    main()
