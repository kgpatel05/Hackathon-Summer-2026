"""Independent full-OOF confirmation for the fixed Iteration 12 TabM blend.

The seed-557 screen selected one immutable configuration: TabM-16 with linear
ReLU embeddings, five epochs, no posterior prior correction, and a 20% blend
into the adopted 694-feature ExtraTrees model.  This script makes one prediction
per training cell under a fresh five-fold partition (seed 613), with all fitting,
scaling and metadata masks scoped to each fold.

Confirm only for >0.30-point gain and exact paired p < 0.05.  If confirmed, run
``replicate`` for a second fresh partition (seed 991); promotion additionally
requires the replicate gain to exceed 0.30 point.  No test label is read and no
submission is written.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".deps"))
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
from iteration10_spatial_fields import current_stack
from iteration12_tabm import metadata_mask

OUT = ROOT / "outputs/iteration12"
MODE = sys.argv[1] if len(sys.argv) > 1 else "confirm"
if MODE not in {"confirm", "replicate"}:
    raise SystemExit("usage: iteration12_tabm_confirm.py [confirm|replicate]")
FOLD_SEED = 613 if MODE == "confirm" else 991
K = 16
EPOCHS = 5
BATCH_SIZE = 128
BLEND_WEIGHT = 0.20


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(n_features, n_classes, device):
    return TabM.make(
        n_num_features=n_features,
        num_embeddings=LinearReLUEmbeddings(n_features, d_embedding=8),
        d_out=n_classes,
        k=K,
        n_blocks=2,
        d_block=256,
        dropout=0.15,
    ).to(device)


def make_loader(x, y, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)),
                            torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train(model, batches, optimizer, device):
    model.train()
    mean_loss = 0.0
    n = 0
    for xb, yb in batches:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, None)
        targets = yb[:, None].expand(-1, logits.shape[1])
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mean_loss += float(loss.detach()) * len(xb)
        n += len(xb)
    return mean_loss / n


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    output = []
    for xb, _ in batches:
        output.append(torch.softmax(model(xb.to(device), None), dim=-1)
                      .mean(dim=1).cpu().numpy())
    return np.vstack(output).astype(np.float32)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    counts, meta, _, meta_test = F.load_challenge()
    y_text = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y_text))
    class_array = np.asarray(classes)
    index = {label: j for j, label in enumerate(classes)}
    y = np.asarray([index[label] for label in y_text], np.int64)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts.columns))
    folds = StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED)
    et_oof = np.zeros((len(y), len(classes)), np.float32)
    tabm_oof = np.zeros_like(et_oof)
    print(f"mode={MODE} fold_seed={FOLD_SEED} device={device} "
          f"features={x.shape[1]} epochs={EPOCHS} k={K}", flush=True)
    t0 = time.time()

    for fold, (train_rows, valid_rows) in enumerate(folds.split(x, y), 1):
        fold_seed = FOLD_SEED*10 + fold
        et = M.fit_extra_trees(
            x[train_rows], pd.Series(y_text[train_rows]), classes, x[valid_rows],
            seeds=(0, 1, 2, 3, 4),
        )
        et = M.correct_prior(
            et, M.prior_vector(pd.Series(y_text[train_rows]), classes), 0.45
        )
        et_oof[valid_rows] = metadata_mask(
            et, meta.iloc[train_rows], y_text[train_rows], meta.iloc[valid_rows], classes
        )

        scaler = StandardScaler().fit(x[train_rows])
        x_train = scaler.transform(x[train_rows]).astype(np.float32)
        x_valid = scaler.transform(x[valid_rows]).astype(np.float32)
        seed_all(fold_seed)
        model = make_model(x.shape[1], len(classes), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
        train_loader = make_loader(x_train, y[train_rows], True, fold_seed)
        valid_loader = make_loader(x_valid, y[valid_rows], False, fold_seed)
        for epoch in range(EPOCHS):
            loss = train(model, train_loader, optimizer, device)
        probabilities = predict(model, valid_loader, device)
        tabm_oof[valid_rows] = metadata_mask(
            probabilities, meta.iloc[train_rows], y_text[train_rows],
            meta.iloc[valid_rows], classes
        )
        print(f"fold={fold}/5 loss={loss:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        del model, optimizer, train_loader, valid_loader
        if device.type == "mps":
            torch.mps.empty_cache()

    blend = (1-BLEND_WEIGHT)*et_oof + BLEND_WEIGHT*tabm_oof
    glia = meta["Region"].isna().to_numpy()
    base_ok = class_array[et_oof.argmax(axis=1)] == y_text
    rows = []
    for name, probabilities in {
        "masked ExtraTrees incumbent": et_oof,
        "TabM-16": tabm_oof,
        "0.80 ET + 0.20 TabM": blend,
    }.items():
        ok = class_array[probabilities.argmax(axis=1)] == y_text
        if name == "masked ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "glia": ok[glia].mean(), "neurons": ok[~glia].mean(),
                     "wins": wins, "losses": losses, "p": p_value,
                     "fold_seed": FOLD_SEED})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"tabm_{MODE}_oof.csv", index=False)
    np.savez_compressed(OUT / f"tabm_{MODE}_oof.npz", et=et_oof,
                        tabm=tabm_oof, blend=blend, truth=y_text,
                        classes=class_array, fold_seed=FOLD_SEED)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    candidate = rows[2]
    passed = (candidate["gain_pt"] > 0.30
              and (candidate["p"] < 0.05 if MODE == "confirm" else True))
    print("VERDICT: " + ("CONFIRMED" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
