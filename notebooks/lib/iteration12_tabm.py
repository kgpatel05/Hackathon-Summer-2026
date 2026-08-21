"""Iteration 12 - TabM parameter-efficient neural ensemble on Apple MPS.

TabM (ICLR 2025, Apache-2.0) jointly trains several MLP predictors with shared
weights.  This differs from the single MLPs already rejected in earlier rounds:
the ensemble is optimized as an ensemble while retaining model-specific scaling
parameters.  The fixed configuration follows the official package defaults with
the smaller k=16 exploration setting recommended for bounded experiments.

Epoch count and posterior prior correction are selected on a 10% split wholly
inside the frozen seed-557 outer training partition.  The selected epoch is then
retrained on all 4,000 outer-training rows before the 1,000 outer rows are scored.
Advance only if TabM is >=78% standalone and a fixed 80/20 ET/TabM blend gains
>0.30 point.  No test label is read and no submission is written.
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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

OUT = ROOT / "outputs/iteration12"
OUT.mkdir(parents=True, exist_ok=True)
GATE = OUT / "tabicl_gate.npz"
SEED = 557
K = 16
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 12
ALPHAS = (0.0, 0.20, 0.45)
BLEND_WEIGHT = 0.20


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(n_features: int, n_classes: int, device: torch.device) -> TabM:
    model = TabM.make(
        n_num_features=n_features,
        num_embeddings=LinearReLUEmbeddings(n_features, d_embedding=8),
        d_out=n_classes,
        k=K,
        n_blocks=2,
        d_block=256,
        dropout=0.15,
    )
    return model.to(device)


def loader(x: np.ndarray, y: np.ndarray, *, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)),
                            torch.from_numpy(y.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=generator, num_workers=0)


def train_epoch(model, batches, optimizer, device):
    model.train()
    loss_sum = 0.0
    n = 0
    for xb, yb in batches:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, None)                    # (B, k, classes)
        targets = yb[:, None].expand(-1, logits.shape[1])
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(xb)
        n += len(xb)
    return loss_sum / n


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    outputs = []
    for xb, _ in batches:
        probabilities = torch.softmax(model(xb.to(device), None), dim=-1).mean(dim=1)
        outputs.append(probabilities.cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def correct_prior(probabilities, y_fit, classes, alpha):
    prior = M.prior_vector(pd.Series(y_fit), classes)
    return M.correct_prior(probabilities, prior, alpha)


def metadata_mask(probabilities, fit_meta, fit_y, eval_meta, classes):
    allow = np.ones_like(probabilities, dtype=bool)
    for column in ("Region", "Excitatory_vs_Inhibitory", "Segment"):
        fit_values = fit_meta[column].astype(str).to_numpy()
        eval_values = eval_meta[column].astype(str).to_numpy()
        known = set(fit_values)
        seen = [set(fit_values[fit_y == label]) for label in classes]
        for i, value in enumerate(eval_values):
            if value in known:
                allow[i] &= np.asarray([value in values for values in seen])
    allow[~allow.any(axis=1)] = True
    out = np.where(allow, probabilities, 0.0)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def main() -> None:
    seed_all(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    counts, meta, _, meta_test = F.load_challenge()
    y_text = meta[F.TARGET].astype(str).to_numpy()
    classes = sorted(set(y_text))
    class_array = np.asarray(classes)
    class_index = {label: j for j, label in enumerate(classes)}
    y = np.asarray([class_index[label] for label in y_text], np.int64)
    gate = np.load(GATE, allow_pickle=True)
    outer_valid = gate["valid"].astype(int)
    outer_train = np.setdiff1d(np.arange(len(y)), outer_valid)
    meta_all = pd.concat([meta.drop(columns=[F.TARGET]), meta_test])
    x = current_stack(meta_all, classes, list(counts.columns))
    fit_rows, early_rows = train_test_split(
        outer_train, test_size=0.10, random_state=SEED, stratify=y[outer_train]
    )
    print(f"device={device} outer={len(outer_train)}/{len(outer_valid)} "
          f"early={len(fit_rows)}/{len(early_rows)} features={x.shape[1]} k={K}", flush=True)

    scaler = StandardScaler().fit(x[fit_rows])
    x_fit = scaler.transform(x[fit_rows]).astype(np.float32)
    x_early = scaler.transform(x[early_rows]).astype(np.float32)
    fit_loader = loader(x_fit, y[fit_rows], shuffle=True, seed=SEED)
    early_loader = loader(x_early, y[early_rows], shuffle=False, seed=SEED)
    model = make_model(x.shape[1], len(classes), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)

    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, fit_loader, optimizer, device)
        raw = predict(model, early_loader, device)
        accuracy = np.mean(class_array[raw.argmax(axis=1)] == y_text[early_rows])
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = float(accuracy)
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={loss:.4f} early_acc={accuracy:.4f} "
              f"best={best_accuracy:.4f}@{best_epoch}", flush=True)
        if stale >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    raw_early = predict(model, early_loader, device)
    alpha_rows = []
    for alpha in ALPHAS:
        probabilities = correct_prior(raw_early, y_text[fit_rows], classes, alpha)
        probabilities = metadata_mask(
            probabilities, meta.iloc[fit_rows], y_text[fit_rows], meta.iloc[early_rows], classes
        )
        alpha_rows.append({"alpha": alpha, "accuracy": np.mean(
            class_array[probabilities.argmax(axis=1)] == y_text[early_rows]
        )})
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame.to_csv(OUT / "tabm_inner_alpha.csv", index=False)
    selected_alpha = float(alpha_frame.sort_values(
        ["accuracy", "alpha"], ascending=[False, True], kind="stable"
    ).iloc[0]["alpha"])
    print(alpha_frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"selected epoch={best_epoch} alpha={selected_alpha:g} in {time.time()-t0:.1f}s",
          flush=True)

    # Refit from scratch on every outer-training label for exactly the selected epoch.
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    seed_all(SEED)
    full_scaler = StandardScaler().fit(x[outer_train])
    x_train = full_scaler.transform(x[outer_train]).astype(np.float32)
    x_valid = full_scaler.transform(x[outer_valid]).astype(np.float32)
    full_loader = loader(x_train, y[outer_train], shuffle=True, seed=SEED)
    valid_loader = loader(x_valid, y[outer_valid], shuffle=False, seed=SEED)
    model = make_model(x.shape[1], len(classes), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(model, full_loader, optimizer, device)
        print(f"refit epoch={epoch:03d}/{best_epoch} loss={loss:.4f}", flush=True)
    tabm_prob = predict(model, valid_loader, device)
    tabm_prob = correct_prior(tabm_prob, y_text[outer_train], classes, selected_alpha)
    tabm_prob = metadata_mask(
        tabm_prob, meta.iloc[outer_train], y_text[outer_train],
        meta.iloc[outer_valid], classes
    )
    et = metadata_mask(
        gate["et"].astype(np.float32), meta.iloc[outer_train], y_text[outer_train],
        meta.iloc[outer_valid], classes
    )
    blend = (1-BLEND_WEIGHT)*et + BLEND_WEIGHT*tabm_prob
    truth = y_text[outer_valid]
    base_ok = class_array[et.argmax(axis=1)] == truth
    rows = []
    for name, probabilities in {
        "masked ExtraTrees incumbent": et,
        "TabM-16": tabm_prob,
        "0.80 ET + 0.20 TabM": blend,
    }.items():
        ok = class_array[probabilities.argmax(axis=1)] == truth
        if name == "masked ExtraTrees incumbent":
            p_value, wins, losses = 1.0, 0, 0
        else:
            p_value, _ = M.paired_mcnemar(ok, base_ok)
            wins = int((ok & ~base_ok).sum())
            losses = int((base_ok & ~ok).sum())
        rows.append({"config": name, "accuracy": ok.mean(),
                     "gain_pt": 100*(ok.mean()-base_ok.mean()),
                     "wins": wins, "losses": losses, "p": p_value,
                     "epoch": best_epoch, "alpha": selected_alpha})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tabm_gate.csv", index=False)
    np.savez_compressed(OUT / "tabm_gate.npz", valid=outer_valid, et=et,
                        tabm=tabm_prob, truth=truth, classes=class_array,
                        epoch=best_epoch, alpha=selected_alpha)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    passed = rows[1]["accuracy"] >= 0.78 and rows[2]["gain_pt"] > 0.30
    print("VERDICT: " + ("ADVANCE TO FULL OOF" if passed else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
