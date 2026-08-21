"""Shared infrastructure for Iteration 22's selective correction router.

The router never sees recovered test truth.  It consumes the exact 40-expert
Iteration-21 manifest and proposes a small set of alternative labels.  A second-stage
model estimates whether an alternative beats the frozen pool, and may abstain rather
than rewriting a correct call.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B


OUT = Path("outputs/iteration22/router")
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("outputs/iteration18/freeze_manifest.json")
PARTITIONS = (18, 41, 59, 83)
EPS = 1e-9
K_ALT = 5


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("candidate") != "it21" or len(manifest["experts"]) != 40:
        raise RuntimeError("Iteration-21 40-expert manifest is not active")
    return manifest


def _manifest_arrays(classes: np.ndarray) -> tuple[list[str], dict[str, tuple[np.ndarray, float]]]:
    manifest = load_manifest()
    names = list(manifest["experts"])
    fits = {}
    for branch in ("glia", "neuron"):
        w = np.asarray(manifest["exponents"][branch], dtype=np.float64)
        if len(w) != len(names):
            raise ValueError("manifest exponent length mismatch")
        fits[branch] = (w, float(manifest["prior_exponent"][branch]))
    return names, fits


def load_experts(seed: int | str) -> dict:
    """Load an OOF partition, or the full-fit test expert bank."""
    data = B.load_all()
    if seed == "test":
        raw = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
        y = None
        meta = data["meta_test"]
    else:
        raw = np.load(B.OUT / f"experts_oof_seed{int(seed)}.npz", allow_pickle=True)
        y = raw["y"].astype(str)
        meta = data["meta_train"]
    classes = raw["classes"].astype(str)
    names, fits = _manifest_arrays(classes)
    missing = [n for n in names if n not in raw.files]
    if missing:
        raise ValueError(f"missing experts: {missing}")
    probs = np.stack([np.maximum(raw[n].astype(np.float64), EPS) for n in names])
    probs /= np.maximum(probs.sum(2, keepdims=True), EPS)
    allow = raw["allow"].astype(bool)
    return dict(data=data, names=names, fits=fits, probs=probs, allow=allow,
                y=y, classes=classes, meta=meta)


def pool_logits(bank: dict) -> np.ndarray:
    """Reconstruct the exact frozen Iteration-21 branch-specific log pool."""
    y_prior = bank["data"]["y"]
    classes = bank["classes"]
    prior = pd.Series(y_prior).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    lp = np.log(prior)
    logs = np.log(bank["probs"])
    glia = bank["meta"]["Region"].isna().to_numpy()
    z = np.empty((logs.shape[1], logs.shape[2]), dtype=np.float64)
    for branch, rows in (("glia", glia), ("neuron", ~glia)):
        w, a = bank["fits"][branch]
        z[rows] = np.tensordot(w, logs[:, rows], axes=(0, 0)) - a * lp[None, :]
    z[~bank["allow"]] = -1e9
    return z


def _rank_desc(a: np.ndarray) -> np.ndarray:
    order = np.argsort(-a, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(a.shape[1])[None, :], axis=1)
    return rank


def proposals(bank: dict, z: np.ndarray, k: int = K_ALT) -> tuple[np.ndarray, np.ndarray]:
    """Propose alternatives from pool rank and expert plurality, without labels."""
    n, c = z.shape
    base = z.argmax(1)
    pool_order = np.argsort(-z, axis=1)
    expert_pred = np.where(bank["allow"][None], bank["probs"], -1).argmax(2)
    votes = np.zeros((n, c), dtype=np.float64)
    for m in range(expert_pred.shape[0]):
        votes[np.arange(n), expert_pred[m]] += 1.0
    votes /= expert_pred.shape[0]

    # A small reference-family boost surfaces a biologically independent challenger
    # when the pool's top few classes are nearly identical challenge-side variants.
    ref = np.array([i for i, name in enumerate(bank["names"])
                    if name.startswith("atlas") or name.startswith("sni")])
    ref_votes = np.zeros_like(votes)
    for m in ref:
        ref_votes[np.arange(n), expert_pred[m]] += 1.0 / max(len(ref), 1)

    ppool = softmax(z, axis=1)
    strength = ppool + 0.30 * votes + 0.12 * ref_votes
    strength[np.arange(n), base] = -np.inf
    strength[~bank["allow"]] = -np.inf

    cand = np.empty((n, k), dtype=np.int64)
    source = np.zeros((n, k, 3), dtype=np.float32)
    vote_order = np.argsort(-votes, axis=1)
    ref_order = np.argsort(-ref_votes, axis=1)
    for i in range(n):
        seq = list(pool_order[i, 1:5]) + list(vote_order[i, :3]) + list(ref_order[i, :2])
        seq += list(np.argsort(-strength[i]))
        chosen = []
        for v in seq:
            v = int(v)
            if v == int(base[i]) or not bank["allow"][i, v] or v in chosen:
                continue
            chosen.append(v)
            if len(chosen) == k:
                break
        # The metadata compatibility mask is intentionally sharp: 842 cells have
        # only one admissible class.  Padding with the incumbent preserves a fixed
        # tensor shape and makes those rows an automatic abstention.
        chosen += [int(base[i])] * (k - len(chosen))
        cand[i] = chosen
        for j, v in enumerate(chosen):
            source[i, j, 0] = float(v in pool_order[i, 1:5])
            source[i, j, 1] = float(v in vote_order[i, :3])
            source[i, j, 2] = float(v in ref_order[i, :2])
    return cand, source


def _hierarchy_features(base: np.ndarray, cand: np.ndarray) -> np.ndarray:
    path = B.OUT / "hierarchy_maps.npz"
    d = np.load(path, allow_pickle=True)
    out = []
    for key in ("r1", "r2", "lam", "nt", "mk"):
        a = d[key].astype(str)
        out.append((a[base][:, None] == a[cand]).astype(np.float32))
    return np.stack(out, axis=2)


def build_features(bank: dict, z: np.ndarray, cand: np.ndarray,
                   source: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build label-free per-(cell, alternative) routing features."""
    probs = bank["probs"]
    n_models, n, c = probs.shape
    k = cand.shape[1]
    rows = np.arange(n)
    base = z.argmax(1)
    ppool = softmax(z, axis=1)
    pool_rank = _rank_desc(z)
    entropy = -(ppool * np.log(np.maximum(ppool, EPS))).sum(1)
    pool_order = np.argsort(-z, axis=1)
    margin = z[rows, pool_order[:, 0]] - z[rows, pool_order[:, 1]]

    pred = np.where(bank["allow"][None], probs, -1).argmax(2)
    votes = np.zeros((n, c), np.float32)
    for m in range(n_models):
        votes[rows, pred[m]] += 1.0 / n_models

    # Family-level vote differences are more stable than individual expert IDs.
    family_rules = {
        "challenge_tree": lambda s: s.startswith("et") or s in ("rf", "xgb", "xgbaug"),
        "atlas": lambda s: s.startswith("atlas"),
        "neural": lambda s: "nn" in s or s in ("mlp",),
        "linear": lambda s: "lin" in s or s in ("logit", "atlaslr"),
        "spatial_ref": lambda s: s.startswith("sni") or s in ("knnp", "meta", "meta2"),
    }
    family = []
    family_names = []
    for fname, rule in family_rules.items():
        idx = [m for m, name in enumerate(bank["names"]) if rule(name)]
        fv = np.zeros((n, c), np.float32)
        for m in idx:
            fv[rows, pred[m]] += 1.0 / max(len(idx), 1)
        family.append(fv)
        family_names.append(fname)

    flat_cand = cand.reshape(-1)
    rep_rows = np.repeat(rows, k)
    rep_base = np.repeat(base, k)
    p_c = probs[:, rep_rows, flat_cand]
    p_b = probs[:, rep_rows, rep_base]
    logodds = np.log(np.maximum(p_c, EPS)) - np.log(np.maximum(p_b, EPS))

    simple = np.column_stack([
        ppool[rep_rows, rep_base],
        ppool[rep_rows, flat_cand],
        z[rep_rows, flat_cand] - z[rep_rows, rep_base],
        pool_rank[rep_rows, flat_cand] / max(c - 1, 1),
        np.repeat(margin, k),
        np.repeat(entropy, k),
        votes[rep_rows, rep_base],
        votes[rep_rows, flat_cand],
        votes[rep_rows, flat_cand] - votes[rep_rows, rep_base],
        p_b.mean(0), p_c.mean(0), np.median(p_b, axis=0), np.median(p_c, axis=0),
        p_c.std(0), logodds.mean(0), logodds.std(0),
    ]).astype(np.float32)
    names = ["pool_base", "pool_cand", "pool_logit_delta", "pool_cand_rank",
             "pool_margin", "pool_entropy", "vote_base", "vote_cand", "vote_delta",
             "expert_mean_base", "expert_mean_cand", "expert_median_base",
             "expert_median_cand", "expert_sd_cand", "logodds_mean", "logodds_sd"]

    family_x = np.column_stack([
        fv[rep_rows, flat_cand] - fv[rep_rows, rep_base] for fv in family
    ]).astype(np.float32)
    names += [f"family_delta_{v}" for v in family_names]

    counts = (bank["data"]["counts_train"] if bank["y"] is not None
              else bank["data"]["counts_test"]).to_numpy()
    meta = bank["meta"]
    qc = np.column_stack([
        np.log1p(counts.sum(1)), (counts > 0).sum(1),
        np.log1p(meta["volume"].to_numpy()), meta["Region"].isna().to_numpy(),
    ]).astype(np.float32)
    qc = np.repeat(qc, k, axis=0)
    names += ["log_depth", "genes_detected", "log_volume", "glia"]

    prior = pd.Series(bank["data"]["y"]).value_counts(normalize=True).reindex(
        bank["classes"]).fillna(EPS).to_numpy()
    prior_x = np.column_stack([
        np.log(prior[rep_base]), np.log(prior[flat_cand]),
        np.log(prior[flat_cand]) - np.log(prior[rep_base]),
    ]).astype(np.float32)
    names += ["logprior_base", "logprior_cand", "logprior_delta"]

    hierarchy = _hierarchy_features(base, cand).reshape(n * k, -1)
    names += ["same_r1", "same_r2", "same_lamina", "same_nt", "same_marker"]

    # Explicit class identity lets the arbiter learn stable asymmetric confusions.
    onehot = np.zeros((n * k, 2 * c), np.float32)
    onehot[np.arange(n * k), rep_base] = 1.0
    onehot[np.arange(n * k), c + flat_cand] = 1.0
    names += [f"base_{v}" for v in bank["classes"]] + [f"cand_{v}" for v in bank["classes"]]

    x = np.column_stack([
        simple, family_x, qc, prior_x, hierarchy,
        source.reshape(n * k, -1), logodds.T.astype(np.float32), onehot,
    ]).astype(np.float32)
    names += ["source_pool", "source_vote", "source_reference"]
    names += [f"logodds_{v}" for v in bank["names"]]
    if x.shape[1] != len(names):
        raise AssertionError((x.shape, len(names)))
    return x.reshape(n, k, -1), names


def metric_row(name: str, pred: np.ndarray, base: np.ndarray, truth: np.ndarray,
               glia: np.ndarray, score: np.ndarray | None = None) -> dict:
    correct = pred == truth
    base_correct = base == truth
    changed = pred != base
    wins = int((correct & ~base_correct).sum())
    losses = int((~correct & base_correct).sum())
    p = 1.0 if wins + losses == 0 else float(
        binomtest(min(wins, losses), wins + losses, 0.5).pvalue)
    return {
        "candidate": name,
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "cohen_kappa": float(cohen_kappa_score(truth, pred)),
        "glia_accuracy": float(correct[glia].mean()),
        "neuron_accuracy": float(correct[~glia].mean()),
        "coverage": float(changed.mean()),
        "changed": int(changed.sum()),
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
        "mcnemar_p": p,
        "mean_score_changed": (float(np.mean(score[changed]))
                               if score is not None and changed.any() else math.nan),
    }
