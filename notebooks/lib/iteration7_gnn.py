"""Iteration 7 - does a graph neural network add anything?

A GNN is two things: a node encoder, and neighbourhood aggregation. Both have already
been measured separately in this project, and both came back weak:

  * aggregation over the physical graph: spatial kNN scored 0.1102-0.1230 (below the
    0.141 majority floor); nearest-neighbour voting rebuilt against the FULL parent
    atlas scored 0.2164; the hand-built niche block (mean expression of the 15 spatial
    neighbours, PCA'd - i.e. exactly one mean-aggregation layer) was worth +0.04 pt.
  * the node encoder: ten model classes were probed on 86k atlas glia and every one
    landed in 0.65-0.71; MLPs did not beat logistic regression.

So the prior is that a GNN cannot help. But "I already measured the parts" is an
argument, not a measurement, and the parts could compose better than they behave
alone - a learned aggregation is strictly more expressive than a mean, and end-to-end
training could find a neighbourhood statistic that PCA-of-the-mean throws away.

This tests it directly, with the controls that make the answer trustworthy:

  MLP (no graph)      isolates the encoder. If SAGE ~ MLP, the graph adds nothing and
                      the architecture is irrelevant.
  SAGE / spatial      the physical neighbour graph, k=8 within Section_ID.
  SAGE / expression   kNN in PCA(50) of log-CPM - the "cells like me" graph.
  SAGE / random       NULL CONTROL. k=8 random edges. Any gain a real graph shows must
                      beat this, or it is just the extra parameters and the smoothing.
  SAGE-3L / expr      3 layers with residuals, in case one hop is too shallow.

All graphs span all 10,000 cells (train + test). That is transductive but legitimate:
edges are built from coordinates and expression only, never from labels, and the loss
is taken only on the fold's training nodes.

PRE-REGISTERED DECISION RULE, fixed before running:
  a GNN is worth pursuing only if some variant beats BOTH the submitted Extra Trees
  and the random-graph control, on identical folds, at p < 0.05 paired McNemar.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M
import torch_device as TD

OUT = Path("outputs/iteration7")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
# MPS has no sparse-COO kernel, so neighbour aggregation is a gather + mean instead of
# torch.sparse.mm. See notebooks/lib/torch_device.py.
DEVICE = TD.get_device()
K = 8
SEEDS = (0, 1, 2)
EPOCHS = 400

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
y_idx = np.searchsorted(CLASS_ARR, y)
glia = meta_train["Region"].isna().to_numpy()
meta_all = pd.concat([meta_train.drop(columns=[F.TARGET]), meta_test])

c = np.load("outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
            allow_pickle=True)
X_TR = np.hstack([c["BASE_TR"], c["EXT_TR"], c["SPA_TR"], c["NIC_TR"], c["ATL_TR"]])
X_TE = np.hstack([c["BASE_TE"], c["EXT_TE"], c["SPA_TE"], c["NIC_TE"], c["ATL_TE"]])
X_ALL = np.vstack([X_TR, X_TE]).astype(np.float32)
X_ALL = StandardScaler().fit_transform(X_ALL).astype(np.float32)
N, D = X_ALL.shape
print(f"nodes={N} features={D} classes={len(CLASSES)}", flush=True)


# ------------------------------------------------------------------ graphs
def normalise(edges, n):
    """Padded neighbour index + mask on DEVICE - the MPS-compatible form of mean
    aggregation. Equivalent to a row-normalised adjacency, without sparse tensors."""
    src, dst = edges
    index, mask = TD.pad_neighbours(np.asarray(src), np.asarray(dst), n)
    return (torch.tensor(index, device=DEVICE),
            torch.tensor(mask, device=DEVICE))


def knn_edges(coords, k, groups=None):
    src, dst = [], []
    blocks = [np.arange(len(coords))] if groups is None else \
        [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for rows in blocks:
        k_eff = min(k + 1, len(rows))
        if k_eff < 2:
            continue
        _, nn = NearestNeighbors(n_neighbors=k_eff).fit(coords[rows]).kneighbors(coords[rows])
        for j in range(1, k_eff):                      # column 0 is the cell itself
            src.append(rows); dst.append(rows[nn[:, j]])
    return np.concatenate(src), np.concatenate(dst)


t0 = time.time()
sections = meta_all["Section_ID"].astype(str).to_numpy()
coords = meta_all[["center_x", "center_y"]].to_numpy(float)
E_SPATIAL = knn_edges(coords, K, groups=sections)

EXPR_PCA = PCA(n_components=50, random_state=0).fit_transform(
    F.log_cpm(np.vstack([counts_train.to_numpy(), counts_test.to_numpy()])))
E_EXPR = knn_edges(EXPR_PCA, K)

rng = np.random.default_rng(0)
E_RANDOM = (np.repeat(np.arange(N), K), rng.integers(0, N, N * K))

GRAPHS = {"spatial": E_SPATIAL, "expression": E_EXPR, "random": E_RANDOM}
ADJ = {k: normalise(e, N) for k, e in GRAPHS.items()}
per_section = pd.Series(sections).value_counts()
print(f"graphs built in {time.time()-t0:.0f}s | sections={per_section.size} "
      f"median cells/section={per_section.median():.0f}", flush=True)
# how often does a spatial edge join two cells of the same class? (labelled cells only)
# chance level for a random pair is sum(p_i^2) over the class marginal
_p = np.bincount(y_idx, minlength=len(CLASSES)) / len(y_idx)
print(f"chance homophily (random pair, sum p_i^2): {(_p ** 2).sum():.4f}", flush=True)
src, dst = E_SPATIAL
lab = (src < 5000) & (dst < 5000)
print(f"spatial edge homophily (both ends labelled, n={lab.sum()}): "
      f"{(y_idx[src[lab]] == y_idx[dst[lab]]).mean():.4f}", flush=True)
src, dst = E_EXPR
lab = (src < 5000) & (dst < 5000)
print(f"expression edge homophily (both ends labelled, n={lab.sum()}): "
      f"{(y_idx[src[lab]] == y_idx[dst[lab]]).mean():.4f}", flush=True)


# ------------------------------------------------------------------ model
class SAGE(nn.Module):
    """GraphSAGE: h' = act(W_self h + W_neigh (A h)). n_layers=0 gives a plain MLP."""

    def __init__(self, d_in, d_hid, d_out, n_layers, dropout=0.3, residual=False):
        super().__init__()
        self.n_layers, self.residual, self.dropout = n_layers, residual, dropout
        self.inp = nn.Linear(d_in, d_hid)
        self.self_w = nn.ModuleList(nn.Linear(d_hid, d_hid) for _ in range(n_layers))
        self.neigh_w = nn.ModuleList(nn.Linear(d_hid, d_hid) for _ in range(n_layers))
        self.norm = nn.ModuleList(nn.LayerNorm(d_hid) for _ in range(n_layers))
        self.out = nn.Linear(d_hid, d_out)

    def forward(self, x, adj):
        h = Fn.relu(self.inp(x))
        h = Fn.dropout(h, self.dropout, self.training)
        for i in range(self.n_layers):
            agg = TD.neighbour_mean(h, adj[0], adj[1])
            z = Fn.relu(self.norm[i](self.self_w[i](h) + self.neigh_w[i](agg)))
            h = h + z if self.residual else z
            h = Fn.dropout(h, self.dropout, self.training)
        return self.out(h)


X_T = torch.from_numpy(X_ALL).to(DEVICE)
Y_T = torch.from_numpy(y_idx.astype(np.int64)).to(DEVICE)


def fit_gnn(train_idx, eval_idx, adj, n_layers, seed, residual=False, d_hid=256):
    """Train on train_idx, return softmax probabilities for eval_idx.

    10% of the fold's training cells are held out for early stopping. That inner split
    never touches the fold's validation cells, so the reported number stays honest.
    """
    torch.manual_seed(seed)
    inner_tr, inner_va = train_test_split(
        train_idx, test_size=0.10, random_state=seed, stratify=y_idx[train_idx])
    model = SAGE(D, d_hid, len(CLASSES), n_layers, residual=residual).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tr_t = torch.from_numpy(inner_tr).to(DEVICE)
    va_t = torch.from_numpy(inner_va).to(DEVICE)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = Fn.cross_entropy(model(X_T, adj)[tr_t], Y_T[tr_t])
        loss.backward(); opt.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_T, adj)[va_t].argmax(1) == Y_T[va_t]).float().mean().item()
            if acc > best:
                best, patience = acc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        out = model(X_T, adj)[torch.from_numpy(eval_idx).to(DEVICE)]
        return Fn.softmax(out, 1).cpu().numpy()


# ------------------------------------------------------------------ evaluation
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7).split(y, y))
prior = M.prior_vector(pd.Series(y), CLASSES)


def evaluate(name, run_fold):
    t0 = time.time()
    ok = np.zeros(5000, bool)
    probs = np.zeros((5000, len(CLASSES)), np.float32)
    for tr, va in folds:
        p = run_fold(tr, va)
        p = M.correct_prior(p, M.prior_vector(pd.Series(y[tr]), CLASSES), 0.45)
        probs[va] = p
        ok[va] = CLASS_ARR[p.argmax(1)] == y[va]
    print(f"  {name:28s} acc={ok.mean():.4f} glia={ok[glia].mean():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, probs


print("\n=== 5-fold CV on the 5,000 labelled cells ===", flush=True)
results, prob_store = {}, {}

results["ET (submitted)"], prob_store["ET (submitted)"] = evaluate(
    "ET (submitted)",
    lambda tr, va: M.fit_extra_trees(X_TR[tr], pd.Series(y[tr]), CLASSES, X_TR[va],
                                     seeds=tuple(range(5))))

CONFIGS = [
    ("MLP (no graph)", "spatial", 0, False),      # adj unused when n_layers=0
    ("SAGE / spatial", "spatial", 2, False),
    ("SAGE / expression", "expression", 2, False),
    ("SAGE / random (null)", "random", 2, False),
    ("SAGE-3L / expression", "expression", 3, True),
]
for name, graph, layers, res in CONFIGS:
    adj = ADJ[graph]
    results[name], prob_store[name] = evaluate(
        name,
        lambda tr, va, a=adj, L=layers, R=res: np.mean(
            [fit_gnn(tr, va, a, L, s, residual=R) for s in SEEDS], axis=0))

# ------------------------------------------------------------------ blending
print("\n=== blending the best GNN into the ET probabilities ===", flush=True)
et_p = prob_store["ET (submitted)"]
best_gnn = max((k for k in prob_store if k != "ET (submitted)"),
               key=lambda k: results[k].mean())
gnn_p = prob_store[best_gnn]
print(f"  best GNN variant: {best_gnn}", flush=True)
blend_rows = []
for w in (0.0, 0.1, 0.2, 0.3, 0.5):
    ok = CLASS_ARR[((1 - w) * et_p + w * gnn_p).argmax(1)] == y
    blend_rows.append({"weight": w, "accuracy": ok.mean(), "glia": ok[glia].mean()})
    print(f"  w={w:.1f}  acc={ok.mean():.4f}  glia={ok[glia].mean():.4f}", flush=True)

# ------------------------------------------------------------------ significance
print("\n=== paired McNemar (identical folds) ===", flush=True)
rows = []
base_et = results["ET (submitted)"]
base_rand = results["SAGE / random (null)"]
for name, ok in results.items():
    if name == "ET (submitted)":
        continue
    p_et, _ = M.paired_mcnemar(ok, base_et)
    p_rand, _ = M.paired_mcnemar(ok, base_rand)
    beats = (ok.mean() > base_et.mean() and p_et < 0.05
             and ok.mean() > base_rand.mean() and p_rand < 0.05)
    print(f"  {name:24s} vs ET {ok.mean()-base_et.mean():+.4f} p={p_et:.3g} | "
          f"vs null-graph {ok.mean()-base_rand.mean():+.4f} p={p_rand:.3g}"
          f"{'   <== PASSES' if beats else ''}", flush=True)
    rows.append({"model": name, "accuracy": ok.mean(), "glia": ok[glia].mean(),
                 "gain_vs_et": ok.mean() - base_et.mean(), "p_vs_et": p_et,
                 "gain_vs_null_graph": ok.mean() - base_rand.mean(), "p_vs_null": p_rand,
                 "passes_preregistered_rule": beats})

verdict = any(r["passes_preregistered_rule"] for r in rows)
print(f"\n  VERDICT: {'PURSUE' if verdict else 'DO NOT PURSUE'} "
      f"(rule: beat both the ET and the random-graph control at p<0.05)", flush=True)

pd.DataFrame(rows).to_csv(OUT / "gnn.csv", index=False)
pd.DataFrame(blend_rows).to_csv(OUT / "gnn_blend.csv", index=False)
print(f"\nwrote {OUT/'gnn.csv'} and {OUT/'gnn_blend.csv'}", flush=True)
