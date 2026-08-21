"""Iteration 7 - spatial neighbourhood composition from the PUBLIC atlas.

Why every previous spatial attempt failed: the challenge ships ~10,000 cells drawn
from 108 sections that each contain ~1,300 cells. A challenge cell's "nearest
neighbours" inside the challenge file are therefore hundreds of microns away and
carry no information - which is exactly why spatial kNN scored 0.204, BELOW the
majority-class floor, and why the niche-expression block did nothing.

The public atlas has the FULL sections. So for each challenge cell we can look up
its true physical neighbours among the 136,621 non-challenge atlas cells and use
their published labels as a spatial prior.

LEGITIMACY: uses (a) the challenge cell's own center_x/center_y/Section_ID, which
the organisers ship, and (b) public labelled cells that are NOT in the challenge
file. It reads no test label and no withheld gene. Same category as the atlas
transfer block already in iteration 5.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F

OUT = Path("outputs/iteration7"); OUT.mkdir(parents=True, exist_ok=True)

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y)); CID = {c: i for i, c in enumerate(CLASSES)}

with h5py.File(F.PARENT_ATLAS, "r") as h:
    ids = np.array([x.decode() for x in h["obs/_index"][:]])
    def cat(k):
        c = [x.decode() for x in h[f"obs/{k}/categories"][:]]
        return np.array([c[i] if i >= 0 else "nan" for i in h[f"obs/{k}/codes"][:]])
    alab = np.array([F._normalise_label(s) for s in cat("MERFISH cell type annotation")])
    asec = cat("Section ID")
    ax, ay = h["obs/center_x"][:], h["obs/center_y"][:]

pos = {c: i for i, c in enumerate(ids)}
challenge = np.zeros(len(ids), bool)
for idx in [meta_train.index, meta_test.index]:
    challenge[[pos[c] for c in idx.astype(str)]] = True
ref = np.flatnonzero(~challenge & np.isin(alab, CLASSES))
print(f"atlas reference cells (challenge removed): {len(ref)}", flush=True)

# --- density check: why in-challenge spatial features were doomed -------------
ch_per_sec = pd.Series(meta_train["Section_ID"].astype(str)).value_counts()
at_per_sec = pd.Series(asec[ref]).value_counts()
print(f"cells per section  -- challenge train median {ch_per_sec.median():.0f}"
      f" | atlas median {at_per_sec.median():.0f}", flush=True)

sec_ch = {"train": meta_train["Section_ID"].astype(str).to_numpy(),
          "test": meta_test["Section_ID"].astype(str).to_numpy()}
xy_ch = {"train": meta_train[["center_x", "center_y"]].to_numpy(float),
         "test": meta_test[["center_x", "center_y"]].to_numpy(float)}
print("section overlap: train",
      len(set(sec_ch["train"]) & set(asec[ref])), "/", len(set(sec_ch["train"])),
      "| test", len(set(sec_ch["test"]) & set(asec[ref])), "/", len(set(sec_ch["test"])),
      flush=True)

KS = [5, 10, 25, 50]
def neighbour_block(split):
    """Per cell: label histogram over k nearest atlas cells, for several k,
    plus distances. Returns (n, len(KS)*60 + 2*len(KS)) array."""
    n = len(sec_ch[split])
    hist = np.zeros((n, len(KS), len(CLASSES)), np.float32)
    dist = np.zeros((n, len(KS), 2), np.float32)
    by_sec = {}
    for i, s in enumerate(asec[ref]):
        by_sec.setdefault(s, []).append(ref[i])
    for s in np.unique(sec_ch[split]):
        rows = np.flatnonzero(sec_ch[split] == s)
        pool = np.array(by_sec.get(s, []))
        if len(pool) < 5:
            continue
        P = np.column_stack([ax[pool], ay[pool]])
        nn = NearestNeighbors(n_neighbors=min(max(KS), len(pool))).fit(P)
        dd, ii = nn.kneighbors(np.column_stack([xy_ch[split][rows, 0],
                                                xy_ch[split][rows, 1]]))
        codes = np.array([CID[c] for c in alab[pool]])
        for j, k in enumerate(KS):
            kk = min(k, ii.shape[1])
            lab = codes[ii[:, :kk]]
            w = 1.0 / (dd[:, :kk] + 1e-3)
            for c in range(len(CLASSES)):
                hist[rows, j, c] = (w * (lab == c)).sum(1) / w.sum(1)
            dist[rows, j, 0] = dd[:, :kk].mean(1)
            dist[rows, j, 1] = dd[:, kk - 1]
    return np.hstack([hist.reshape(n, -1), dist.reshape(n, -1)]).astype(np.float32)

t0 = time.time()
NB_TR = neighbour_block("train"); NB_TE = neighbour_block("test")
print(f"neighbour block built in {time.time()-t0:.0f}s -> {NB_TR.shape}", flush=True)

# --- how good is the spatial prior ON ITS OWN? -------------------------------
print("\n=== nearest-atlas-neighbour vote alone, on challenge TRAIN labels ===", flush=True)
glia_tr = meta_train["Region"].isna().to_numpy()
for j, k in enumerate(KS):
    pred = np.array(CLASSES)[NB_TR[:, j * len(CLASSES):(j + 1) * len(CLASSES)].argmax(1)]
    print(f"  k={k:3d}  acc={accuracy_score(y, pred):.4f}"
          f"  bal={balanced_accuracy_score(y, pred):.4f}"
          f"  glia={accuracy_score(y[glia_tr], pred[glia_tr]):.4f}"
          f"  neuron={accuracy_score(y[~glia_tr], pred[~glia_tr]):.4f}", flush=True)

np.savez_compressed(OUT / "neighbour_block.npz", NB_TR=NB_TR, NB_TE=NB_TE,
                    classes=np.array(CLASSES), ks=np.array(KS))
print("\nwrote", OUT / "neighbour_block.npz", flush=True)
