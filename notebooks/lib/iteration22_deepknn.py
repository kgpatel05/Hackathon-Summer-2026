"""Retrieval against the atlas in a learned embedding rather than in gene space.

Iteration 10's hybrid retrieval scored 0.4362 standalone because it matched cells in raw
expression, where 21 transcripts over 200 genes carry almost no metric signal.  The
reference network trained on 136,612 atlas cells embeds those same cells in a space where
the classes are linearly separable; nearest neighbours *there* are a genuinely different,
non-parametric read of the same reference, and cost one forward pass plus a cosine search
on the GPU.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration19_laminae as L

OUT = Path("outputs/iteration19")
KS = (25, 100)


def build(name="atlasknn", seeds=(0, 1, 2), hidden=(1024, 512), epochs=55, dropout=0.25):
    import torch, torch.nn as nn
    dest = OUT / f"{name}.npz"
    if dest.exists():
        print(f"{name}: cached"); return
    Xa, ya, Xc, data = L.design()
    classes, y = data["classes"], data["y"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev)
    yt = torch.tensor(ya, device=dev)
    Xq = torch.tensor(Xc, device=dev)
    n, bs = len(Xt), 4096
    acc = np.zeros((len(Xc), len(classes)), np.float32)
    for s in seeds:
        t0 = time.time(); torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        trunk = nn.Sequential(*layers).to(dev)
        head = nn.Linear(d, len(classes)).to(dev)
        opt = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()),
                                lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, 2e-3, total_steps=epochs * ((n + bs - 1) // bs))
        lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            trunk.train(); head.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad(); lossf(head(trunk(Xt[idx])), yt[idx]).backward()
                opt.step(); sched.step()
        trunk.eval()
        with torch.no_grad():
            def embed(X):
                z = [trunk(X[i:i + 8192]) for i in range(0, len(X), 8192)]
                z = torch.cat(z)
                return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)
            Ea, Eq = embed(Xt), embed(Xq)
            onehot = torch.zeros(len(ya), len(classes), device=dev)
            onehot[torch.arange(len(ya)), yt] = 1.0
            hist = torch.zeros(len(Xc), len(classes), device=dev)
            for i in range(0, len(Eq), 2048):
                sim = Eq[i:i + 2048] @ Ea.T
                for k in KS:
                    v, j = torch.topk(sim, k, dim=1)
                    wgt = torch.softmax(v * 12.0, dim=1)
                    hist[i:i + 2048] += torch.einsum(
                        "bk,bkc->bc", wgt, onehot[j]) / len(KS)
            acc += hist.cpu().numpy()
        print(f"  seed {s} ({time.time()-t0:.0f}s)", flush=True)
        del trunk, head
    probs = acc / len(seeds)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    np.savez_compressed(dest, probs=probs.astype(np.float32), classes=classes)
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    pred = classes[probs[:len(y)].argmax(1)]
    print(f"{name}: standalone {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f})")


if __name__ == "__main__":
    build()
