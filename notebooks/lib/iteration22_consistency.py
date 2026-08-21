"""Reference network regularised on the unlabelled challenge cells.

Iteration 14's self-training failed because it fed hard pseudo-labels back into the
classifier.  This is the other semi-supervised family: the network is trained on the
labelled atlas as usual, plus an unsupervised term that asks its prediction for a
*challenge* cell to be stable under input perturbation and low-entropy.  No label of any
challenge cell is read - only the released 200 genes and metadata of cells we are asked to
predict, which are given.  It moves the decision boundary out of dense regions of the
challenge distribution rather than memorising its own guesses.
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


def build(name="atlascons", seeds=(0, 1, 2, 3), hidden=(1024, 512), epochs=55,
          dropout=0.25, lam=0.3, noise=0.15, ent=0.05):
    import torch, torch.nn as nn
    dest = OUT / f"{name}.npz"
    if dest.exists():
        print(f"{name}: cached"); return
    Xa, ya, Xc, data = L.design(
        drop_label_meta=VARIANTS.get(name, {}).get("drop_label_meta", False))
    classes, y = data["classes"], data["y"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev); yt = torch.tensor(ya, device=dev)
    Xu = torch.tensor(Xc, device=dev)            # all 10,000 challenge cells, no labels
    n, bs, ubs = len(Xt), 4096, 1024
    acc = np.zeros((len(Xc), len(classes)), np.float32)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
    for s in seeds:
        t0 = time.time(); torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        net = nn.Sequential(*layers, nn.Linear(d, len(classes))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = epochs * ((n + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        step = 0
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            ramp = min(1.0, ep / max(epochs * 0.4, 1))     # warm up the unsupervised term
            net.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                uidx = torch.randint(0, len(Xu), (ubs,), device=dev)
                xb = Xu[uidx]
                p1 = torch.softmax(net(xb + noise * torch.randn_like(xb)), 1)
                p2 = torch.softmax(net(xb + noise * torch.randn_like(xb)), 1)
                cons = ((p1 - p2) ** 2).sum(1).mean()
                entropy = -(p1 * torch.log(p1.clamp_min(1e-8))).sum(1).mean()
                loss = loss + ramp * (lam * cons + ent * entropy)
                loss.backward(); opt.step(); sched.step(); step += 1
        net.eval()
        with torch.no_grad():
            acc += np.vstack([torch.softmax(net(Xu[i:i + 8192]), 1).cpu().numpy()
                              for i in range(0, len(Xu), 8192)])
        print(f"  seed {s} ({time.time()-t0:.0f}s)", flush=True)
    probs = acc / len(seeds)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    np.savez_compressed(dest, probs=probs.astype(np.float32), classes=classes)
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    pred = classes[probs[:len(y)].argmax(1)]
    print(f"{name}: standalone {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f})")


VARIANTS = {
    "atlascons": dict(seeds=(0, 1, 2, 3), lam=0.3, noise=0.15, ent=0.05),
    "atlascons2": dict(seeds=(4, 5, 6, 7, 8, 9), lam=0.6, noise=0.25, ent=0.10,
                       epochs=70),
    "atlascons_md": dict(seeds=(10, 11, 12, 13), lam=0.4, noise=0.2, ent=0.05,
                         drop_label_meta=True),
}


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["atlascons"]):
        build(nm, **{k: v for k, v in VARIANTS[nm].items() if k != "drop_label_meta"})
