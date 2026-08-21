"""Linear reference model on the Segment-aware design, fine-tuned on challenge cells.

The linear softmax is the strongest parent-atlas family on this design (0.8056 standalone
against 0.7984 for the MLP and 0.7930 for ExtraTrees), so it is the one worth giving the
challenge's own labelled cells.  Atlas rows carry weight 1, challenge rows weight 20, and
the fit is fold-scoped: a fold's model sees only that fold's training cells.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration19_laminae as L

OUT = Path("outputs/iteration19")
CH_WEIGHT = 20.0
SEEDS = (0, 1, 2, 3)
EPOCHS = 40


def _fit_predict(X, yv, wv, Xq, n_class, seeds=SEEDS, epochs=EPOCHS, lr=3e-3):
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev)
    yt = torch.tensor(yv, device=dev)
    wt = torch.tensor(wv, dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xq, device=dev)
    out = np.zeros((len(Xq), n_class), np.float32)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.03, reduction="none")
    n, bs = len(Xt), 4096
    for s in seeds:
        torch.manual_seed(s)
        net = nn.Linear(X.shape[1], n_class).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * ((n + bs - 1) // bs))
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                ((lossf(net(Xt[idx]), yt[idx]) * wt[idx]).mean()).backward()
                opt.step(); sched.step()
        with torch.no_grad():
            out += torch.softmax(net(Xv), 1).cpu().numpy()
    return out / len(seeds)


def run(seed=18, folds=5):
    dest = OUT / f"lamft_oof_seed{seed}.npz"
    if dest.exists():
        print(f"lamft {seed}: cached"); return
    Xa, ya, Xc, data = L.design()
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    yc = np.array([ci[v] for v in y])
    n = len(y)
    out = np.zeros((n, len(classes)), np.float32)
    for k, (fit, val) in enumerate(StratifiedKFold(
            folds, shuffle=True, random_state=seed).split(Xc[:n], y)):
        t0 = time.time()
        X = np.vstack([Xa, Xc[:n][fit]])
        yv = np.concatenate([ya, yc[fit]])
        wv = np.concatenate([np.ones(len(ya)), np.full(len(fit), CH_WEIGHT)])
        out[val] = _fit_predict(X, yv, wv, Xc[:n][val], len(classes))
        print(f"  fold {k+1}/{folds} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(dest, probs=out, classes=classes)
    allow = np.load(B.OUT / f"experts_oof_seed{seed}.npz")["allow"]
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    pred = classes[np.where(allow, out, -1).argmax(1)]
    print(f"lamft partition {seed}: OOF {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f})")


def test_block():
    Xa, ya, Xc, data = L.design()
    classes, y = data["classes"], data["y"]
    ci = {c: i for i, c in enumerate(classes)}
    n = len(y)
    X = np.vstack([Xa, Xc[:n]])
    yv = np.concatenate([ya, np.array([ci[v] for v in y])])
    wv = np.concatenate([np.ones(len(ya)), np.full(n, CH_WEIGHT)])
    out = _fit_predict(X, yv, wv, Xc[n:], len(classes))
    np.savez_compressed(OUT / "lamft.npz", probs=out.astype(np.float32), classes=classes)
    print("wrote lamft.npz")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "test":
        test_block()
    else:
        for s in [int(x) for x in (sys.argv[1:] or [18])]:
            run(seed=s)
