"""Scaled semi-supervised reference model: mean teacher on the unlabelled challenge cells.

Iteration 22 showed that the one curve still moving is unsupervised regularisation against
the challenge distribution: a reference network trained only on atlas labels went from
0.8064 to 0.8108 by asking its prediction for an unlabelled challenge cell to be stable
under perturbation.  This scales that idea properly.

  * mean teacher - the consistency target comes from an exponential moving average of the
    student's own weights rather than a second stochastic pass, which is the standard fix
    for the noisy-target problem in the plain Pi-model;
  * perturbation by Gaussian noise AND random feature masking, so consistency is asked
    across a genuinely wider neighbourhood;
  * entropy minimisation on the student, ramped in over the first 40% of training;
  * deep ensembling over many seeds, and prediction from the teacher.

No challenge label is read anywhere: the unlabelled term uses only the released 200 genes
and metadata that the organisers supply for the cells we are asked to predict.
"""
from __future__ import annotations
import copy, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration19_laminae as L

OUT = Path("outputs/iteration23")
OUT.mkdir(parents=True, exist_ok=True)
_CACHE = {}


def _design(drop_label_meta=False):
    key = bool(drop_label_meta)
    if key not in _CACHE:
        _CACHE[key] = L.design(drop_label_meta=drop_label_meta)
    return _CACHE[key]


def train(seeds, hidden=(1024, 512), epochs=70, dropout=0.25, lam=0.6, noise=0.25,
          ent=0.10, mask=0.1, ema=0.999, lr=2e-3, drop_label_meta=False, verbose=True):
    import torch, torch.nn as nn
    Xa, ya, Xc, data = _design(drop_label_meta)
    classes, y = data["classes"], data["y"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xt = torch.tensor(Xa, device=dev); yt = torch.tensor(ya, device=dev)
    Xu = torch.tensor(Xc, device=dev)
    n, bs, ubs = len(Xt), 4096, 1024
    lossf = nn.CrossEntropyLoss(label_smoothing=0.03)
    acc = np.zeros((len(Xc), len(classes)), np.float32)

    def perturb(x):
        out = x + noise * torch.randn_like(x)
        if mask > 0:
            keep = (torch.rand_like(x) > mask).float()
            out = out * keep
        return out

    for s in seeds:
        t0 = time.time(); torch.manual_seed(s)
        layers, d = [], Xa.shape[1]
        for w in hidden:
            layers += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            d = w
        student = nn.Sequential(*layers, nn.Linear(d, len(classes))).to(dev)
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * ((n + bs - 1) // bs))
        for ep in range(epochs):
            ramp = min(1.0, ep / max(epochs * 0.4, 1))
            perm = torch.randperm(n, device=dev)
            student.train(); teacher.train()
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 8:
                    continue
                opt.zero_grad()
                loss = lossf(student(Xt[idx]), yt[idx])
                uidx = torch.randint(0, len(Xu), (ubs,), device=dev)
                xb = Xu[uidx]
                ps = torch.softmax(student(perturb(xb)), 1)
                with torch.no_grad():
                    pt = torch.softmax(teacher(perturb(xb)), 1)
                cons = ((ps - pt) ** 2).sum(1).mean()
                entropy = -(ps * torch.log(ps.clamp_min(1e-8))).sum(1).mean()
                (loss + ramp * (lam * cons + ent * entropy)).backward()
                opt.step(); sched.step()
                with torch.no_grad():
                    m = min(ema, 1.0 - 1.0 / (1 + sched.last_epoch))
                    for tp, sp in zip(teacher.parameters(), student.parameters()):
                        tp.mul_(m).add_(sp.detach(), alpha=1 - m)
                    for tb, sb in zip(teacher.buffers(), student.buffers()):
                        tb.copy_(sb)
        teacher.eval()
        with torch.no_grad():
            acc += np.vstack([torch.softmax(teacher(Xu[i:i + 8192]), 1).cpu().numpy()
                              for i in range(0, len(Xu), 8192)])
        if verbose:
            cur = acc / (list(seeds).index(s) + 1)
            pred = classes[cur[:len(y)].argmax(1)]
            print(f"    seed {s}: running standalone {np.mean(pred == y):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del student, teacher
    probs = acc / len(seeds)
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    return probs, data


def report(name, probs, data):
    classes, y = data["classes"], data["y"]
    neu = ~data["meta_train"]["Region"].isna().to_numpy()
    pred = classes[probs[:len(y)].argmax(1)]
    print(f"{name}: standalone {np.mean(pred == y):.4f} "
          f"(neurons {np.mean(pred[neu]==y[neu]):.4f}, glia {np.mean(pred[~neu]==y[~neu]):.4f})",
          flush=True)
    return float(np.mean(pred == y))


SWEEP = [
    dict(lam=0.6, noise=0.25, ent=0.10, mask=0.10),
    dict(lam=1.5, noise=0.25, ent=0.10, mask=0.10),
    dict(lam=0.6, noise=0.40, ent=0.20, mask=0.20),
    dict(lam=1.5, noise=0.40, ent=0.20, mask=0.20),
    dict(lam=3.0, noise=0.30, ent=0.15, mask=0.15),
    dict(lam=1.5, noise=0.25, ent=0.00, mask=0.10),
]


def sweep():
    rows = []
    for i, cfg in enumerate(SWEEP):
        probs, data = train((100 + i,), verbose=False, **cfg)
        a = report(f"cfg{i} {cfg}", probs, data)
        rows.append((a, i, cfg))
    rows.sort(reverse=True)
    print("\nbest:", rows[0])
    (OUT / "sweep.txt").write_text("\n".join(f"{a:.4f} {c}" for a, _, c in rows))
    return rows[0][2]


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "sweep":
        sweep()
    else:
        cfg = eval((OUT / "best_cfg.txt").read_text()) if (OUT / "best_cfg.txt").exists() \
            else SWEEP[0]
        n = int(sys.argv[1]) if sys.argv[1:] else 16
        probs, data = train(tuple(range(200, 200 + n)), **cfg)
        report(f"meanteacher x{n}", probs, data)
        np.savez_compressed(OUT / "meanteacher.npz", probs=probs.astype(np.float32),
                            classes=data["classes"])
        print(f"wrote {OUT/'meanteacher.npz'}")
