"""Scale the semi-supervised reference model: deep ensembles of the best configurations."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration23_meanteacher as MT

OUT = MT.OUT

RUNS = {
    # mean teacher without feature masking: isolates the EMA target
    "mt_ema":    dict(lam=0.6, noise=0.25, ent=0.10, mask=0.0,  epochs=70,  n=16),
    # the same, trained longer - consistency methods usually want a long schedule
    "mt_long":   dict(lam=0.6, noise=0.25, ent=0.10, mask=0.0,  epochs=120, n=12),
    # mean teacher with masking, the sweep winner
    "mt_mask":   dict(lam=0.6, noise=0.25, ent=0.10, mask=0.10, epochs=70,  n=12),
    # wider network, longer schedule
    "mt_wide":   dict(lam=0.6, noise=0.25, ent=0.10, mask=0.0,  epochs=110, n=10,
                      hidden=(1536, 768)),
    # metadata-free view for ensemble diversity
    "mt_md":     dict(lam=0.6, noise=0.25, ent=0.10, mask=0.0,  epochs=90,  n=10,
                      drop_label_meta=True),
}


def main(names):
    for nm in names:
        cfg = dict(RUNS[nm])
        n = cfg.pop("n")
        dest = OUT / f"{nm}.npz"
        if dest.exists():
            print(f"{nm}: cached"); continue
        print(f"=== {nm}: {n} seeds, {cfg}", flush=True)
        probs, data = MT.train(tuple(range(300, 300 + n)), verbose=True, **cfg)
        MT.report(nm, probs, data)
        np.savez_compressed(dest, probs=probs.astype(np.float32),
                            classes=data["classes"])


if __name__ == "__main__":
    main(sys.argv[1:] or list(RUNS))
