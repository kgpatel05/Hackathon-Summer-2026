"""Freeze the gated log-pool and write a candidate submission.

Exponents (and their class-frequency / cell-depth interactions) are fitted by out-of-fold
likelihood on the pooled four fold partitions of the released training cells, separately
for the glia and neuron branches.  Recovered test labels are not imported here.
"""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
import iteration18_logpool2 as LP
import iteration20_gated as G

PARTITIONS = (18, 41, 59, 83)
EXCLUDE = ("rank",)
L2_INT = 1e-2          # selected on worst-case two-way held-out accuracy
OUTDIR = B.OUT / "predictions"
EPS = 1e-9


def _depth(counts, ref_counts):
    d = np.log1p(counts.sum(1))
    r = np.log1p(ref_counts.sum(1))
    return ((d - r.mean()) / (r.std() + 1e-9)).astype(np.float64)


def frozen():
    data = B.load_all()
    classes, y = data["classes"], data["y"]
    glia = data["meta_train"]["Region"].isna().to_numpy()
    parts = [LP.load_partition(s) for s in PARTITIONS]
    common = set(parts[0][1])
    for p in parts[1:]:
        common &= set(p[1])
    used = sorted(n for n in common if n not in EXCLUDE)
    logs = np.concatenate(
        [np.stack([p[0][p[1].index(n)] for n in used]) for p in parts], axis=1)
    allow = np.concatenate([p[2] for p in parts], axis=0)
    yy = np.concatenate([p[3] for p in parts])

    counts_tr = data["counts_train"].to_numpy()
    d1 = _depth(counts_tr, counts_tr)
    prior = pd.Series(yy).value_counts(normalize=True).reindex(classes).fillna(
        EPS).to_numpy()
    log_prior = np.log(prior)
    l_c = (log_prior - log_prior.mean()) / (log_prior.std() + 1e-9)
    d_i = np.tile(d1, len(PARTITIONS))
    gl = np.tile(glia, len(PARTITIONS))
    fits = {}
    for tag, rr in (("glia", np.flatnonzero(gl)), ("neuron", np.flatnonzero(~gl))):
        fits[tag] = G.fit(logs, yy, classes, l_c, d_i, log_prior, allow, rows=rr,
                          use_v=True, use_u=True, l2_int=L2_INT)
    return used, fits, classes, l_c, log_prior, data


def main(tag="gated"):
    used, fits, classes, l_c, log_prior, data = frozen()
    for name, (w, v, u, a) in fits.items():
        top = sorted(zip(used, w, v, u), key=lambda t: -t[1])[:8]
        print(f"[{name}] prior exponent a={a:.3f}; top experts "
              + ", ".join(f"{n}(w={ww:.2f},v={vv:+.2f},u={uu:+.2f})"
                          for n, ww, vv, uu in top))

    d = np.load(B.OUT / "experts_test.npz", allow_pickle=True)
    missing = [n for n in used if n not in d.files]
    if missing:
        raise SystemExit(f"test probabilities missing for {missing}")
    logs = np.stack([np.log(np.maximum(d[n], EPS)) for n in used])
    allow = d["allow"]
    d_te = _depth(data["counts_test"].to_numpy(), data["counts_train"].to_numpy())
    glia_te = data["meta_test"]["Region"].isna().to_numpy()

    z = np.zeros((len(allow), len(classes)))
    z[glia_te] = G.apply(logs[:, glia_te], *fits["glia"], l_c, d_te[glia_te],
                         log_prior, allow[glia_te])
    z[~glia_te] = G.apply(logs[:, ~glia_te], *fits["neuron"], l_c, d_te[~glia_te],
                          log_prior, allow[~glia_te])
    pred = classes[z.argmax(1)]

    meta_test = data["meta_test"]
    example = pd.read_csv("prediction/prediction.csv", nrows=0)
    sub = pd.DataFrame({"Cell_ID": meta_test.index.astype(str),
                        example.columns[1]: pred})
    assert len(sub) == 5000 and not sub.Cell_ID.duplicated().any()
    assert np.array_equal(sub.Cell_ID.to_numpy(), meta_test.index.astype(str).to_numpy())
    assert set(pred) <= set(classes)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"prediction_iteration20_{tag}.csv"
    text = sub.to_csv(index=False).rstrip("\n")
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    incumbent = pd.read_csv("prediction/prediction.csv", dtype={"Cell_ID": str}
                            ).set_index("Cell_ID").iloc[:, 0].reindex(
        meta_test.index.astype(str)).to_numpy()
    (B.OUT.parent / "iteration20" / "freeze_manifest.json").write_text(json.dumps({
        "candidate": tag, "file": str(path), "sha256": digest,
        "experts": list(used), "gate": "class log-prior + cell log-depth, per branch",
        "l2_int": L2_INT, "fit_partitions": list(PARTITIONS),
        "test_truth_read": False, "production_modified": False,
        "changed_vs_production": int((pred != incumbent).sum()),
    }, indent=2))
    print(f"\nwrote {path}\n  sha256 {digest}")
    print(f"  distinct labels {sub.iloc[:,1].nunique()}/{len(classes)}   "
          f"changed vs production {int((pred != incumbent).sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gated")
