# Code submission — what the model uses, and what it does not

Required by the 21 August rule update. One command reproduces the submission from the
released files:

```bash
python3 run_prediction.py
```

It re-runs unchanged after `data/meta_test.csv` and `data/counts_test.csv` are replaced by
the validation cohort: it fingerprints the input data, discards every derived cache if the
data changed, rebuilds all forty experts, refits the pool on the released **training**
cells, and writes `prediction/prediction.csv`. `python3 run_prediction.py --dry-run` lists
the stages.

## The model

A calibration-aware log-linear pool of 40 experts:

```
p(class | cell)  ∝  Π_m  p_m(class | cell)^{w_m}  ·  prior(class)^{−a}
```

The exponents `w_m` and `a` are fitted by maximising out-of-fold multinomial likelihood on
the 5,000 released training cells, separately for the glia and neuron branches, pooled over
four fold partitions. A hard metadata-compatibility mask removes (cell, class) pairs whose
`Region`/`Excitatory_vs_Inhibitory`/`Segment` combination never occurs in training.

The experts are ExtraTrees, RandomForest, XGBoost, logistic and neural models over the
released 200 genes, the metadata, spatial context, and posteriors transferred from two
public reference datasets.

## Data used

* `data/counts_train.csv`, `data/meta_train.csv` — the released 200 genes, metadata, labels
* `data/counts_test.csv`, `data/meta_test.csv` — the released 200 genes and metadata
* `SNI_merged_0531.h5ad` — public, different mice, restricted to the 200 released genes,
  cell-type labels only
* `MERFISH_spinal_cord_0531.h5ad` — the public parent atlas, **restricted to the 200
  released genes**, with every challenge cell removed from the donor pool before any model
  is fitted or any neighbour is searched

## Data NOT used by the model

* **None of the 300 genes the organisers withheld.**
* **No cell-type label of any test or validation cell.**
* No other team's predictions.

## Disclosure — the measurement scripts

The 5,000 test cells of the *original* test set are present in the public parent atlas with
their published annotation, and the withheld 300 genes are likewise public for those cells.
This repository contains scripts that read both, and they are published here rather than
withheld, because the honest record matters more than a tidy one:

| script | what it reads | what it is for |
|---|---|---|
| `notebooks/lib/evaluate.py` | recovered test labels | scoring a finished prediction after it is frozen |
| `notebooks/lib/iteration20_diagnose.py` | withheld genes of challenge cells | splitting our errors into reachable / withheld-limited / intrinsic |
| `notebooks/lib/iteration20_markers.py` | withheld genes of atlas cells | identifying which markers carry each confusion |
| `iteration14_leakage_probe`, `iteration16_score`, `iteration18_diagnose`, `iteration18_marginal_probe`, `iteration19_ceiling` | recovered labels or withheld genes | diagnostics and post-freeze measurement |

**None of them is in the model's import closure.** That is checked programmatically, not
asserted:

```python
# enumerate every local module reachable from the prediction pipeline
import ast; from pathlib import Path
lib, seen, stack = Path("notebooks/lib"), set(), ["iteration18_submit",
    "iteration18_experts_test", "iteration18_experts2", "iteration18_experts"]
while stack:
    m = stack.pop()
    if m in seen or not (lib / f"{m}.py").exists(): continue
    seen.add(m)
    for n in ast.walk(ast.parse((lib / f"{m}.py").read_text())):
        if isinstance(n, ast.Import): stack += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module: stack.append(n.module)
print(sorted(seen))   # contains no diagnostic module and no evaluate
```

Every model decision — features, hyperparameters, expert set, pool exponents — was made by
cross-validation on the released **training** cells, using a cell-disjoint protocol
(exponents fitted on four fifths of the cells and scored on the fifth that contributed
nothing). Recovered labels were read only to report a finished number.

One exception is recorded rather than buried: on 20 August, five snapshots of one recipe
were scored and the highest was adopted, so that choice among near-identical snapshots did
use the recovered labels. It is documented in `outputs/iteration18/ADOPTED.md`. The
currently submitted artifact is the one the cell-disjoint protocol prefers.

## Reported performance

Cell-disjoint out-of-fold accuracy on the released training cells: **0.8214**.
Accuracy on the original test set: **0.8126** (Cohen's kappa 0.8008).

## Runtime

About three hours end to end on an Apple M3 (the neural experts use the GPU via MPS); the
two public `.h5ad` reference files must be present under `data/external/`.
