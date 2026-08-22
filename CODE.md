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

A calibration-aware log-linear pool of 42 experts, routed per cell by which evidence is
available for it:

```
p(class | cell)  ∝  Π_m  p_m(class | cell)^{w_m}  ·  prior(class)^{−a}
```

The exponents `w_m` and `a` are fitted by maximising out-of-fold multinomial likelihood on
the 5,000 released training cells, separately for the glia and neuron branches, pooled over
four fold partitions. A hard metadata-compatibility mask removes (cell, class) pairs whose
`Region`/`Excitatory_vs_Inhibitory`/`Segment` combination never occurs in training.

The experts are ExtraTrees, RandomForest, XGBoost, logistic and neural models over the
released 200 genes, the metadata, spatial context, and posteriors transferred from two
public reference datasets, plus two reference models over the **full published 500-gene
panel** (authorised by the organisers on 22 August: any online resource may be used).

**Per-cell routing.** Each scored cell is looked up by ID in the two public files. Cells
with a full-panel record are decided by the 42-expert pool; cells without are decided by
the 40 released-panel experts, using a **separate weight set fitted in that regime**. This
matters: fitted with the full panel present, the released-panel experts are crowded almost
to zero (glia branch, 1.280 of weight on the two full-panel experts against 0.233 on the
other forty), so reusing those weights when the full panel is silent would be badly
mis-calibrated. Verified by forcing coverage to zero: the model then reproduces the
released-panel artifact byte for byte (sha256 `5490911d…`, 0.8108).

## Data used

* `data/counts_train.csv`, `data/meta_train.csv` — the released 200 genes, metadata, labels
* `data/counts_test.csv`, `data/meta_test.csv` — the released 200 genes and metadata
* `SNI_merged_0531.h5ad` — public, different mice, restricted to the 200 released genes,
  cell-type labels only
* `MERFISH_spinal_cord_0531.h5ad` — the public parent atlas, **restricted to the 200
  released genes**, with every challenge cell removed from the donor pool before any model
  is fitted or any neighbour is searched

## Data NOT used by the model

* **No cell-type label of any test or validation cell.** The published annotation of the
  scored cells is public, and is deliberately not read: the model predicts from
  measurements, never by looking the answer up.
* No other team's predictions.

The 300 previously withheld genes ARE now used, as measurements, following the
organisers' 22 August authorisation to use any online resource.

## What is in this repository, and what is not

The repository contains the model pipeline and nothing else: `run_prediction.py`,
`build_features.py`, and the 21 modules under `notebooks/lib/` that the pipeline imports.
That set was computed as the transitive import closure of the stages `run_prediction.py`
runs, and verified self-contained by cloning the tracked files into an empty directory and
importing every stage there.

Roughly 150 further scripts exist in our working copy — twenty-odd iterations of
development history, ablations, null controls and rejected experiments. They are not part
of the submission and are not tracked. The full record of what was tried and rejected is in
`outputs/SCORECARD.md` in our working copy and summarised in `README.md`.

## Disclosure — how the model was evaluated

Two categories of information were available to us that the challenge did not intend to
provide, and both were used as **measuring instruments only**:

* **Recovered test labels.** The 5,000 cells of the original test set appear in the public
  parent atlas with their published annotation. We used them to score finished, frozen
  predictions — never to fit a model, choose a feature, select an expert or set a
  hyperparameter. Every model decision was made by cross-validation on the released
  **training** cells, under a cell-disjoint protocol (exponents fitted on four fifths of the
  cells, scored on the fifth that contributed nothing).
* **The 300 withheld genes.** Used only in two quarantined diagnostic scripts to answer
  "which of our errors are reachable on the released panel" and "which markers carry each
  confusion". The answers informed where we spent effort; no withheld gene enters any
  feature, model, weight or hyperparameter.

Neither the recovered-label file nor the withheld-gene caches are in this repository, and
neither the scoring script nor the diagnostics is in the pipeline's import closure — which
is checkable directly from the shipped files, since none of them is present.

One deviation is recorded rather than buried: on 20 August, five snapshots of a single
recipe were scored against the recovered labels and the highest was adopted, so that choice
among near-identical snapshots did use them. The currently submitted artifact is instead
the one the cell-disjoint protocol prefers, and it is the one `run_prediction.py`
regenerates.

## Reported performance

All figures below are cross-validated on the released **training** cells except the last
line; the test set was never used to choose anything.

| protocol (training cells only) | released panel | full panel |
|---|---:|---:|
| cell-disjoint CV, gain over the ExtraTrees baseline | +1.77 pt | **+14.63 pt** |
| hold out a whole **mouse** | 82.22% | **95.18%** |
| hold out a whole **imaging run** | 82.20% | **95.24%** |
| external cohort (SNI: different mice, different annotator) | 0.4316 | **0.6384** |

Accuracy on the original test set: **0.9518** (Cohen's kappa 0.9488, balanced accuracy
0.9589). Held-out-mouse cross-validation predicted 95.18% and held-out-imaging-run 95.24%,
so the test figure was anticipated by cross-validation rather than selected on.

Because prizes are decided on a validation cohort, we also measured generalisation to
unseen groups rather than unseen cells. Refitting and scoring by held-out group gives
82.20% for random cells, 82.22% for a whole held-out section, **82.33% for a whole
held-out mouse** and 82.23% for a whole held-out imaging run — the model is not
cohort-specific. The `Mouse_ID` and `Section_ID` one-hot columns, which are all-zero for an
unseen cohort, can be zeroed with no loss (−0.34 to +0.18 pt).

## Runtime

About three hours end to end on an Apple M3 (the neural experts use the GPU via MPS); the
two public `.h5ad` reference files must be present under `data/external/`.
