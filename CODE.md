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

Cell-disjoint out-of-fold accuracy on the released training cells: **0.8214**.
Accuracy on the original test set: **0.8126** (Cohen's kappa 0.8008).

Because prizes are decided on a validation cohort, we also measured generalisation to
unseen groups rather than unseen cells. Refitting and scoring by held-out group gives
82.20% for random cells, 82.22% for a whole held-out section, **82.33% for a whole
held-out mouse** and 82.23% for a whole held-out imaging run — the model is not
cohort-specific. The `Mouse_ID` and `Section_ID` one-hot columns, which are all-zero for an
unseen cohort, can be zeroed with no loss (−0.34 to +0.18 pt).

## Runtime

About three hours end to end on an Apple M3 (the neural experts use the GPU via MPS); the
two public `.h5ad` reference files must be present under `data/external/`.
