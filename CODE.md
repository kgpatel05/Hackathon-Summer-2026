# Code submission

One command reproduces the submission from the released data:

```bash
python3 run_prediction.py
```

It re-runs unchanged after `data/meta_test.csv` and `data/counts_test.csv` are replaced by
the validation cohort: it checks the cohort, fingerprints the input data, discards every
derived cache if the data changed, rebuilds every expert, refits the pool on the released
**training** cells, and writes `prediction/prediction.csv`. `--dry-run` lists the stages.

Everything needed is in this repository. There are no external downloads.

## The source dataset is not used

Per the 22 August clarification, training on the source data — the published dataset the
challenge was carved out of, `MERFISH_spinal_cord_0531.h5ad` — is not in the spirit of the
event. That file supplied the previous version of this model with its largest components.
All of it has been removed: the atlas transfers, the neighbourhood-composition and
atlas-niche feature blocks, the Laminae/Segment correspondence, the fine-tuned reference
networks, and the full 500-gene panel.

**The companion SNI dataset has been removed as well**, and that needs explaining, because
it is not obviously source data — `SNI_merged_0531.h5ad` is a different experiment on
different animals, and an earlier version of this model leaned on it hard. It is a
companion dataset of the same publication, whose methods state that its cell types were
assigned by "an ensemble label transfer strategy that integrates four state-of-the-art
methods: SingleR, Tangram, Seurat and RCTD", each predicting labels "using annotations from
the manually annotated MERFISH reference dataset". Its labels are therefore the source
atlas's annotations carried onto other cells. Training on them is training on the source
labels at one remove, so dropping the atlas while keeping SNI would have been a distinction
without a difference.

Removal is enforced, not asserted:

* **no shipped module can read either file.** The atlas functions and the SNI reference
  loader were deleted outright; nothing in this repository opens an `.h5ad`.
* **`no_source_data.py` blocks the source file at runtime.** It replaces `h5py.File` so any
  attempt to open `MERFISH_spinal_cord_0531.h5ad` raises immediately.
* **verified by construction.** The whole pipeline was re-run from a clone that does not
  contain either file.

## The model

A calibration-aware log-linear pool of 13 experts,

```
p(class | cell)  ∝  Π_m  p_m(class | cell)^{w_m}  ·  prior(class)^{−a}
```

with `w` and `a` fitted by maximising out-of-fold multinomial likelihood on the 5,000
released training cells, separately for the glia and neuron branches, pooled over four fold
partitions. A hard metadata-compatibility mask removes (cell, class) pairs whose
`Region`/`Excitatory_vs_Inhibitory`/`Segment` combination never occurs in training.

Feature stack (409 columns), all of it released data:

| block | cols | source |
|---|---:|---|
| `BASE` | 371 | the 200 released genes, 9 QC columns, metadata one-hot |
| `SPA` | 8 | registered spatial coordinates |
| `NIC` | 30 | niche expression over the challenge cells' own released counts |

Experts: ExtraTrees in five geometries (full stack, genes only, context only, wide, and two
multi-fold variants), RandomForest, XGBoost, logistic regression and two neural models on
that stack, a count-native multinomial fitted on the released training counts, and a
hierarchical metadata prior.

### What was deliberately left out

Spatial propagation of the *training* labels — a class histogram over each cell's nearest
labelled neighbours — is legitimate and would score well here, because the released test
cells share all 108 sections and all 10 mice with the training cells, giving every test
cell about 70 labelled neighbours. It is not used. `meta_train.csv` is not replaced by the
validation cohort, so if the validation cells come from new sections that feature collapses
to zeros and any expert leaning on it fails. It buys accuracy on this cohort at the cost of
the one that decides the prize.

## Data used

* `data/counts_train.csv`, `data/meta_train.csv` — released genes, metadata, labels
* `data/counts_test.csv`, `data/meta_test.csv` — released genes and metadata

## Not used

* the source dataset, in any form
* any dataset whose labels were transferred from it
* any cell-type label of any test or validation cell

## Performance

Cross-validated on the released **training** cells. The test line is reported for
information and was not used to select anything.

| protocol | accuracy |
|---|---:|
| cell-disjoint CV (hold out random cells) | **0.7736** |
| hold out a whole **mouse** | **0.7740** |
| released test set | 0.7838 |

Holding out an entire animal is no worse than holding out random cells, so the model is not
cohort-specific and should carry to a validation cohort from new tissue. That property is
the reason for the omission described above.

For reference, the version that used SNI scored 0.7854 cell-disjoint CV and 0.7872 on the
test set; excluding it cost 1.2 points of cross-validated accuracy.

## Runtime

About one hour end to end on an Apple M3 (the neural experts use the GPU via MPS).
