# Code submission

One command reproduces the submission from the released data:

```bash
python3 run_prediction.py
```

It re-runs unchanged after `data/meta_test.csv` and `data/counts_test.csv` are replaced by
the validation cohort: it fingerprints the input data, discards every derived cache if the
data changed, rebuilds every expert, refits the pool on the released **training** cells,
and writes `prediction/prediction.csv`. `--dry-run` lists the stages.

## The source dataset is not used

Per the 22 August clarification, training on the source data — the published dataset the
challenge was carved out of, `MERFISH_spinal_cord_0531.h5ad` — is not in the spirit of the
event. That file supplied the previous version of this model with its largest components.
All of it has been removed: the atlas transfers, the neighbourhood-composition and
atlas-niche feature blocks, the Laminae/Segment correspondence, the fine-tuned reference
networks, and the full 500-gene panel. On the released test set that cost 0.9518 → 0.7864.

Removal is enforced, not asserted:

* **no shipped module can read it.** The source-atlas functions were deleted from
  `iteration5_features.py`; no file in this repository opens that dataset.
* **`no_source_data.py` blocks it at runtime.** It replaces `h5py.File` so that any attempt
  to open `MERFISH_spinal_cord_0531.h5ad` raises immediately. Both pipeline entry points
  import it. Outside data passes through untouched.

## The model

A calibration-aware log-linear pool of 26 experts,

```
p(class | cell)  ∝  Π_m  p_m(class | cell)^{w_m}  ·  prior(class)^{−a}
```

with `w` and `a` fitted by maximising out-of-fold multinomial likelihood on the 5,000
released training cells, separately for the glia and neuron branches, pooled over four fold
partitions. A hard metadata-compatibility mask removes (cell, class) pairs whose
`Region`/`Excitatory_vs_Inhibitory`/`Segment` combination never occurs in training.

Feature stack (469 columns), all of it released data or outside data:

| block | cols | source |
|---|---:|---|
| `BASE` | 371 | the 200 released genes, 9 QC columns, metadata one-hot |
| `EXT` | 60 | posteriors transferred from SNI, restricted to the 200 shared genes |
| `SPA` | 8 | registered spatial coordinates |
| `NIC` | 30 | niche expression over the challenge cells' own released counts |

Experts: ExtraTrees (four geometries), RandomForest, XGBoost, logistic and two neural
models on that stack; the same on a stack augmented with the outside-data posteriors; a
count-native multinomial fitted on the released training counts; a hierarchical metadata
prior; and ten transfers from `SNI_merged_0531.h5ad`.

**SNI is used far harder than before.** It is a different experiment on different animals —
outside data — and 11× the size of the released training set. It carries five independent
annotations (`voting`, RCTD, Seurat, SingleR, Tangram); each is a differently-biased view,
so transfers trained on them make different mistakes, which is what the pool exploits.
Adding them was worth +0.32 point.

*Disclosure:* SNI's own labels were produced by its authors by transferring from the
published atlas. We do not touch that atlas; we use a separate published dataset and the
annotations released with it.

## Data used

* `data/counts_train.csv`, `data/meta_train.csv` — released genes, metadata, labels
* `data/counts_test.csv`, `data/meta_test.csv` — released genes and metadata
* `data/external/SNI_merged_0531.h5ad` — outside data, restricted to the 200 shared genes

## Not used

* the source dataset, in any form
* any cell-type label of any test or validation cell

## Performance

All figures are cross-validated on the released **training** cells except the last line.

| protocol | accuracy |
|---|---:|
| cell-disjoint CV (hold out random cells) | **0.7942** |
| hold out a whole **mouse** | **0.7938** |
| test set | 0.7872 |

Holding out an entire animal is no worse than holding out random cells, so the model is not
cohort-specific and should carry to a validation cohort from new tissue.

## Runtime

About one hour end to end on an Apple M3 (the neural experts use the GPU via MPS).
`data/external/SNI_merged_0531.h5ad` must be present.
