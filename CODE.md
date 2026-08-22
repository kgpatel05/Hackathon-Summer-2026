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
partitions, with a lightly regularised per-class bias on the pooled score. A hard
metadata-compatibility mask removes (cell, class) pairs whose
`Region`/`Excitatory_vs_Inhibitory`/`Segment` combination never occurs in training.

The exponents correct how *sharp* each expert is; they cannot correct a class the whole
panel reads systematically high or low, which is what the bias absorbs. It is worth
+0.0016 cell-disjoint and +0.0010 leave-one-mouse-out — small, but every penalty in
0.03–0.3 is at least as good as none on both protocols, so it is not a lucky point.

Feature stack (285 columns), all of it released data:

| block | cols | source |
|---|---:|---|
| `BASE` | 247 | the 200 released genes, 9 QC columns, metadata one-hot |
| `SPA` | 8 | registered spatial coordinates |
| `NIC` | 30 | niche expression over the challenge cells' own released counts |

Experts: ExtraTrees in five geometries (full stack, genes only, context only, wide, and two
multi-fold variants), RandomForest, XGBoost, logistic regression and two neural models on
that stack, a count-native multinomial fitted on the released training counts, and a
hierarchical metadata prior.

### Why no cohort identifiers

`Datasets`, `Mouse_ID` and `Section_ID` name this cohort rather than describing a cell.
One-hot encoded they were 124 of 409 columns, and on a validation cohort from new tissue
every one is zero, so the model would lose a third of its stack exactly when it is scored.
Removing them is worth +0.0034 cell-disjoint and +0.0028 leave-one-mouse-out, and costs
0.0046 on the released test set and 0.0021 on the simulated new-tissue cohort. The four
measurements split two against two and no gap exceeds 23 cells, so this is a wash rather
than a win: the identifier-free stack was kept for being 285 columns instead of 409 and
for not depending on how a cohort happens to be named, not because it measures better.
`Section_ID` is still read for spatial registration and niche grouping, where only the
grouping matters and the label is never encoded.

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
| cell-disjoint CV (hold out random cells) | **0.7786** |
| hold out a whole **mouse** | **0.7778** |
| released test set | 0.7798 |

Holding out an entire animal is no worse than holding out random cells, so the model is not
cohort-specific and should carry to a validation cohort from new tissue. That property is
the reason for the omission described above.

For reference, the version that used SNI scored 0.7854 cell-disjoint CV and 0.7872 on the
test set; excluding it cost 1.2 points of cross-validated accuracy.

## Verification

The submission was not merely produced; the shipped code was rehearsed under Sunday's
conditions. A fresh `git clone` of this repository — 14 files, no `data/external`, no
caches — had its test cohort replaced by a **3,137-cell** subsample, to check that nothing
assumes the released cohort's size, and was run with the documented command:

```
[preflight] 3137 cells, 200 genes, metadata complete
[1/7] ... [7/7] fit the pool and write the submission
VERIFIED prediction/prediction.csv: 3137 rows, 59 distinct labels, order matches meta_test.csv
```

It completed in 28 minutes and scored **0.7845** on that cohort, against 0.7844 on the full
5,000 — so the pipeline is insensitive to cohort size, not merely tolerant of it.

A second rehearsal replaced every `Section_ID`, `Mouse_ID` and `Datasets` value with one
never seen in training — the "new tissue" case — on a 4,211-cell cohort. The earlier
version, which one-hot encoded those columns, fell from 0.7844 to 0.7758 there.

That run is also worth reporting against expectation. The shipped model encodes no
identifier, so it should have been indifferent; it was not, scoring 0.7737 against 0.7773
for the same cells in the ordinary run. The reason is that `Section_ID` does more than
label a cell: it groups the spatial registration and the niche neighbourhoods. Renaming
the sections splits each one into a training half and a validation half, so a validation
cell's nearest neighbours are no longer the labelled cells beside it. That cost is
structural, applies to either model, and is the honest price of scoring genuinely new
tissue — not something the feature set can be arranged around.

Two further properties were checked rather than assumed:

* **the metadata mask degrades safely.** It is a hard constraint, so a validation cohort
  carrying metadata values absent from training could in principle mask every class off a
  cell. It cannot: unseen values are ignored, and a row that would be left with nothing is
  reset to all-allowed. Forcing every mask column to an unseen value leaves all 60 classes
  available and no cell blocked.
* **the caches cannot go stale.** The input data is fingerprinted; if it changes, every
  derived artifact is discarded before rebuilding, so a validation run cannot silently
  serve predictions built from the previous cohort.

## Runtime

About one hour end to end on an Apple M3 (the neural experts use the GPU via MPS).
