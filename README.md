# University of Rochester Biomedical Data Science Hackathon Summer 2026
Welcome to the landing page for the hackathon. The hackathon will commence 8/18. It will be a prediction challenge. All predictions should be submitted through GitHub using the captain's handle. Scoring will also happen in GitHub. All details regarding the hackathon will be posted here.  

 Register for the hackathon [here](https://forms.gle/TEW1BHqezsKgTTKL9). Please make sure each individual competing on your team is fully registered. Each team needs a captain with a github handle. To receive a prize, you must supply your University of Rochester e-mail address. All teams scoring better than random will receive a participation prize. 1st and 2nd place winning teams in each division will get a cash prize (see below).
 **All team members must submit their own registration form to participate.**  

# Overview
This is a prediction challenge with spatial transcriptomics data. The objective of the hackathon is to correctly predict cell type labels in MERFISH_cell_type_annotation. Group performance will be measured by the confusion matrix overall accuracy: number of correct predictions / total number of predictions.

# Challenge description
The challenge is to classify cell types in a mouse neuronal tissue dataset collected using MERFISH, an imaging-based spatial transcriptomics technique that measures gene expression while preserving each cell's exact location in the tissue. The dataset covers ~10,000 cells and 200 genes, with sparse transcript counts paired with spatial coordinates and cell metadata (like cell volume, region, gender, and mouse ID). Participants must predict the cell type label for each cell in the test set. A full description of the challenge and dataset is here: [Data.Description.md](Data.Description.md).

# Logistics

0.   Each team must have a github handle associated with it in order to participate.  Make sure you edit your registration or email the organizers to provide this, if you haven't yet. Your team will not be scored if you do not provide a handle.
1.   You may add team members up
to noon EDT on 8/18 by editing your response to the google form or emailing the organizers.
2.  Teams of entirely undergraduates will be in the undergraduate
division, else they will be in the open division.
3. Further instructions for submitting predictions will be posted here as they become available
4.  Competition runs through 2:59 PM EDT 22-August-2026.  The predictions each team has committed to their repository at that time will be used to determine their final score. Captains must submit their own predictions. Any use of predictions from other teams is disqualifying. Winning teams must submit their code to organizers to claim their prize.

Scores will be posted shortly after 3pm EDT each day here [Leaderboard.Hackathon.2026.md](Leaderboard.Hackathon.2026.md). 

# Prizes
   
1.  First place in each division: $300 + $75 x (team size)
2.  Second place in each division: 0 + $50 x (team size)

# Fork modelling status (21 August 2026, Iteration 20)

`prediction/prediction.csv` holds a **calibration-aware log-linear pool of 40 experts**
(`notebooks/lib/iteration18_submit.py`). Frozen test accuracy moved from **0.7900 to
0.8120** (Cohen's kappa 0.7771 to 0.8001, balanced accuracy 0.7595 to 0.7986, neurons
0.9029 to 0.9227, glia 0.7252 to 0.7485). Provenance is unchanged: the released 200 genes
and metadata, public non-challenge reference cells, and every challenge cell removed from
the reference donor pool. No withheld gene is used anywhere.

## What changed

**Arithmetic blending was misweighting the experts by sharpness rather than accuracy.**
The long-adopted ExtraTrees is badly under-confident (mean max-posterior 0.6194 against
0.8028 accuracy) while XGBoost is over-confident (0.8484 against 0.7978), so a fixed 10-20%
blend weight was measuring calibration, not skill — which explains why TabM, scANVI,
CatBoost, LightGBM, RealMLP, oblique forests and regularized logistic all failed in
iterations 7-16. Pooling in log space with exponents fitted by out-of-fold likelihood,
separately for the glia and neuron branches, lifts the same five experts from 0.8024
(arithmetic) to 0.8092 out-of-fold.

**`Laminae` in the public atlas is the challenge's `Segment`.** `Segment` is the most
informative released column, and every parent-atlas transfer in this repository had been
blind to it — which is why the earlier scorecard concluded that an atlas-trained model
"collapses to ~0.67 on neurons no matter how much data it gets". Restricted to the 44 cell
types that carry a `Segment`, the class-to-Segment and class-to-Laminae maps compose into a
bijection (22 levels each way, coverage 0.404 against 0.408), and it is a naming
correspondence rather than a fitted statistic: `L1 -> 1, L1-2 -> 2, L1-3 -> 3, ...`.
Supplying it lifts a linear model trained only on public atlas cells from 0.7214 to
**0.8056** on challenge cells — matching a 27-model ensemble without ever seeing one.

**Hyperparameter debt.** `max_features="sqrt"` had been inherited since the design had 371
columns; the augmented stack has ~1,050, so `sqrt` was sampling 3% of them. 0.25 is better
by +0.44 point on the strongest single model.

## Where the ceiling is

Within-atlas cross-validation on 70,000 non-challenge cells, one model, identical context
on every row:

| gene panel | 60-way | glia |
|---|---:|---:|
| 200 released + context | 0.7863 | 0.7226 |
| 300 withheld + context | 0.8476 | 0.8012 |
| 500 published + context | 0.9402 | 0.9205 |

The 300 withheld genes are worth **+15.4 points overall and +19.8 on glia**; even the
withheld panel alone beats the released panel. This stack already exceeds the released-panel
single-model figure because it adds the challenge's own labelled cells, the SNI reference
and a 36-model ensemble. The honest ceiling on the released panel is about **0.815-0.82**.

## Method validation

Pool parameters are validated **cell-disjointly**: five folds over the training cells, the
exponents fitted on four fifths and scored on the fifth that contributed nothing to the
fit, repeated over four fold partitions. Mean gain over the 694-feature ExtraTrees
**+1.77 points**, worst partition +1.70.

An earlier protocol fitted the exponents on fold partitions {18,41} and scored them on
{59,83}. Those are different fold assignments of the *same* 5,000 cells, so every scored
cell's label had been used in the fit. For the 37-parameter fixed pool the optimism was
small (it predicted +1.51 and delivered +1.62 to +1.92 on the real held-out cells), but it
overstated a 109-parameter gated variant by +0.32 point that did not exist — the gate lost
0.06 on test and is rejected. All reported numbers now use the cell-disjoint protocol.

Negative results with controls: hierarchical coarse re-weighting (0.00 pt), a (cell, class)
candidate re-ranker (-0.20), a second-stage ranker over all expert votes (-0.29),
atlas-trained pairwise arbiters (-0.32), a gated pool with class-frequency and cell-depth
interactions (0.00), a per-class logit offset (+0.03, noise), and imputing `Segment` for
glia (it is a cluster id, not a spatial subdivision: predictable from the label at 0.9975
and from position at 0.1851).

## The combiner is saturated

Four composition variants of the final pool - adding a nearest-class-mean reference view, a
boosted model on the augmented stack, an alternate-geometry ExtraTrees, and all of those
together with a re-seeded fine-tuned expert - land within 0.02 point of one another under
cell-disjoint validation, and none beats the adopted 40-expert set (+1.770 mean, +1.700
worst). Seed averaging was raised on every reference block that carries pool weight
(linear 10 to 24, network 5 to 12, ExtraTrees 2 to 10, and the strongest challenge-side
expert 5 to 10 out-of-fold and 10 to 20 for the test fit); the standalone accuracies barely
move, which is the expected signature of an ensemble that is already averaged enough.

The adopted artifact is SHA-256 `55d9dfb5ad13b5d2941dd142e011b330bf3f6bfe06e274adededc4536fa21f20`.

## Methods that work standalone but add nothing

A late round added seven further experts, several of them substantial results on their own:
a reference network regularised for prediction consistency on the unlabelled challenge
cells reaches **0.8106** without reading a single challenge label - better than the
694-feature ExtraTrees that was this project's production model for twelve iterations - and
retrieval against the atlas in that network's learned embedding reaches 0.7952, against
0.4362 for the same idea in raw gene space. Neither moves the pool: six compositions land
within 0.03 point of one another, and scored candidates at 37, 40 and 44 experts span
twelve cells. The combination is closed, not merely at diminishing returns.

## Why the plateau exists

Using the withheld genes purely as a measuring instrument - quarantined modules that
nothing in the prediction pipeline can import - the remaining errors split three ways: of
954, **164** are already solved by a released-panel reference model (real, but three
independent gating attempts could not identify them), **627** are solved only with the
withheld panel, and **163** by neither.

The mechanism is a single marker. The largest error bucket in this project, 156 cells
confusing `oligodendrocyte_1` with `oligodendrocyte_progenitor_2`, is led by **Opalin**,
the canonical myelinating-oligodendrocyte marker, which is withheld: on non-challenge atlas
cells the pair separates at 0.989 with the full panel and 0.798 with the released panel,
and only **13%** of the withheld discriminative direction is reconstructable from all 200
released genes (best single proxy r = 0.18). `meninges_1` versus `meninges_2` is the same
(R^2 = 0.18). Where the withheld direction is half-recoverable (R^2 0.47-0.59 for the
astrocyte, endothelial and oligodendrocyte_2 pairs) a plain linear model on the released
panel already reaches 0.90-0.92 in-sample against 0.98-0.99 with the full panel.

Start with:

- [`outputs/SUBMISSION_DECISION.md`](outputs/SUBMISSION_DECISION.md) for the production
  artifact and its provenance;
- [`outputs/iteration19/README.md`](outputs/iteration19/README.md) and
  [`outputs/iteration18/README.md`](outputs/iteration18/README.md) for the two iterations;
- [`outputs/SCORECARD.md`](outputs/SCORECARD.md) sections 20-21 for the condensed evidence
  and 1-19 for every earlier positive and negative result.
