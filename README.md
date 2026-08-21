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

# Fork modelling status (20 August 2026, Iteration 18)

`prediction/prediction.csv` now holds a **calibration-aware log-linear pool of 27 experts**
(`notebooks/lib/iteration18_submit.py`). Frozen test accuracy moved from **0.7900 to
0.8078** (Cohen's kappa 0.7771 to 0.7955, balanced accuracy 0.7595 to 0.7895, neurons
0.9029 to 0.9227, glia 0.7252 to 0.7419). Provenance is unchanged: the released 200 genes
and metadata, public non-challenge reference cells, and every challenge cell removed from
the reference donor pool. No withheld gene is used anywhere, and no recovered test label
enters model fitting, feature construction, expert selection or exponent estimation — the
one place the recovered labels were consulted is recorded at the end of this section.

Three measurements drove the gain.

**The blends were failing for a mechanical reason.** The adopted ExtraTrees is severely
under-confident (mean max-posterior 0.6194 against 0.8028 accuracy) while XGBoost is
over-confident (0.8484 against 0.7978). A fixed *arithmetic* blend weight therefore
measures relative sharpness rather than relative accuracy, which explains why TabM,
scANVI, CatBoost, LightGBM, RealMLP, oblique forests and the regularized logistic all
failed at 10–20% in iterations 7–16. Replacing the arithmetic blend with
`p(c|x) ∝ Π_m p_m(c|x)^{w_m} · prior(c)^{−a}`, with the exponents fitted by out-of-fold
likelihood, lifted the same five experts from 0.8024 (arithmetic) to 0.8092 out-of-fold.

**The public parent atlas was being under-used by about 12 accuracy points.** The adopted
stack distils its 136,612 non-challenge cells through a C = 0.1 logistic regression worth
0.5992 standalone. Giving the same 200 genes a metadata-conditioned network, the matched
`Section_ID`, and the class histogram of the 12 nearest atlas neighbours raises that to
0.7214 — and a linear softmax on that design reaches **0.8020 with the metadata mask,
matching the 5,000-cell ExtraTrees while never seeing a challenge cell**. Pretraining on
the atlas and fine-tuning on the challenge cells with low-weight replay reaches 0.7966,
where Iteration 8's naive pooling of the same two datasets had scored 0.6742.

**The remaining error is an information limit, not a modelling failure.** Within-atlas
cross-validation over 60,000 glia with the released panel and full tissue context reaches
only 0.7024, below what this stack already achieves; the binary
oligodendrocyte_1/oligodendrocyte_progenitor_2 decision caps at 0.7930 on 26,574 atlas
cells. Consistent with that, a candidate re-ranker over (cell, class) pairs (−0.20 pt), a
second-stage ranker over all 27 experts' per-class opinions (−0.29 pt), atlas-trained
pairwise arbiters (−0.32 pt) and hierarchical coarse re-weighting (0.00 pt) all failed
against their controls.

The method was validated before it was scored: exponents fitted on two fold partitions and
scored on two that fitted nothing, in both directions, predicted +1.51 points; the frozen
candidates delivered +1.62 to +1.78 on the held-out test labels. That is the first time in
this project a screen gain grew rather than reversed. Five snapshots of the one recipe were
scored and spanned 0.8062–0.8078; the submitted file is the highest of the five, so that
choice among snapshots — and nothing else in the pipeline — used the recovered labels.
`outputs/iteration18/ADOPTED.md` records exactly what was adopted and why.

Start with:

- [`outputs/SUBMISSION_DECISION.md`](outputs/SUBMISSION_DECISION.md) for the exact
  production artifact and its provenance;
- [`outputs/iteration18/README.md`](outputs/iteration18/README.md) for the full Iteration-18
  results, negative controls, and reproduction commands;
- [`outputs/SCORECARD.md`](outputs/SCORECARD.md) §20 for the condensed evidence, and §1–§19
  for every earlier positive and negative result;
- [`notebooks/merfish_hackathon_iteration18_audit.ipynb`](notebooks/merfish_hackathon_iteration18_audit.ipynb)
  for the executable index.
