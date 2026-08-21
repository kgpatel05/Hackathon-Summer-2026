"""Ask Gemini for an independent, auditable Iteration 9 research critique.

The API key is read only from ``GEMINI_API_KEY``.  The prompt contains aggregate CV
findings and method descriptions, not cell-level records, recovered test labels, or
withheld expression.  The response is saved verbatim so every adopted idea has a visible
provenance trail.

Run from the repository root after sourcing ``.env.local``:

    set -a; source .env.local; set +a
    "$(cat graphify-out/.graphify_python)" notebooks/lib/gemini_iteration9_research.py
"""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


MODEL = os.environ.get("GEMINI_RESEARCH_MODEL", "gemini-3.1-pro-preview")
OUT = Path("outputs/iteration9/gemini_research.md")

SYSTEM = """You are a skeptical senior computational-biology and machine-learning
researcher. Your job is to propose falsifiable methods, detect leakage, and reject ideas
that merely rename experiments already run. Be quantitative, implementation-aware, and
explicit about why a method could improve top-1 accuracy rather than only representation
quality. Never recommend using held-out test labels or unreleased genes."""

PROMPT = """We need improve an honest 60-class MERFISH cell-type classifier using exactly
5,000 labelled cells, 5,000 unlabelled cells, 200 sparse count genes, spatial coordinates,
and metadata. Train/test are IID halves containing all 10 mice and 108 sections. The
current frozen model is ExtraTrees over log1p counts, QC, one-hot metadata, registered
spatial features, a challenge-neighbour expression PCA, and 60-column probability
transfers from two public references restricted to the same 200 genes. It averages 20
seeds, divides probabilities by class_prior**0.45, and hard-masks label/metadata
combinations never observed in labelled training data. Honest CV is about 0.79 overall,
0.73 glia and 0.897 neurons.

The dominant errors are within glia: oligodendrocyte_1 / oligodendrocyte_2 /
oligodendrocyte_progenitor_2 and astrocyte_1 / astrocyte_2. The released panel omits
canonical discriminating markers such as Plp1, Mbp, Mog, Sox10, Pdgfra, Cspg4, Aqp4,
Gfap, Opalin, C4b, C3, Slc7a10 and Flt1. A public 136k-cell parent atlas restricted to
these 200 genes saturates around 0.715 on glia, suggesting information limitation.

Already tested and rejected or exhausted: logistic/ExtraTrees/RandomForest/HistGB/XGBoost
and blends; hierarchical glia/neuron and pairwise specialists; class weighting and nested
prior calibration (Platt, vector, isotonic); external reference transfer, atlas logistic,
mouse harmonisation and atlas label neighbours; spatial kNN, niche PCA, full-atlas niche,
GraphSAGE with spatial/expression/random controls; PCA/autoencoder/masked denoising/atlas
pretraining/fine-tuning; label smoothing/self-training-like graph propagation; count
thinning and TTA; gene transforms/ranks/normalisations; target encoding of metadata;
learning curves; and hyperparameter retuning. A new joint Hungarian decoder that shrank
predicted class totals toward train-derived quotas lost 0.36 point at 25% strength; a 95%
sampling-interval version lost 0.22 point. Do not suggest these again.

Constraints: candidate selection must use training-only CV, one OOF prediction per cell,
paired McNemar, an independent fold-partition confirmation, and then at most one sealed
test evaluation. Runtime should be practical on an Apple M3; PyTorch MPS is available,
tree models use CPU. External atlas labels are allowed only for non-challenge cells and
only the released 200 genes may be read for challenge cells.

Task:
1. Audit whether >0.82 is statistically/biologically plausible under these constraints.
2. Identify 5 genuinely distinct algorithmic candidates not subsumed by the experiments
   above. For each give the exact mechanism, why it changes top-1 decisions, leakage
   risks, compute estimate, and a minimal decisive experiment.
3. Rank them by expected accuracy gain and probability of a reproducible positive gain.
4. Select one first experiment and specify enough detail (loss, folds, features,
   hyperparameters, null/control, adoption rule) to implement without further tuning.
5. Be willing to conclude that no honest 3-point gain is plausible, but still find the
   highest-value experiment rather than stopping at that conclusion.
"""


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        max_completion_tokens=12000,
    )
    answer = response.choices[0].message.content or ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"# Gemini Iteration 9 research critique\n\n"
        f"Model: `{MODEL}`\n\n"
        f"## Prompt\n\n{PROMPT.strip()}\n\n"
        f"## Response\n\n{answer.strip()}\n"
    )
    print(f"wrote {OUT} ({len(answer):,} response characters)")


if __name__ == "__main__":
    main()
