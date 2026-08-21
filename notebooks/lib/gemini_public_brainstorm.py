"""Request a generic method brainstorm from Gemini without sharing project results.

The prompt contains no repository text, experiment metrics, cell records, test labels,
or secret values.  It describes only a generic targeted-MERFISH classification setting.
"""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

MODEL = os.environ.get("GEMINI_RESEARCH_MODEL", "gemini-3.1-pro-preview")
OUT = Path("outputs/iteration10/gemini_public_brainstorm.md")

SYSTEM = """You are a computational-biology ML researcher. Propose falsifiable,
competition-honest algorithms. Never use held-out labels, unreleased query genes, or
another team's predictions. Focus on top-1 multiclass accuracy and practical Apple
Silicon implementation."""

PROMPT = """Consider a generic targeted MERFISH annotation problem: 5,000 labelled
training cells, 5,000 unlabelled query cells, 200 sparse measured genes, 60 imbalanced
cell types, spatial coordinates, tissue-section/sample metadata, and a much larger
public labelled atlas from the same assay. The public atlas may be used only after
removing every query/challenge cell, and query cells have only the 200 released genes.

Suggest five technically distinct, modern algorithms likely to beat a strong ExtraTrees
baseline. Avoid generic hyperparameter tuning. For each provide: exact features and
loss, how external atlas data enter, how to prevent leakage during CV, a negative/null
control, expected compute on Apple MPS/CPU, and a minimal go/no-go experiment. Include
at least one tabular foundation model, one domain-adaptation method, and one generative
count model. Rank the five, then fully specify the top candidate with fixed defaults and
an independent-confirmation rule. Do not assume access to any project-specific result.
"""


def main() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")
    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": PROMPT}],
        max_completion_tokens=9000,
    )
    answer = response.choices[0].message.content or ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"# Sanitized Gemini methodology brainstorm\n\nModel: `{MODEL}`\n\n"
        f"## Prompt\n\n{PROMPT.strip()}\n\n## Response\n\n{answer.strip()}\n"
    )
    print(f"wrote {OUT} ({len(answer):,} response characters)")


if __name__ == "__main__":
    main()
