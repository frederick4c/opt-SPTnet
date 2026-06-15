# Writing Style Guide

This guide captures the current writing style of `report/main.tex`, with broader tendencies drawn from the medical imaging `report.tex`. It is intended to help new sections sound continuous with the existing report while preserving the same academic voice.

## Overall Voice

Use a clear, technical, and measured academic tone. The writing should explain specialised material in plain language while retaining domain-specific terms where they are necessary. Aim for confident but careful claims: describe what a method does, why it matters, and what its practical consequences are without overstating the result.

The style is primarily explanatory rather than rhetorical. It favours clarity, reproducibility, and scientific motivation over decorative phrasing. Paragraphs usually move from context to mechanism to implication.

## Sentence Style

Prefer medium-length sentences that link concepts explicitly. The writing often uses connective phrases such as "however", "therefore", "in contrast", "for this reason", "relative to", "together", and "this means that" to make the logic easy to follow.

Common sentence patterns include:

- "X provides a way to..." for introducing methods or concepts.
- "Unlike X, Y..." for contrast.
- "This allows..." for explaining consequences.
- "For this reason..." for causal links.
- "Together, these..." for synthesis.
- "In this project..." for narrowing broad background into the specific work.
- "The key point is that..." or equivalent phrasing for stating the takeaway.

Use active explanations where helpful, but passive construction is acceptable for methods and results: "Synthetic videos were generated", "Model performance was assessed", "Predictions were filtered".

## Paragraph Structure

Build paragraphs around one clear idea. A typical paragraph:

1. Introduces the concept, method, or result.
2. Explains the mechanism or evidence.
3. States why it matters for the project.

For background sections, start broad and then narrow toward SPTnet, reproducibility, or the project aims. For methods sections, describe the concrete implementation first, then the design reason or advantage. For results-style writing, begin with the observed trend, compare alternatives, and close with the practical interpretation.

## Technical Explanation

Explain technical details through their role in the workflow. Avoid isolated definitions unless they support the report's argument. When introducing a concept, connect it to inference, robustness, reproducibility, usability, or evaluation.

The style often pairs mathematical or algorithmic detail with a plain-language interpretation. For example, after defining a loss, metric, or physical quantity, explain what it measures and why it is useful.

Equations should be introduced before they appear and interpreted after they appear. Variables should be defined immediately after the equation in concise prose.

## Comparisons and Trade-Offs

A strong recurring tendency is to compare methods through balanced trade-offs. Phrase comparisons in terms of conditions, strengths, and practical implications.

Useful comparison frames:

- "X performs best under ..., whereas Y is more reliable when ..."
- "This improves ..., but comes at the cost of ..."
- "The difference becomes apparent when ..."
- "Neither approach is universally better; each is suited to ..."
- "This highlights a trade-off between ..."

Emphasise what each approach is useful for. When discussing limitations or failure cases, frame them constructively as motivation for design choices, safeguards, evaluation, or future work.

## Evidence and Claims

Ground claims in citations, figures, tables, metrics, or implementation details. The existing style is strongest when it uses specific evidence rather than broad assertion.

When citing literature, use citations naturally at the end of the relevant clause or sentence. The prose should still make sense without the citation command.

When referring to figures and tables, integrate them into the text:

- "The results are shown in \Cref{...}."
- "As shown in \Cref{...}, ..."
- "The metrics in \Cref{...} indicate ..."

Figure captions should be descriptive and self-contained. They should state what is being shown and, when useful, include the main visible trend or parameter setting.

## Methods Writing

Methods sections should be procedural, reproducible, and implementation-aware. Include enough concrete detail that the workflow can be understood without reading the code.

Prefer descriptions such as:

- data shapes and axis order;
- normalisation steps;
- matching, filtering, or thresholding rules;
- command-line or package-level interfaces;
- saved metadata and manifests;
- hardware or runtime context where relevant;
- safeguards that improve stability or reproducibility.

Explain design choices through practical consequences: making the system easier to run, easier to verify, less dependent on licensed tools, more robust to shape mismatches, or more suitable for batch experiments.

## Results and Discussion Writing

Results should combine qualitative and quantitative interpretation. Start with the main trend, then describe differences between conditions or methods, then state the implication.

Use language such as "suggests", "indicates", "shows", and "is consistent with" for measured scientific interpretation. Prefer careful conclusions tied to the evidence.

When discussing performance, distinguish between ideal conditions and degraded or realistic conditions. The writing often contrasts peak performance with robustness, or accuracy with practical usability.

## Terminology and Tone

Use precise technical terms consistently:

- "end to end" for whole-pipeline learning;
- "trajectory level" and "image level" for inference stages;
- "synthetic" and "experimental" for data sources;
- "robustness", "reproducibility", "generalisability", and "usability" for project-level themes;
- "signal to noise ratio", "artefact", "blur", "edge preservation", and "structure preservation" for image analysis.

Prefer British spelling where it appears in the reports: "normalised", "optimised", "localised", "artefact", "generalisation".

Use "which" clauses to add short explanatory detail, especially when explaining why a method behaves as observed. Keep the tone practical and scientific.

## Section-Level Tendencies

Introductions should motivate the scientific problem first, then the computational difficulty, then the reason the project approach is needed.

Background sections should compare families of methods and explain why the chosen method is relevant.

Methods sections should describe what was implemented, how it works, and how it supports reproducibility.

Evaluation sections should be explicit about metrics, thresholds, matching rules, and what each metric rewards.

Conclusion-style passages should summarise findings through their implications rather than simply repeating the procedure.

## Useful Phrases

- "This project is motivated by..."
- "The conventional approach is based on..."
- "The main difficulty is that..."
- "This reduces the risk of..."
- "The resulting workflow..."
- "A separate utility was written to..."
- "This made runs reproducible while still allowing..."
- "The results suggest that..."
- "This highlights the trade-off between..."
- "Overall, X provides the best balance between..."

## Style Priorities

Prioritise:

- clarity over ornament;
- causal explanation over listing;
- balanced comparison over simple ranking;
- implementation detail where it supports reproducibility;
- careful interpretation over absolute claims;
- smooth transitions between scientific motivation, algorithmic mechanism, and practical consequence.
