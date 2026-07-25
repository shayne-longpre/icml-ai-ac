# Final Pipeline Methodology

This document describes the current production pipeline for identifying
high-impact ICML papers. It is written as a concise methods section: the running
decision history remains in `docs/methodology_decisions.md`.

## Objective

The pipeline ranks papers by expected broad scientific and machine-learning
impact, not by conventional reviewer score alone. The target output is an
auditable overall ranking plus category-specific rankings for distinct
contribution routes: algorithms, theory, benchmarks/datasets, infrastructure,
scientific tools, safety/evaluation, applications, and analysis papers.

All model stages are paper-grounded. Retrieval, web search, citation memory,
author identity, institution, and external tools are excluded from scoring
prompts. The only allowed evidence is stored metadata, extracted paper text,
PDF-derived visual/table evidence where explicitly supplied, and previous
pipeline judgments.

## Paper Representation

Each paper is stored with metadata, PDF path, and parse status. PDF text is
extracted into multiple representations. The main scoring representation uses
the ICML page-limit prior, then removes references and detected
appendix/supplement material. For ICML 2026 camera-ready papers this is the
first 9 PDF pages; for originally submitted versions it should be set to 8
pages. This reduces accidental inclusion of non-main-paper content while preserving
experiments, tables, baselines, ablations, and method details that were missing
from earlier compact representations.

For PDF-aware stages, the pipeline creates matching first-page PDF excerpts. Oversized
excerpts can be compressed before upload so that image-heavy papers do not
cause provider failures. Text remains included alongside the PDF excerpt.

## Stage 1: Contribution Routing

A cheap model first classifies papers by contribution route using compact text:
title, abstract, introduction, and conclusion-style content. These labels are
used for stratified batching and class-balanced selection. They are not treated
as quality judgments.

## Stage 2: Cheap Ensemble Triage

All parsed papers are evaluated by the `production_2026_v2` OpenRouter
ensemble: `nvidia/nemotron-3-ultra-550b-a55b`,
`google/gemini-3.1-flash-lite`, `openai/gpt-5.6-luna`, and
`x-ai/grok-4.3`. The design uses listwise forced-ranking batches rather than
independent absolute 1-10 scores, because early Qwen runs produced compressed
score distributions. The cheap ensemble's goal is high recall into the
strong-model stages; its ordering is not trusted for headline ranking.

The shortlist builder aggregates multiple cheap score files, normalizes by
source model so one model cannot dominate by producing more rows, and selects a
class-balanced semifinal pool. It stores the ensemble signal, including
aggregate priority, source-model priorities, advance votes, and source rows, so
later ranking layers can reuse the cheap evidence. The key production knobs are:

```text
score-pass1-ensemble --model-preset
score-pass1-ensemble --aggregate-out
build-shortlist --limit
```

For the full run, the initial target is a generous cheap shortlist of 15%-20%
of parsed papers.

## Stage 3: Strong Semifinal Ranking

Two independent text-only judges rank the full cheap shortlist over the
main-paper scoring text: GPT-5.6 Terra high through OpenAI and Claude Sonnet 5
high through OpenRouter. Each judge must return exact paper coverage. A local,
deterministic ensemble then averages equal-weight normalized ranks. It stores
both source rows, requested and served model IDs, source ranks, consensus rank,
and normalized judge disagreement.

Equal weighting avoids an unsupported calibration claim. Normalized ranks avoid
mixing model-specific absolute score scales. Consensus ties prefer lower judge
disagreement; disagreement is retained separately so the selector can preserve
papers strongly favored by either judge. This stage improves comparative
ordering and rationales but is not allowed to make a narrow final cut.

The production commands are:

```text
rank-pass2 --provider openai --model gpt-5.6-terra --reasoning-effort high
rank-pass2 --provider openrouter --model anthropic/claude-sonnet-5 --reasoning-effort high
ensemble-semifinal-rankings --ranking ... --ranking ... --label terra --label sonnet
```

The recommended default is to give both judges the same complete shortlist.
Claude Opus 4.8 is an optional adjudicator for high-disagreement cases or a
small robustness sample, not a default full-shortlist judge.

## Stage 4: Conservative Finalist Selection

Finalists are selected deterministically from the cheap ensemble and semifinal
consensus. The selector keeps top consensus papers, cheap-stage leaders,
per-class leaders, cheap/semifinal disagreement saves, and configurable
Terra/Sonnet disagreement saves. This replaces the previous GPT stage-2 top-N
cut.

The tunable knobs are:

```text
select-finalists --limit
select-finalists --semifinal-top
select-finalists --cheap-top
select-finalists --min-per-class
select-finalists --disagreement-saves
select-finalists --judge-disagreement-saves
```

For the full run, the initial target is 150-300 PDF-aware finalists.
The union of preservation sets must fit within the finalist limit; configuration
fails closed rather than silently discarding a requested class or disagreement
save.

## Stage 5: PDF-Aware Frontier Cards

Only the conservative finalist pool receives expensive PDF-aware review. Each
frontier judge sees the configured first-page PDF excerpt plus extracted main-paper text
and writes a structured judgment card with contribution class, impact route,
scores, best-case impact, main risk, and visual/table evidence notes.

The production panel uses GPT-5.6 Sol xhigh, Claude Fable 5 high, and Gemini
3.1 Pro high. This provides independent OpenAI, Anthropic, and Google judgments.
Each card stores both the requested and served model ID so provider routing is
auditable. The important design principle is to pay the PDF cost once per
finalist and reuse the resulting cards for ranking.

One-paper live smoke cards completed 3/3 with valid structured outputs and no
served-model substitutions. Reported costs were $0.258 for Sol, $0.608 for
Fable, and $0.065 for Gemini, or $0.93 total. This is an interface/cost check,
not a claim that the refreshed panel has independently replaced the historical
accepted-50 gold. A 250-finalist run projects to roughly $233 at this observed
paper size.

The finalist file flows directly into card generation:

```text
rank-frontier-pdf-cards --paper-list finalists.jsonl
```

## Stage 6: Hybrid Tournament Ranking

The final ranking uses card evidence rather than raw PDFs, with GPT-5.6 Sol
xhigh as the synthesis and pairwise adjudication model. For small finalist sets,
all-pairs comparison is preferred. For larger finalist sets, the current
cost-control design is a hybrid tournament:

1. Run Swiss pairwise comparisons over a broad card-ranked pool.
2. Select a provisional top subset from Swiss standings.
3. Run all-pairs comparisons only within that subset.
4. Use the dense playoff as the headline ranking and Swiss standings as
   supporting evidence outside the playoff.

Fit regularized Bradley-Terry models separately to the Swiss schedule and the
dense playoff. The Swiss fit adjusts provisional standings for opponent
difficulty under an uneven schedule. The playoff remains directly auditable by
win rate because every finalist faces every other finalist; its Bradley-Terry
ordering and conditional standard errors are reported as a sensitivity
analysis. These intervals are conditional on the observed judgments and do not
represent uncertainty over prompts, judge models, or model families.

The tunable knobs are:

```text
rank-frontier-card-tournament --strategy
rank-frontier-card-tournament --top-n
rank-frontier-card-tournament --swiss-rounds
rank-frontier-card-tournament --playoff-top-n
rank-frontier-card-tournament --pairs-per-batch
rank-bradley-terry --stage auto
```

Recommended first production defaults:

```text
PDF-aware finalists: 250
Swiss pool: 150
Swiss rounds: 10
Dense playoff: 60
```

## Gold-Set Evaluation

The current diagnostic reference is a 50-paper ICML 2025 accepted-paper set. The
old reference was a single GPT-5.4 text-only ranking; it is now a historical
baseline only. The upgraded reference uses:

1. Three independent PDF-aware frontier cards per paper.
2. A GPT-5.5 synthesis over the three cards.
3. A GPT-5.5 xhigh all-pairs tournament over the synthesis top 30.

The tournament completed 435/435 comparisons in 18 resumable batches, cost about
$8.10 in OpenRouter usage, and took about 57.5 minutes. Ranks 1-30 are
pairwise-adjudicated; ranks 31-50 remain from the synthesis seed.

Against this tournament gold:

| Candidate pipeline artifact | Main result |
| --- | --- |
| Old text-only gold | Spearman 0.2875 vs tournament gold |
| PDF card ensemble | Spearman 0.9608 |
| PDF card synthesis | Spearman 0.9769 |
| Cheap ensemble top45 | Captured 10/10 gold top-10 and 19/20 gold top-20 |
| GPT stage-2 top35 cut | Missed 2 gold top-10 and 3 gold top-20 papers |
| Conservative finalist selector top45 | Captured 10/10 gold top-10 and 19/20 gold top-20 |

The principal empirical lesson is that the cheap ensemble is already effective
as a generous recall filter, while the aggressive strong-model narrowing step
was the main recall failure. The final pipeline therefore favors conservative
candidate preservation before PDF-aware carding and tournament adjudication.
The Terra/Sonnet ensemble was added after this historical evaluation and must be
compared with Terra alone at the same finalist budget before production results
are frozen.

## July 2026 Model Refresh

Before the ICML 2026 run, 14 current OpenRouter models were screened on the
accepted-50 set. Eleven produced usable complete or near-complete rankings.
Step 3.7 Flash and Gemini 3.5 Flash rejected the shared reasoning-off
configuration, DeepSeek V4 Pro returned one malformed final batch, and
DeepSeek V4 Flash high was stopped after two very slow batches. Across all 330
four-model subsets of the 11 usable rankings, the selected production panel
gave the best practical recall/reliability/runtime tradeoff:

| Cheap panel | Gold top-10 in top 20 | Gold top-20 in top 30 | Gold top-30 in top 30 | Accepted-50 cost |
| --- | ---: | ---: | ---: | ---: |
| Previous three-model preset | 10/10 | 17/20 | 23/30 | $0.31 |
| Selected four-model preset | 10/10 | 18/20 | 24/30 | $0.78 |
| All 11 usable models | 10/10 | 18/20 | 23/30 | $3.05 |

The production panel is therefore limited to four. No other four-model subset
that matched its recall and category coverage was cheaper or faster. Grok 4.5
low, Claude Sonnet 5 low, and Qwen3.7 Max were specifically tested and did not
improve first-pass recall enough to justify replacing a selected model. At the
observed context budget, the panel's accepted-50 cost projects to roughly $104
for 6,628 papers.

## Tournament Cost Evaluation

The all-pairs top-30 tournament is the highest-confidence reference. To estimate
whether a cheaper production tournament is viable, the paid all-pairs outcomes
were reused to simulate sparse schedules. A 10-round Swiss pass plus all-pairs
among the provisional top 20 used 256 pair comparisons instead of 435, recovered
9/10 of the exact all-pairs top 10, placed all 10 all-pairs top-10 papers within
its top 15, and reached Spearman 0.9764 against all-pairs. Pure neighborhood
round-robin schedules were much weaker.

This supports Swiss as a broad reranking step before the final dense playoff,
not as the final ranking by itself.

## Auditability

Every stage writes resumable artifacts: prompts, raw provider responses, parsed
JSON, validation errors, run metadata, costs when reported by the provider, and
output rankings. Later invocations reuse successful paper-level or batch-level
artifacts and retry only missing or failed units. Evaluation uses top-k recall,
gold-top-N-in-candidate-top-M recall, rank correlation over shared papers, and
gold-anchored contribution-class recall.
