# AI as Area Chair: What Science Would AI Send to ICML?

## Motivation

Our scientific literature is increasingly co-planned, co-executed, and
co-written with AI [1]. Our reviews, rebuttals, and likely even final decisions
are increasingly shaped by AI assistance [2]. This influence, for better or
worse, is rising quickly and remains poorly understood.

As a natural experiment, we take this to the extreme: what if ICML peer review
and acceptance were decided *only* by AI? In other words, what if the
progress of science were shaped by AI?

To study this question, we crawl ICML 2026's public papers and ask AI systems to
judge which papers are most technically strong, most impactful for machine
learning, and most important for scientific progress. We compare AI's rankings
to those of human reviewers (the spotlights, orals, and paper awards).

## Research Questions

1. Which papers would AI rank as the best contributions to science and the
   future of machine learning?
2. Which rejected papers would AI identify as overlooked contributions?
3. How different are these rankings from the conference's human-led outcomes?
4. How sensitive are these rankings to model choice, prompt design, and
   contamination controls?

## Current Data Status

As of **2026-07-15**:

- ICML's public 2026 virtual-site index contains **6,628 accepted poster-paper
  rows**, including **574 spotlights** and **168 papers with oral
  presentations**. Oral event rows are merged into their canonical poster
  records rather than counted twice.
- **6,614/6,628** accepted papers have authenticated OpenReview forum PDFs.
  All 6,614 official files were downloaded and validated; parsing produced
  6,611 `ok`, three `needs_review`, and zero failures. The remaining 14 records
  have no OpenReview forum ID rather than failed PDF requests.
- The current metadata snapshot includes published reviewer ratings for
  **6,341** accepted papers: overall assessment, soundness, and confidence.
- Official awards are linked for two Outstanding Papers, five Honorable
  Mentions, one Outstanding Position Paper, and one position-paper Honorable
  Mention.
- Public OpenReview rejected-note queries currently return **214 rejected
  notes**. Rejected-paper analysis therefore applies only to the public opt-in
  rejected subset, not all rejected submissions.
- arXiv is useful as fallback and provenance enrichment, but not as the primary
  PDF source. In a 500-paper pilot, high-confidence arXiv coverage was 329/500.
  Among the top 50 accepted papers by public reviewer score, arXiv coverage was
  35/50 after deep-search retry.
- ICML reported **497 desk rejects, about 2% of submissions**, implying roughly
  **25,000 total submissions** overall.

## Methodology

### 1. Corpus and PDF Processing

We use the ICML virtual-site JSON feed as the accepted-paper inventory, resolve
official PDFs through OpenReview forum IDs, and download accepted plus public
opt-in rejected PDFs. arXiv is fallback only, with separate provenance.

PDFs are parsed with Poppler. For ICML 2026 camera-ready papers, we use the
first **9 PDF pages** as the main-paper prior, then strip references and detected
appendix/supplement material; original submissions use 8 pages. The pipeline
writes compact, scoring, and full paper representations. Substantive scoring
uses the main-paper `scoring_repr`.

### 2. Cheap High-Recall Triage

Every parsed paper is scored by a low-cost OpenRouter ensemble:

- `nvidia/nemotron-3-ultra-550b-a55b`
- `google/gemini-3.1-flash-lite`
- `openai/gpt-5.6-luna`
- `x-ai/grok-4.3`

This stage uses listwise forced-ranking batches rather than independent 1-10
scores, which were too compressed in early tests. Models judge technical
soundness, novelty, clarity, empirical credibility, ML-field impact, broader
scientific importance, contribution type, and whether the paper should advance.

Papers are grouped by contribution route: algorithm, theory, benchmark/data,
infrastructure, scientific tool, safety/evaluation, application, or analysis.
We aggregate by source model and select a top **15-20%** class-balanced
shortlist, preserving per-class leaders, strong rejected candidates, and
cheap/strong disagreement cases. This pass is a recall filter, not the final
ranking.

### 3. Strong Semifinal and Conservative Finalists

The shortlist is independently reranked over `scoring_repr` by:

- `gpt-5.6-terra` with `high` reasoning
- `anthropic/claude-sonnet-5` with `high` reasoning

We average normalized ranks with equal weight and retain each judge's rank,
rationale, requested/served model ID, and disagreement signal. Because our ICML
2025 gold-set evaluation found that aggressive intermediate cuts lost true top
papers, this stage improves ordering but does not make a narrow cut.

Finalists combine the consensus rank, cheap-ensemble and per-class leaders, and
both cheap/strong and Terra/Sonnet disagreement saves. The initial full-run
target is **150-300 PDF-aware finalists**. Claude Opus 4.8 is reserved for
optional disagreement adjudication rather than the full semifinal pool.

### 4. PDF-Aware Frontier Judgment

For finalists only, three model families review a first-9-page PDF excerpt plus
extracted main-paper text and produce independent structured judgment cards:

- `openai/gpt-5.6-sol` with `xhigh` reasoning
- `anthropic/claude-fable-5` with `high` reasoning
- `google/gemini-3.1-pro-preview` with `high` reasoning

Requested and served model IDs are stored for every card. This matters because
a safeguarded Fable request can be routed to Opus. The historical ICML 2025
gold-set protocol used:

- `openai/gpt-5.5` with `xhigh` reasoning
- `openai/gpt-5.4` with `high` reasoning
- `anthropic/claude-opus-4.7` with `high` reasoning

The historical panel is retained for reproducibility. Production pays the PDF
reading cost once per finalist, then reuses the cards for ranking and audit.
All three refreshed judges passed a live PDF/JSON smoke card with exact served
model IDs; the architecture, rather than the new panel's ranking, is what the
historical gold set validates.

### 5. Final Tournament Ranking

The final ranking uses frontier cards, not raw PDFs, with GPT-5.6 Sol xhigh as
the synthesis and pairwise adjudication model. Small finalist sets use
all-pairs comparison; larger pools use Swiss pairwise comparisons over a broad
pool, followed by dense all-pairs comparison within the provisional playoff
subset. Headline claims come from the dense playoff, while Swiss standings
provide broader finalist ordering. Regularized Bradley-Terry strengths adjust
the sparse Swiss standings for opponent difficulty; direct win rates remain the
transparent primary check for the balanced all-pairs playoff.

Recommended defaults: cheap shortlist top 20%, Terra/Sonnet semifinal over the
full shortlist, 250 PDF-aware finalists, 150-paper Swiss pool, 10 Swiss rounds,
and a 60-paper dense playoff.

## Gold-Set Validation

We validated the ranking design on a 50-paper ICML 2025 accepted-paper testbed.
The current reference is not a single text-only model ranking. It uses:

1. three independent PDF-aware frontier cards per paper;
2. GPT-5.5 synthesis over those cards;
3. GPT-5.5 xhigh all-pairs tournament over the synthesis top 30.

The all-pairs tournament completed **435/435** comparisons in 18 resumable
batches, cost about **$8.10** in reported OpenRouter usage, and took about
57.5 minutes.

Key lessons:

- the old text-only gold ranking had Spearman 0.2875 against the tournament
  gold;
- the PDF card ensemble had Spearman 0.9608;
- the PDF card synthesis had Spearman 0.9769;
- the cheap ensemble top-45 captured 10/10 gold top-10 and 19/20 gold top-20;
- an aggressive GPT stage-2 top-35 cut missed 2 gold top-10 and 3 gold top-20
  papers.

The main empirical lesson is to preserve candidates generously before
PDF-aware carding and tournament adjudication.

The July 2026 Terra/Sonnet semifinal ensemble is an auditable diversity and
reliability improvement, but it has not yet replaced the historical gold-set
metrics above. We will compare it with Terra alone at an equal finalist budget
before freezing the production analysis.

A July 2026 refresh screened 14 current OpenRouter models on the same 50-paper
set and produced 11 usable rankings. All 330 four-model subsets were evaluated.
The selected four-model panel retained
10/10 gold top-10 papers by shortlist rank 20 and 18/20 gold top-20 papers by
rank 30, versus 17/20 for the previous three-model preset. The full 11-model
aggregate matched 18/20 but recovered only 23/30 gold top-30 papers versus
24/30 for the selected four at roughly four times the cost. No equally strong
four-model subset was cheaper or faster.

## Cost and Runtime

- **Metadata and ratings:** cheap and fast. Accepted OpenReview ratings were
  fetched in 7 batched requests in about 18 seconds.
- **Official PDF download:** complete for all 6,614 available OpenReview PDFs.
  The resumable authenticated crawl used a two-second delay and zero in-run
  retries; two isolated timeouts recovered on later bounded invocations.
- **PDF parsing:** local and fast. A 200-PDF parse took about 54 seconds;
  full-corpus parsing should be on the order of tens of minutes once PDFs are
  present.
- **Cheap first pass:** the selected four-model panel cost **$0.78** for the
  accepted-50 probe, projecting to roughly **$104** for 6,628 papers under the
  same batching and context budget. Sequential runtime projects to about 18
  hours; model-level parallelism can reduce wall time subject to provider limits.
- **Strong semifinal:** Terra and Sonnet run independently, so either result can
  be retried without repeating the other. At equal token usage, the two-judge
  panel is roughly 1.7-1.8 times the cost of Terra alone at current rates and
  remains substantially cheaper than PDF-aware frontier review.
- **Frontier PDF cards:** the historical ICML 2025 gold-set cards cost about
  **$33.82** for 150 card rows. The refreshed production panel is more capable
  but Fable 5 is pricier. Its three one-paper smoke cards cost **$0.93** total,
  a rough **$233** projection for 250 finalists before paper-level variation.
- **Tournament:** the 30-paper all-pairs tournament cost about **$8.10** for 435
  comparisons.

Working full-run estimate: **$500-$2,000**, depending mainly on context budgets,
reruns, final tournament size, and the public rejected-paper count. Runtime
should fit in a same-day or overnight run after PDFs are downloaded.

## Limitations

- **Rejected-paper coverage is incomplete.** Only authors who opted in have
  public rejected papers, so the "best rejected papers" analysis applies to the
  public rejected subset, not all rejects.
- **Possible contamination.** Some papers may already be in model training data.
  This can be mitigated, but not eliminated, by disabling retrieval, using
  paper-grounded prompts, comparing model families, and testing pre-decision or
  open-weight snapshots when feasible.
- **AI is not reproducing the real review process.** This is intentional: the
  study measures a paper-only executive AI judgment, not an AI simulation of
  reviewer discussion.
- **Prompt dependence matters.** Rankings may shift depending on whether the
  rubric emphasizes technical correctness, likely citation impact, conceptual
  novelty, or broad scientific importance.
- **arXiv is incomplete.** It misses roughly 30% of the top-rated accepted
  papers in our pilot and may expose older or non-camera-ready versions.
- **PDF parsing is imperfect.** Text extraction can miss figure/table nuance.
  PDF-aware frontier cards partially address this, but still use constrained
  excerpts plus extracted text.
- **Pairwise tournaments are expensive.** They are more defensible than a single
  ranked list, but scale quadratically, so the production plan uses Swiss
  exploration plus a dense playoff.

## Why This Is Interesting

This project isolates a question that is already implicit in modern peer review:
if AI is increasingly mediating how science is read and judged, what science
would it elevate if given final executive authority?

The result would not replace peer review. It would provide a concrete,
measurable picture of what an AI-centered evaluation regime selects for, misses,
and potentially gets right.

## Internal Methodology Docs

- Detailed pipeline: [`docs/final_pipeline_methodology.md`](docs/final_pipeline_methodology.md)
- Scoring and model details: [`docs/scoring.md`](docs/scoring.md)
- Crawling and PDF acquisition: [`docs/crawling.md`](docs/crawling.md)
- Methodology decision log: [`docs/methodology_decisions.md`](docs/methodology_decisions.md)

## References

[1] Ziming Luo, Zonglin Yang, Zexin Xu, Wei Yang, and Xinya Du. 2025.
**LLM4SR: A Survey on Large Language Models for Scientific Research.**
Surveying LLM use across hypothesis discovery, experiment planning,
implementation, scientific writing, and peer reviewing.
https://arxiv.org/abs/2501.04306

[2] Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng,
Fei Sha, Rose Yu, Carl Vondrick, and James Zou. 2025. **Can LLM feedback enhance
review quality? A randomized study of 20K reviews at ICLR 2025.** The study
deployed optional LLM feedback to more than 20,000 ICLR 2025 reviews; 27% of
reviewers who received feedback updated their reviews.
https://arxiv.org/abs/2504.09737

## Repository

Repository: [github.com/shayne-longpre/icml-ai-ac](https://github.com/shayne-longpre/icml-ai-ac)
