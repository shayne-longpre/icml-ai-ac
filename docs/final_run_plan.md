# ICML 2026 Final Run Plan

## Fixed Inputs and Blindness Boundary

The production population is the frozen 6,617-paper scoring manifest: 6,614
official OpenReview PDFs and three validated arXiv fallbacks. Eleven of 6,628
indexed papers lack a usable PDF and remain explicitly excluded.

Model-facing inputs contain only opaque paper identifiers, titles, identity-
redacted extracted text, identity-redacted PDF excerpts where configured, and
judgments from earlier AI stages.
They do not contain reviews, reviewer ratings, area-chair comments, decisions,
presentation tiers, or awards. Retrieval, web search, and external tools are
disabled. Derived artifacts remove direct byline, affiliation, email,
acknowledgement, and PDF-metadata cues and fail closed on validation; canonical
PDFs remain unchanged. This reduces direct prestige cues but cannot prevent
identification from titles, self-citations, project names, or model memory.
Human outcomes remain separate until the AI ranking is frozen.
The model-facing manifest itself also removes authors, canonical URLs, source
labels, and all nonessential metadata. Human outcomes remain in the separate
canonical manifest for post hoc joins.

## Stage 0 Gate

Identity redaction is complete under `direct_identity_redaction_v25`.
The model-facing manifest contains 6,617/6,617 successful rows, 6,617 validated
PDF derivatives, and 19,851 validated text artifacts. The final independent
audit reports zero errors; all 6,049 metadata titles present in the source title
region were retained, while 568 metadata/PDF title differences are recorded as
provenance drift. Canonical PDFs are unchanged. No paid ranking request was
made during anonymization or its audit.

## Production Stages

1. **Route contributions (6,617 papers).** Gemini 3.1 Flash Lite assigns
   contribution classes in fingerprinted 16-paper batches.
2. **Cheap recall ensemble (6,617 papers).** Nemotron 3 Ultra, Gemini 3.5 Flash
   Lite, GPT-5.6 Luna, and Grok 4.3 each rank two deterministic
   class-stratified partitions. Preserve the full aggregate signal and advance
   the top 20%, approximately 1,323 papers.
3. **Strong semifinal (approximately 1,323 papers).** GPT-5.6 Terra high and
   Claude Sonnet 5 high independently rank the same shortlist in size-8,
   two-partition batches. Preserve both source rankings, normalized consensus,
   and disagreement.
4. **Select finalists (250 papers).** Take a deterministic union of consensus
   leaders, cheap-stage leaders, contribution-class leaders, and large
   cheap/strong or Terra/Sonnet disagreements.
5. **Create PDF-aware cards (750 calls).** GPT-5.6 Sol xhigh, Claude Fable 5
   high, and Gemini 3.1 Pro high each review the first nine PDF pages plus the
   extracted main paper. Fable failures fall back explicitly to Opus 4.8 high.
6. **Synthesize and adjudicate.** Sol xhigh ranks the 250 reusable card bundles,
   then runs ten Swiss rounds over the top 150 and a dense all-pairs playoff
   over the top 60. Report direct playoff ordering and regularized
   Bradley-Terry strengths with conditional standard errors.
7. **Compare with humans post hoc.** Only after freezing AI outputs, join
   awards, presentation tier, and published reviewer scores. Report agreement,
   advancement recall, divergence tails, contribution-category rankings, and
   qualitative divergence codes.

## Artifact and Resume Contract

Use a distinct, stable output and run directory for every stage and model.
Never use `--overwrite` when resuming an interrupted production run. Reinvoke
the exact same command after correcting the interruption.

Every multi-call stage stores:

- exact prompts and request fingerprints;
- raw provider responses and parsed structured judgments;
- validation failures and error records;
- requested and served model identifiers;
- token and cost usage when reported;
- per-batch or per-paper coverage and resume state; and
- source judgments, normalized ranks, selector reasons, disagreements, pair
  schedules, A/B randomization, pair decisions, and aggregation outputs.

Matching valid units are reused; only failed or missing units are called again.
HTTP 400, 401, 402, 403, 404, 422, and exhausted 429 responses stop remaining
requests in that model stream. In particular, insufficient OpenRouter credits
produce a persisted partial run rather than consuming requests across the rest
of the stage. After credits are replenished, the same command resumes at the
first failed unit.

The expensive PDF cards are independent reusable artifacts. Synthesis,
tournament schedule, playoff size, direct aggregation, Bradley-Terry fitting,
and human-comparison analyses can therefore be replaced or rerun without
re-reading PDFs or repeating earlier model stages.
