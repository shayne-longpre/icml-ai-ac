# Static report

Build the data bundle from the frozen ranking and canonical post hoc metadata:

```bash
python3 site/build_data.py
```

For a separately downloaded analysis bundle, pass `--data-root`, `--summary`,
`--overrides`, and `--output`. The higher-level reproducibility command in
[`docs/reproducibility.md`](../docs/reproducibility.md) regenerates the human
comparison reports and website data together, then verifies them against the
frozen production references.

Preview from the repository root:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080/site/`.

The mock program uses final tournament ranks 1-2 for Outstanding Papers,
ranks 3-7 for Honorable Mentions, and the complete 60-paper all-pairs playoff
for AI Orals. Paper panels link to 59 verified arXiv records; SRPO, the only
paper without a discoverable arXiv record, links to its official OpenReview
page. Six renamed or older arXiv versions are documented in
`arxiv_overrides.json`. First-page previews come from the original official
OpenReview PDFs. `build_data.py` fails if the published slate, expected link
coverage, canonical titles, tournament schedule, or post hoc human-outcome
counts change. The methodology diagram and human-comparison figures are also
generated from the frozen tournament, shortlist, and Stage 7 analysis rather
than maintained as hand-entered graphics.
