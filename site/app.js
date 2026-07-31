(() => {
  "use strict";

  const data = window.ICML_AI_AC_DATA;
  if (!data || !Array.isArray(data.papers)) {
    throw new Error("Missing generated site data. Run: python3 site/build_data.py");
  }

  const classLabels = {
    analysis_position: "Analysis / position",
    application_method: "Application method",
    benchmark_dataset: "Benchmark / dataset",
    core_ml_algorithm: "Core ML algorithm",
    infrastructure_systems: "Infrastructure / systems",
    safety_governance_eval: "Safety / governance / eval",
    scientific_modeling_tool: "Scientific modeling tool",
    theory: "Theory",
    other: "Other",
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function externalLink(href, text, className = "") {
    const link = element("a", className, text);
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    return link;
  }

  function paperPanel(paper, className, context) {
    const destination = paper.links.primaryKind === "arxiv" ? "arXiv" : "OpenReview";
    const link = externalLink(paper.links.primary, undefined, className);
    link.setAttribute("aria-label", `${context}: ${paper.title}. View on ${destination}.`);
    return link;
  }

  function paperLinkCue(paper) {
    const text = paper.links.primaryKind === "arxiv" ? "View on arXiv" : "Official paper";
    return element("span", "paper-link-cue", text);
  }

  function humanTierLabel(tier) {
    if (tier === "oral") return "Human oral";
    if (tier === "spotlight") return "Human spotlight";
    return "Human poster";
  }

  function classLabel(value) {
    return classLabels[value] || value.replaceAll("_", " ");
  }

  function svgElement(tag, attributes = {}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function starPath(cx, cy, outerRadius = 7, innerRadius = 3.2) {
    const points = [];
    for (let index = 0; index < 10; index += 1) {
      const radius = index % 2 === 0 ? outerRadius : innerRadius;
      const angle = -Math.PI / 2 + (index * Math.PI) / 5;
      points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
    }
    return `M ${points.join(" L ")} Z`;
  }

  function renderMethodDiagram() {
    const container = document.querySelector("#method-diagram");
    const flow = element("div", "method-flow");
    data.method.stages.forEach((stage, index) => {
      const node = element("div", `method-node method-${stage.tone}`);
      node.append(
        element("span", "method-stage", `Stage ${stage.stage}`),
        element("strong", "method-count", stage.count.toLocaleString()),
        element("span", "method-title", stage.title),
        element("span", "method-models", stage.models),
        element("small", "method-detail", stage.detail),
      );
      flow.append(node);
      if (index < data.method.stages.length - 1) {
        const arrow = element("div", "method-arrow");
        arrow.setAttribute("aria-hidden", "true");
        flow.append(arrow);
      }
    });
    const phaseLabels = element("div", "method-phase-labels");
    phaseLabels.append(
      element("span", "", "Progressive paper evaluation"),
      element("span", "", "Hybrid tournament"),
    );
    container.append(flow, phaseLabels);
  }

  function renderRecognitionTrajectory() {
    const stages = data.humanComparison.stages;
    const width = 1120;
    const height = 430;
    const left = 130;
    const right = 1070;
    const top = 52;
    const baseline = 255;
    const maxRate = 0.45;
    const x = (index) => left + (index * (right - left)) / (stages.length - 1);
    const y = (rate) => baseline - (rate / maxRate) * (baseline - top);
    const svg = svgElement("svg", {
      class: "recognition-svg",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": stages
        .map(
          (stage) =>
            `${stage.label}: ${(stage.honoredRate * 100).toFixed(1)} percent human oral or spotlight, ${stage.awards} award papers retained`,
        )
        .join(". "),
    });

    [0, 0.2, 0.4].forEach((rate) => {
      const gridY = y(rate);
      svg.append(
        svgElement("line", { class: "recognition-grid", x1: left, x2: right, y1: gridY, y2: gridY }),
        svgElement(
          "text",
          { class: "recognition-axis-label", x: left - 18, y: gridY + 4, "text-anchor": "end" },
          `${Math.round(rate * 100)}%`,
        ),
      );
    });
    svg.append(
      svgElement("text", { class: "recognition-axis-title", x: left, y: 24 }, "Share designated oral or spotlight by ICML"),
    );

    const pointPairs = stages.map((stage, index) => [x(index), y(stage.honoredRate)]);
    const area = [
      `M ${pointPairs[0][0]} ${baseline}`,
      ...pointPairs.map(([pointX, pointY]) => `L ${pointX} ${pointY}`),
      `L ${pointPairs[pointPairs.length - 1][0]} ${baseline}`,
      "Z",
    ].join(" ");
    const line = pointPairs
      .map(([pointX, pointY], index) => `${index === 0 ? "M" : "L"} ${pointX} ${pointY}`)
      .join(" ");
    svg.append(
      svgElement("path", { class: "recognition-area", d: area }),
      svgElement("path", { class: "recognition-line", d: line }),
    );

    stages.forEach((stage, index) => {
      const pointX = x(index);
      const pointY = y(stage.honoredRate);
      svg.append(
        svgElement("circle", { class: "recognition-point", cx: pointX, cy: pointY, r: 7 }),
        svgElement(
          "text",
          { class: "recognition-rate", x: pointX, y: pointY - 16, "text-anchor": "middle" },
          `${(stage.honoredRate * 100).toFixed(1)}%`,
        ),
        svgElement(
          "text",
          { class: "recognition-stage-label", x: pointX, y: 286, "text-anchor": "middle" },
          stage.label,
        ),
        svgElement(
          "text",
          { class: "recognition-paper-count", x: pointX, y: 307, "text-anchor": "middle" },
          `${stage.papers.toLocaleString()} papers`,
        ),
      );

      const starSpacing = 12;
      const firstStarX = pointX - ((stage.awards - 1) * starSpacing) / 2;
      for (let star = 0; star < stage.awards; star += 1) {
        svg.append(
          svgElement("path", {
            class: "recognition-star",
            d: starPath(firstStarX + star * starSpacing, 365),
          }),
        );
      }
      svg.append(
        svgElement(
          "text",
          { class: "recognition-award-count", x: pointX, y: 397, "text-anchor": "middle" },
          `${stage.awards} retained`,
        ),
      );
    });
    svg.append(
      svgElement("line", { class: "recognition-award-rule", x1: left, x2: right, y1: 335, y2: 335 }),
      svgElement(
        "text",
        { class: "recognition-award-label", x: left, y: 327 },
        "Official awards",
      ),
    );
    document.querySelector("#recognition-trajectory").append(svg);
  }

  function renderPlayoffRankMap() {
    const papers = data.humanComparison.playoffOutcomes;
    const width = 1120;
    const height = 310;
    const left = 150;
    const right = 1080;
    const rows = { award: 66, oral: 120, spotlight: 178, poster: 236 };
    const x = (rank) => left + ((rank - 1) * (right - left)) / 59;
    const svg = svgElement("svg", {
      class: "rank-map-svg",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Human presentation outcomes positioned by AI rank from one to sixty",
    });
    const topTenEnd = x(10) + (x(2) - x(1)) / 2;
    svg.append(
      svgElement("rect", {
        class: "rank-map-top-ten",
        x: left - 7,
        y: 32,
        width: topTenEnd - left + 7,
        height: 224,
      }),
      svgElement("text", { class: "rank-map-band-label", x: left, y: 23 }, "AI top 10"),
    );

    [
      ["Official award", rows.award],
      ["Human oral", rows.oral],
      ["Human spotlight", rows.spotlight],
      ["Human poster", rows.poster],
    ].forEach(([label, rowY]) => {
      svg.append(
        svgElement("line", { class: "rank-map-row", x1: left, x2: right, y1: rowY, y2: rowY }),
        svgElement("text", { class: "rank-map-row-label", x: left - 18, y: rowY + 4, "text-anchor": "end" }, label),
      );
    });

    for (let rank = 10; rank <= 60; rank += 10) {
      const tickX = x(rank);
      svg.append(
        svgElement("line", { class: "rank-map-tick", x1: tickX, x2: tickX, y1: 32, y2: 256 }),
        svgElement("text", { class: "rank-map-tick-label", x: tickX, y: 282, "text-anchor": "middle" }, String(rank)),
      );
    }
    svg.append(svgElement("text", { class: "rank-map-tick-label", x: x(1), y: 282, "text-anchor": "middle" }, "1"));

    papers.forEach((paper) => {
      const mark = svgElement("g", { class: `rank-mark rank-${paper.tier}` });
      mark.append(
        svgElement(
          "title",
          {},
          `AI #${paper.rank} · Human ${paper.tier}${paper.actualAward ? " · Official award" : ""} · ${paper.title}`,
        ),
      );
      if (paper.tier === "spotlight") {
        const pointX = x(paper.rank);
        mark.append(
          svgElement("rect", {
            x: pointX - 4.5,
            y: rows.spotlight - 4.5,
            width: 9,
            height: 9,
            transform: `rotate(45 ${pointX} ${rows.spotlight})`,
          }),
        );
      } else {
        mark.append(
          svgElement("circle", {
            cx: x(paper.rank),
            cy: rows[paper.tier],
            r: paper.tier === "oral" ? 6 : 3.2,
          }),
        );
      }
      svg.append(mark);
      if (paper.actualAward) {
        svg.append(
          svgElement("line", {
            class: "rank-map-award-link",
            x1: x(paper.rank),
            x2: x(paper.rank),
            y1: rows.award + 9,
            y2: rows[paper.tier] - 9,
          }),
          svgElement("path", {
            class: "rank-map-award-star",
            d: starPath(x(paper.rank), rows.award, 10, 4.6),
          }),
        );
      }
    });
    document.querySelector("#playoff-rank-map").append(svg);

    const oralRanks = papers.filter((paper) => paper.tier === "oral").map((paper) => paper.rank);
    const spotlightRanks = papers
      .filter((paper) => paper.tier === "spotlight")
      .map((paper) => paper.rank);
    const awardRanks = papers.filter((paper) => paper.actualAward).map((paper) => paper.rank);
    document.querySelector("#playoff-rank-caption").textContent =
      `Human orals landed at AI ranks ${oralRanks.join(", ")}; spotlights at ${spotlightRanks.join(", ")}. ` +
      `The only official award paper in the playoff was AI #${awardRanks[0]}.`;
  }

  function renderMosaic() {
    const container = document.querySelector("#paper-mosaic");
    data.papers.slice(0, 5).forEach((paper) => {
      const figure = element("figure", "paper-preview");
      const link = paperPanel(paper, "paper-preview-link", `AI rank ${paper.rank}`);
      const image = document.createElement("img");
      image.src = `assets/paper-${paper.paperId}.jpg`;
      image.alt = `Official first page of ${paper.title}`;
      image.loading = paper.rank <= 3 ? "eager" : "lazy";
      const caption = element("figcaption");
      caption.append(
        element("span", "preview-rank", `AI #${paper.rank}`),
        element("span", "preview-title", paper.title),
      );
      link.append(image, caption);
      figure.append(link);
      container.append(figure);
    });
  }

  function renderTopTen() {
    const container = document.querySelector("#top-ten");
    data.papers.slice(0, 10).forEach((paper) => {
      const row = paperPanel(paper, "rank-row paper-panel", `AI rank ${paper.rank}`);
      const rank = element("div", "rank-number", String(paper.rank).padStart(2, "0"));
      const content = element("div", "rank-content");
      const title = element("div", "rank-title", paper.title);
      const meta = element("div", "rank-meta");
      meta.append(
        element("span", `class-tag class-${paper.primaryClass}`, classLabel(paper.primaryClass)),
        element("span", `human-tag human-${paper.human.tier}`, humanTierLabel(paper.human.tier)),
      );
      if (paper.human.reviewerMean !== null) {
        meta.append(element("span", "review-score", `Human review ${paper.human.reviewerMean.toFixed(2)}`));
      }
      meta.append(paperLinkCue(paper));
      content.append(title, element("p", "rank-summary", paper.whyRankedHere), meta);
      row.append(rank, content);
      container.append(row);
    });
  }

  function awardCard(paper) {
    const card = paperPanel(paper, "outstanding-card paper-panel", `Outstanding Paper, AI rank ${paper.rank}`);
    const imageWrap = element("div", "award-paper-image");
    const image = document.createElement("img");
    image.src = `assets/paper-${paper.paperId}.jpg`;
    image.alt = `Official first page of ${paper.title}`;
    imageWrap.append(image);

    const copy = element("div", "award-paper-copy");
    copy.append(
      element("div", "award-rank", `AI rank ${paper.rank}`),
      element("div", "award-title", paper.title),
      element("p", "award-why", paper.whyRankedHere),
    );
    const scores = element("div", "score-strip");
    [
      ["Technical", paper.scores.technical],
      ["ML impact", paper.scores.mlImpact],
      ["Science", paper.scores.scienceImpact],
      ["Evidence", paper.scores.evidence],
    ].forEach(([label, value]) => {
      const item = element("div", "score-item");
      item.append(element("strong", "", value.toFixed(1)), element("span", "", label));
      scores.append(item);
    });
    const posthoc = element("div", "posthoc-line");
    posthoc.append(
      element("span", "", "Post hoc human outcome"),
      element("strong", "", humanTierLabel(paper.human.tier).replace("Human ", "")),
    );
    if (paper.human.reviewerMean !== null) {
      posthoc.append(element("span", "", `review mean ${paper.human.reviewerMean.toFixed(2)}`));
    }
    posthoc.append(paperLinkCue(paper));
    copy.append(scores, posthoc);
    card.append(imageWrap, copy);
    return card;
  }

  function renderAwards() {
    const outstanding = document.querySelector("#outstanding-papers");
    data.papers.slice(0, 2).forEach((paper) => outstanding.append(awardCard(paper)));

    const mentions = document.querySelector("#honorable-mentions");
    data.papers.slice(2, 7).forEach((paper) => {
      const row = paperPanel(paper, "mention-row paper-panel", `Honorable Mention, AI rank ${paper.rank}`);
      const contribution = element("div", "mention-class", classLabel(paper.primaryClass));
      contribution.append(paperLinkCue(paper));
      row.append(
        element("div", "mention-rank", `#${paper.rank}`),
        element("div", "mention-title", paper.title),
        contribution,
        element("p", "mention-why", paper.whyRankedHere),
      );
      mentions.append(row);
    });
  }

  function oralRow(paper) {
    const row = paperPanel(paper, "oral-row paper-panel", `AI oral, rank ${paper.rank}`);
    const rank = element("div", "oral-rank", String(paper.rank).padStart(2, "0"));
    const main = element("div", "oral-main");
    const title = element("div", "oral-title", paper.title);
    const meta = element("div", "oral-meta");
    meta.append(
      element("span", `class-tag class-${paper.primaryClass}`, classLabel(paper.primaryClass)),
      element("span", `human-tag human-${paper.human.tier}`, humanTierLabel(paper.human.tier)),
      paperLinkCue(paper),
    );
    main.append(title, element("p", "oral-summary", paper.whyRankedHere), meta);
    const impact = element("div", "oral-impact");
    impact.append(
      element("strong", "", paper.scores.mlImpact.toFixed(1)),
      element("span", "", "ML impact"),
    );
    row.append(rank, main, impact);
    return row;
  }

  function renderOrals() {
    const container = document.querySelector("#oral-list");
    const search = document.querySelector("#oral-search").value.trim().toLowerCase();
    const selectedClass = document.querySelector("#oral-class-filter").value;
    const filtered = data.papers.filter((paper) => {
      const matchesText =
        !search ||
        paper.title.toLowerCase().includes(search) ||
        paper.paperId.toLowerCase().includes(search);
      return matchesText && (!selectedClass || paper.primaryClass === selectedClass);
    });
    container.replaceChildren(...filtered.map(oralRow));
    document.querySelector("#oral-result-count").textContent =
      `${filtered.length} of ${data.papers.length} papers`;
  }

  function populateFilters() {
    const select = document.querySelector("#oral-class-filter");
    const classes = [...new Set(data.papers.map((paper) => paper.primaryClass))].sort((a, b) =>
      classLabel(a).localeCompare(classLabel(b)),
    );
    classes.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = classLabel(value);
      select.append(option);
    });
    select.addEventListener("change", renderOrals);
    document.querySelector("#oral-search").addEventListener("input", renderOrals);
  }

  function showView(view, updateHash = true) {
    const target = view === "awards" ? "awards" : "story";
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      const active = panel.dataset.viewPanel === target;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-view]").forEach((tab) => {
      const active = tab.dataset.view === target;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (updateHash) history.replaceState(null, "", `#${target}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  document.querySelectorAll("[data-view]").forEach((tab) => {
    tab.addEventListener("click", () => showView(tab.dataset.view));
  });
  document.querySelectorAll("[data-go-awards]").forEach((button) => {
    button.addEventListener("click", () => showView("awards"));
  });
  document.querySelectorAll("[data-go-story]").forEach((button) => {
    button.addEventListener("click", () => showView("story"));
  });
  window.addEventListener("hashchange", () => showView(location.hash.slice(1), false));

  renderMosaic();
  renderMethodDiagram();
  renderTopTen();
  renderRecognitionTrajectory();
  renderPlayoffRankMap();
  renderAwards();
  populateFilters();
  renderOrals();
  showView(location.hash.slice(1), false);
})();
