"""Quantitative AI-vs-human agreement metrics over the comparison join.

Two reports, both deterministic (no model calls), built on the same
``load_comparison_rows`` foundation as the divergence gallery:

- **Q-A agreement** (``build_agreement_report``): how well an AI ranking recovers
  the human presentation tier — recall@k / precision@k and ROC-AUC for surfacing
  orals+spotlights, Kendall tau-b vs the graded tier and vs reviewer means, and
  per-tier AI-score distributions. When a strong ranking is supplied it also
  reports the same metrics on the finalist subset (the "layered" view: does the
  stronger judge agree with humans more than the cheap full-coverage signal?).

- **Q-B axis decomposition** (``build_axis_decomposition_report``): which AI axis
  best predicts human honors. Ranks every stored axis by tau-b vs tier and
  highlights the impact-vs-polish contrast (``conventional_acceptance_strength``
  minus ``executive_ac_priority``).

Metric primitives are pure standard library (Kendall tau-b via concordant/
discordant counting, ROC-AUC via the rank-based Mann-Whitney identity).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from icml_ai_ac.analysis.human_comparison import ComparisonRow, tier_counts


HONORED_TIERS = ("oral", "spotlight")


# --- metric primitives ----------------------------------------------------


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    n0 = n * (n - 1) // 2
    tie_x = tie_y = concordant = discordant = 0
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0:
                tie_x += 1
            if dy == 0:
                tie_y += 1
            if dx != 0 and dy != 0:
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom == 0:
        return 0.0
    return round((concordant - discordant) / denom, 4)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_positive = sum(rank for rank, label in zip(ranks, labels) if label)
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(auc, 4)


def distribution_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "median": _quantile(ordered, 0.5),
        "q25": _quantile(ordered, 0.25),
        "q75": _quantile(ordered, 0.75),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return round(ordered[low] * (1 - frac) + ordered[high] * frac, 3)


# --- Q-A: agreement -------------------------------------------------------


def is_honored(row: ComparisonRow, include_award: bool) -> bool:
    return row.tier in HONORED_TIERS or (include_award and row.is_award)


def agreement_metrics(
    rows: list[ComparisonRow],
    *,
    score_of: Callable[[ComparisonRow], float],
    k_values: Iterable[int],
    include_award: bool,
    include_per_tier: bool,
) -> dict[str, Any]:
    scored = [(score_of(row), row) for row in rows]
    scores = [score for score, _ in scored]
    tiers = [float(row.tier_rank) for _, row in scored]

    reviewer_pairs = [(score, row.reviewer_overall) for score, row in scored if row.reviewer_overall is not None]

    discrimination = [(score, row) for score, row in scored if row.tier == "poster" or is_honored(row, include_award)]

    honored_total = sum(1 for _, row in scored if is_honored(row, include_award))
    oral_total = sum(1 for _, row in scored if row.tier == "oral")
    order = sorted(scored, key=lambda pair: (-pair[0], pair[1].paper_id))

    recall_at_k: dict[str, Any] = {}
    for k in k_values:
        top = order[:k]
        captured = sum(1 for _, row in top if is_honored(row, include_award))
        captured_oral = sum(1 for _, row in top if row.tier == "oral")
        recall_at_k[str(k)] = {
            "recall_honored": round(captured / honored_total, 4) if honored_total else None,
            "precision_honored": round(captured / len(top), 4) if top else None,
            "recall_oral": round(captured_oral / oral_total, 4) if oral_total else None,
        }

    metrics: dict[str, Any] = {
        "n": len(rows),
        "honored_count": honored_total,
        "oral_count": oral_total,
        "kendall_tau_b": {
            "vs_tier": kendall_tau_b(scores, tiers),
            "vs_reviewer_overall": kendall_tau_b(
                [score for score, _ in reviewer_pairs], [value for _, value in reviewer_pairs]
            ),
        },
        "roc_auc": {
            "honored_vs_poster": roc_auc(
                [score for score, _ in discrimination],
                [is_honored(row, include_award) for _, row in discrimination],
            ),
            "oral_vs_rest": roc_auc(scores, [row.tier == "oral" for _, row in scored]),
        },
        "recall_at_k": recall_at_k,
    }
    if include_per_tier:
        by_tier: dict[str, Any] = {}
        for tier in ("oral", "spotlight", "poster"):
            values = [row.ai_percentile for _, row in scored if row.tier == tier]
            by_tier[tier] = distribution_summary(values)
        metrics["per_tier_ai_percentile"] = by_tier
    return metrics


def build_agreement_report(
    rows: list[ComparisonRow],
    *,
    k_values: list[int],
    include_award: bool = False,
) -> dict[str, Any]:
    full = agreement_metrics(
        rows,
        score_of=lambda row: row.ai_score,
        k_values=k_values,
        include_award=include_award,
        include_per_tier=True,
    )
    strong_rows = [row for row in rows if row.strong_rank is not None]
    strong = None
    if len(strong_rows) >= 2:
        strong = agreement_metrics(
            strong_rows,
            score_of=lambda row: -float(row.strong_rank),
            k_values=k_values,
            include_award=include_award,
            include_per_tier=False,
        )
    return {
        "analysis": "human_ai_agreement",
        "config": {"k_values": k_values, "honored_includes_award": include_award},
        "counts": {"papers": len(rows), "by_tier": tier_counts(rows)},
        "full_coverage": full,
        "strong_subset": strong,
    }


# --- Q-B: axis decomposition ----------------------------------------------


def build_axis_decomposition_report(
    rows: list[ComparisonRow],
    *,
    include_award: bool = False,
) -> dict[str, Any]:
    axis_names = sorted({name for row in rows for name in row.ai_axes})
    results: list[dict[str, Any]] = []
    for axis in axis_names:
        pairs = [(row.ai_axes[axis], row) for row in rows if axis in row.ai_axes]
        if len(pairs) < 2:
            continue
        scores = [value for value, _ in pairs]
        tiers = [float(row.tier_rank) for _, row in pairs]
        discrimination = [(value, row) for value, row in pairs if row.tier == "poster" or is_honored(row, include_award)]
        reviewer_pairs = [(value, row.reviewer_overall) for value, row in pairs if row.reviewer_overall is not None]
        results.append(
            {
                "axis": axis,
                "n": len(pairs),
                "kendall_tau_b_vs_tier": kendall_tau_b(scores, tiers),
                "roc_auc_honored_vs_poster": roc_auc(
                    [value for value, _ in discrimination],
                    [is_honored(row, include_award) for _, row in discrimination],
                ),
                "kendall_tau_b_vs_reviewer_overall": kendall_tau_b(
                    [value for value, _ in reviewer_pairs], [value for _, value in reviewer_pairs]
                ),
            }
        )
    results.sort(key=lambda item: (item["kendall_tau_b_vs_tier"] is not None, item["kendall_tau_b_vs_tier"] or 0.0), reverse=True)

    def tau_of(name: str) -> float | None:
        return next((item["kendall_tau_b_vs_tier"] for item in results if item["axis"] == name), None)

    executive = tau_of("executive_ac_priority")
    conventional = tau_of("conventional_acceptance_strength")
    delta = round(conventional - executive, 4) if executive is not None and conventional is not None else None
    return {
        "analysis": "ai_axis_decomposition",
        "config": {"honored_includes_award": include_award},
        "axes": results,
        "impact_vs_polish": {
            "executive_ac_priority_tau_b_vs_tier": executive,
            "conventional_acceptance_strength_tau_b_vs_tier": conventional,
            "delta_conventional_minus_executive": delta,
            "note": "Positive delta means AI's model of conventional reviewer strength tracks human tier better than its impact-forward priority.",
        },
    }
