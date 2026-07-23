"""Join AI ranking signals with human outcomes, and build a divergence gallery.

This is the deterministic foundation of the human-vs-AI comparison. It loads:

- an enriched paper manifest (human labels: presentation tier from
  ``extra.is_oral`` / ``extra.is_spotlight``, reviewer means from
  ``extra.openreview_scores``, awards from ``extra.award_labels``), and
- a first-pass AI score JSONL (per-paper ``scores`` schema: the full-coverage AI
  signal via ``scores.scores.executive_ac_priority`` plus rationale text), and
- optionally a strong ranking JSON (tournament / Bradley-Terry) for the finalist
  subset, shown as a cross-reference.

It then reports a full tier x AI-decile crosstab and per-paper divergence
residuals for *every* paper, and selects two rule-defined tails:

- **overlooked gems** — posters the AI ranks near the top, and
- **blind spots** — orals / spotlights / award papers the AI ranks low.

The gallery is an explicit windowed view of the reported distribution (not a
hand-picked set): the same divergence score orders every paper, and the report
records how many cases exist beyond the shown window.

No model calls: this whole module is deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icml_ai_ac.storage import read_jsonl, read_paper_records


HONORED_TIERS = ("oral", "spotlight")

# AI rationale fields surfaced verbatim on each case card (path within `scores`).
RATIONALE_PATHS: dict[str, tuple[str, ...]] = {
    "why_it_might_matter": ("executive_lens", "why_it_might_matter"),
    "sweeping_impact_scenario": ("executive_lens", "sweeping_impact_scenario"),
    "top_paper_case": ("ranking_signals", "top_paper_case"),
    "broad_scientific_impact_argument": ("ranking_signals", "broad_scientific_impact_argument"),
    "dealbreaker_risks": ("ranking_signals", "dealbreaker_risks"),
    "why_not_higher": ("calibration", "why_not_higher"),
    "why_not_lower": ("calibration", "why_not_lower"),
    "reviewer_vs_executive_delta": ("calibration", "reviewer_vs_executive_delta"),
    "human_review_alignment": ("reviewer_lens", "human_review_alignment"),
    "likely_reviewer_concerns": ("reviewer_lens", "likely_reviewer_concerns"),
    "observed_weaknesses": ("evidence_audit", "observed_weaknesses"),
    "contamination_sensitivity": ("uncertainty", "contamination_sensitivity"),
}

# Numeric axes carried on each case so the taxonomy (QL-B) can ground reason
# codes in the impact-vs-polish contrast rather than the AI's prose alone.
AXIS_SNAPSHOT_PATHS: dict[str, tuple[str, ...]] = {
    "executive_ac_priority": ("scores", "executive_ac_priority"),
    "conventional_acceptance_strength": ("scores", "conventional_acceptance_strength"),
    "technical_soundness": ("scores", "technical_soundness"),
    "novelty": ("scores", "novelty"),
    "broader_science_impact_forecast": ("scores", "broader_science_impact_forecast"),
    "ml_field_impact_forecast": ("scores", "ml_field_impact_forecast"),
    "field_building_potential": ("impact_axes", "field_building_potential"),
    "benchmark_or_dataset_value": ("impact_axes", "benchmark_or_dataset_value"),
}


@dataclass(slots=True)
class ComparisonRow:
    paper_id: str
    title: str
    contribution_class: str | None
    # human outcome
    tier: str
    tier_rank: int
    is_award: bool
    award_labels: list[Any]
    reviewer_overall: float | None
    reviewer_soundness: float | None
    reviewer_confidence: float | None
    review_count: int
    # ai signal
    ai_score: float
    ai_reported_percentile: float | None
    ai_rank: int = 0
    ai_percentile: float = 0.0
    strong_rank: int | None = None
    # derived divergence (filled by build_divergence_report)
    human_percentile: float | None = None
    residual: float | None = None
    rationale: dict[str, Any] = field(default_factory=dict)
    ai_axes: dict[str, float] = field(default_factory=dict)


def load_comparison_rows(
    *,
    manifest: Path,
    ai_scores: Path,
    strong_ranking: Path | None = None,
) -> list[ComparisonRow]:
    ai_by_paper = load_ai_scores(ai_scores)
    strong_rank_by_paper = load_strong_ranks(strong_ranking) if strong_ranking else {}

    rows: list[ComparisonRow] = []
    for record in read_paper_records(manifest):
        scores = ai_by_paper.get(record.paper_id)
        if scores is None:
            continue
        tier, tier_rank = tier_of(record.extra)
        reviewer = reviewer_means(record.extra)
        ai_score, reported_percentile = ai_position(scores)
        rows.append(
            ComparisonRow(
                paper_id=record.paper_id,
                title=record.title or "",
                contribution_class=contribution_class_of(scores),
                tier=tier,
                tier_rank=tier_rank,
                is_award=bool(record.extra.get("is_award_paper")),
                award_labels=list(record.extra.get("award_labels") or []),
                reviewer_overall=reviewer["overall"],
                reviewer_soundness=reviewer["soundness"],
                reviewer_confidence=reviewer["confidence"],
                review_count=reviewer["count"],
                ai_score=ai_score,
                ai_reported_percentile=reported_percentile,
                strong_rank=strong_rank_by_paper.get(record.paper_id),
                rationale=extract_rationale(scores),
                ai_axes=extract_axes(scores),
            )
        )

    # Derive AI rank (1 = best) and AI percentile among the loaded accepted papers.
    ai_percentiles = percentile_ranks([row.ai_score for row in rows])
    for row, percentile in zip(rows, ai_percentiles):
        row.ai_percentile = percentile
    for rank, row in enumerate(sorted(rows, key=lambda r: (-r.ai_score, r.paper_id), reverse=False), start=1):
        row.ai_rank = rank
    return rows


def build_divergence_report(
    rows: list[ComparisonRow],
    *,
    human_axis: str = "tier",
    top_frac: float = 0.10,
    cases_per_direction: int = 20,
    min_reviews: int = 0,
) -> dict[str, Any]:
    if human_axis not in {"tier", "reviewer"}:
        raise ValueError("human_axis must be 'tier' or 'reviewer'")
    if not 0 < top_frac < 1:
        raise ValueError("top_frac must be in (0, 1)")

    eligible = [row for row in rows if row.review_count >= min_reviews]
    human_values = human_metric_values(eligible, human_axis)
    scored = [row for row, value in zip(eligible, human_values) if value is not None]
    human_percentiles = percentile_ranks([value for value in human_values if value is not None])
    for row, percentile in zip(scored, human_percentiles):
        row.human_percentile = percentile
        row.residual = round(row.ai_percentile - percentile, 3)

    residual_rows = [row for row in scored if row.residual is not None]
    gems_all = sorted(
        (row for row in residual_rows if row.tier == "poster" and not row.is_award),
        key=lambda r: (-(r.residual or 0.0), -r.ai_percentile, r.paper_id),
    )
    blind_all = sorted(
        (row for row in residual_rows if row.tier in HONORED_TIERS or row.is_award),
        key=lambda r: ((r.residual or 0.0), r.ai_percentile, r.paper_id),
    )
    gems = gems_all[:cases_per_direction]
    blind_spots = blind_all[:cases_per_direction]

    return {
        "analysis": "human_ai_divergence_gallery",
        "config": {
            "human_axis": human_axis,
            "top_frac": top_frac,
            "cases_per_direction": cases_per_direction,
            "min_reviews": min_reviews,
        },
        "counts": {
            "papers": len(rows),
            "scored_papers": len(residual_rows),
            "by_tier": tier_counts(rows),
            "award_papers": sum(1 for row in rows if row.is_award),
        },
        "tier_ai_decile_crosstab": tier_ai_decile_crosstab(rows, top_frac=top_frac),
        "residual_summary": residual_summary(residual_rows),
        "overlooked_gems": {
            "description": "Posters the AI ranks far above their human tier (largest positive AI-minus-human residual).",
            "total_matching": len(gems_all),
            "shown": len(gems),
            "cases": [case_card(row) for row in gems],
        },
        "blind_spots": {
            "description": "Orals/spotlights/awards the AI ranks low (largest negative residual).",
            "total_matching": len(blind_all),
            "shown": len(blind_spots),
            "cases": [case_card(row) for row in blind_spots],
        },
    }


# --- human-side helpers ---------------------------------------------------


def tier_of(extra: dict[str, Any]) -> tuple[str, int]:
    if extra.get("is_oral"):
        return "oral", 2
    if extra.get("is_spotlight"):
        return "spotlight", 1
    return "poster", 0


def reviewer_means(extra: dict[str, Any]) -> dict[str, Any]:
    scores = extra.get("openreview_scores") if isinstance(extra.get("openreview_scores"), dict) else {}
    return {
        "overall": as_float(scores.get("overall_mean")),
        "soundness": as_float(scores.get("soundness_mean")),
        "confidence": as_float(scores.get("confidence_mean")),
        "count": int(scores.get("review_count") or 0),
    }


# --- ai-side helpers ------------------------------------------------------


def load_ai_scores(path: Path) -> dict[str, dict[str, Any]]:
    """One representative pass-1 score dict per paper (highest executive priority)."""
    best: dict[str, dict[str, Any]] = {}
    best_priority: dict[str, float] = {}
    for row in read_jsonl(path):
        if row.get("status") not in {None, "ok"}:
            continue
        paper_id = str(row.get("paper_id") or "")
        scores = row.get("scores")
        if not paper_id or not isinstance(scores, dict):
            continue
        priority, _ = ai_position(scores)
        if paper_id not in best or priority > best_priority[paper_id]:
            best[paper_id] = scores
            best_priority[paper_id] = priority
    return best


def ai_position(scores: dict[str, Any]) -> tuple[float, float | None]:
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    priority = as_float(core.get("executive_ac_priority"))
    if priority is None:
        priority = as_float(core.get("overall_significance")) or 0.0
    reported = as_float(calibration.get("estimated_percentile_among_accepted_papers"))
    return float(priority), reported


def contribution_class_of(scores: dict[str, Any]) -> str | None:
    profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
    value = profile.get("primary_contribution_class")
    return str(value) if isinstance(value, str) and value else None


def extract_rationale(scores: dict[str, Any]) -> dict[str, Any]:
    rationale: dict[str, Any] = {}
    for label, path in RATIONALE_PATHS.items():
        value = nested_get(scores, *path)
        if value not in (None, "", [], {}):
            rationale[label] = value
    return rationale


def extract_axes(scores: dict[str, Any]) -> dict[str, float]:
    axes: dict[str, float] = {}
    for label, path in AXIS_SNAPSHOT_PATHS.items():
        value = as_float(nested_get(scores, *path))
        if value is not None:
            axes[label] = value
    return axes


def load_strong_ranks(path: Path) -> dict[str, int]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    ranked = payload.get("ranked_papers") if isinstance(payload, dict) else None
    ranks: dict[str, int] = {}
    if isinstance(ranked, list):
        for row in ranked:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("paper_id") or "")
            rank = row.get("rank")
            if paper_id and isinstance(rank, int):
                ranks[paper_id] = rank
    return ranks


# --- aggregation helpers --------------------------------------------------


def human_metric_values(rows: list[ComparisonRow], human_axis: str) -> list[float | None]:
    if human_axis == "tier":
        # Awards are the strongest human honor; rank them above orals.
        return [float(3 if row.is_award else row.tier_rank) for row in rows]
    return [row.reviewer_overall for row in rows]


def percentile_ranks(values: list[float]) -> list[float]:
    """Mean percentile rank (0-100); higher value -> higher percentile, ties averaged."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [50.0]
    result: list[float] = []
    for value in values:
        below = sum(1 for other in values if other < value)
        equal = sum(1 for other in values if other == value)
        result.append(round(100.0 * (below + 0.5 * equal) / n, 3))
    return result


def tier_counts(rows: list[ComparisonRow]) -> dict[str, int]:
    counts = {"oral": 0, "spotlight": 0, "poster": 0}
    for row in rows:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    return counts


def tier_ai_decile_crosstab(rows: list[ComparisonRow], *, top_frac: float) -> dict[str, Any]:
    cut = 100.0 * (1.0 - top_frac)
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = "ai_top" if row.ai_percentile >= cut else "ai_rest"
        table.setdefault(row.tier, {"ai_top": 0, "ai_rest": 0})[bucket] += 1
    return {"ai_top_threshold_percentile": round(cut, 3), "table": table}


def residual_summary(rows: list[ComparisonRow]) -> dict[str, Any]:
    residuals = sorted(row.residual for row in rows if row.residual is not None)
    if not residuals:
        return {"count": 0}
    return {
        "count": len(residuals),
        "min": residuals[0],
        "q25": quantile(residuals, 0.25),
        "median": quantile(residuals, 0.50),
        "q75": quantile(residuals, 0.75),
        "max": residuals[-1],
    }


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    frac = position - low
    return round(sorted_values[low] * (1 - frac) + sorted_values[high] * frac, 3)


def case_card(row: ComparisonRow) -> dict[str, Any]:
    return {
        "paper_id": row.paper_id,
        "title": row.title,
        "contribution_class": row.contribution_class,
        "human": {
            "tier": row.tier,
            "is_award": row.is_award,
            "award_labels": row.award_labels,
            "reviewer_overall": row.reviewer_overall,
            "reviewer_soundness": row.reviewer_soundness,
            "reviewer_confidence": row.reviewer_confidence,
            "review_count": row.review_count,
        },
        "ai": {
            "executive_ac_priority": round(row.ai_score, 4),
            "rank": row.ai_rank,
            "percentile": row.ai_percentile,
            "reported_percentile": row.ai_reported_percentile,
            "strong_rank": row.strong_rank,
        },
        "divergence": {"residual": row.residual, "human_percentile": row.human_percentile},
        "ai_axes": row.ai_axes,
        "ai_rationale": row.rationale,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# AI-vs-human divergence gallery", ""]
    counts = report["counts"]
    lines.append(
        f"Papers: {counts['papers']} (oral={counts['by_tier'].get('oral', 0)}, "
        f"spotlight={counts['by_tier'].get('spotlight', 0)}, poster={counts['by_tier'].get('poster', 0)}, "
        f"awards={counts['award_papers']}). Human axis: {report['config']['human_axis']}."
    )
    lines.append("")
    for key, heading in (("overlooked_gems", "Overlooked gems"), ("blind_spots", "AI blind spots")):
        section = report[key]
        lines.append(f"## {heading} (showing {section['shown']} of {section['total_matching']})")
        lines.append(f"_{section['description']}_")
        lines.append("")
        for card in section["cases"]:
            lines.extend(render_case_markdown(card))
        lines.append("")
    return "\n".join(lines)


def render_case_markdown(card: dict[str, Any]) -> list[str]:
    human = card["human"]
    ai = card["ai"]
    lines = [
        f"### {card['title'] or card['paper_id']} (`{card['paper_id']}`)",
        f"- **Human:** {human['tier']}"
        + (", award" if human["is_award"] else "")
        + (f", reviewer overall {human['reviewer_overall']}" if human["reviewer_overall"] is not None else ""),
        f"- **AI:** rank {ai['rank']} (percentile {ai['percentile']}, executive priority {ai['executive_ac_priority']})"
        + (f", strong rank {ai['strong_rank']}" if ai["strong_rank"] is not None else ""),
        f"- **Residual (AI - human):** {card['divergence']['residual']}",
    ]
    for label in ("sweeping_impact_scenario", "top_paper_case", "reviewer_vs_executive_delta", "why_not_higher", "dealbreaker_risks"):
        value = card["ai_rationale"].get(label)
        if value:
            lines.append(f"- _{label}:_ {inline(value)}")
    lines.append("")
    return lines


# --- small utilities ------------------------------------------------------


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_get(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def inline(value: Any, limit: int = 300) -> str:
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."
