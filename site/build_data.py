"""Build the static blog data bundle from frozen experiment artifacts."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT = ROOT / "data/scores/icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json"
TOURNAMENT_SEED = ROOT / "data/scores/icml_2026_tournament_seed_union172.json"
FINALISTS = ROOT / "data/scores/icml_2026_finalists250.jsonl"
MANIFEST = ROOT / "data/metadata/icml_2026_scoring_manifest.jsonl"
ARXIV_MATCHES = ROOT / "data/metadata/icml_2026_ai_orals60_with_arxiv.jsonl"
ARXIV_OVERRIDES = ROOT / "site/arxiv_overrides.json"
SUMMARY = ROOT / "data/evals/icml_2026_stage7_summary.json"
OUTPUT = ROOT / "site/data.js"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def human_tier(extra: dict[str, Any]) -> str:
    if extra.get("is_oral"):
        return "oral"
    if extra.get("is_spotlight"):
        return "spotlight"
    return "poster"


def track_name(extra: dict[str, Any]) -> str:
    source = str(extra.get("sourceurl") or "")
    if source.endswith("/Conference"):
        return "main"
    if "Position_Paper_Track" in source:
        return "position"
    return "journal"


def normalized_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.casefold().split())


def primary_link(
    paper_id: str,
    row: dict[str, Any],
    source: dict[str, Any],
    overrides: dict[str, dict[str, str]],
) -> dict[str, str | None]:
    extra = row.get("extra") or {}
    status = extra.get("arxiv_resolution_status")
    confidence = extra.get("arxiv_match_confidence")
    arxiv_id = extra.get("arxiv_id")
    if status == "matched" and confidence in {"high", "exact"} and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        return {"primary": url, "primaryKind": "arxiv", "arxiv": url}

    override = overrides.get(paper_id)
    if override:
        url = f"https://arxiv.org/abs/{override['arxiv_id']}"
        return {"primary": url, "primaryKind": "arxiv", "arxiv": url}

    openreview = (source.get("extra") or {}).get("paper_url")
    if not openreview:
        raise ValueError(f"Paper {paper_id} has neither an arXiv nor official paper URL")
    return {"primary": openreview, "primaryKind": "openreview", "arxiv": None}


def outcome_counts(
    paper_ids: list[str],
    manifest: dict[str, dict[str, Any]],
    *,
    main_track_only: bool = False,
) -> dict[str, Any]:
    counts = {"papers": 0, "oral": 0, "spotlight": 0, "poster": 0, "awards": 0}
    for paper_id in paper_ids:
        row = manifest[paper_id]
        extra = row.get("extra") or {}
        if main_track_only and track_name(extra) != "main":
            continue
        counts["papers"] += 1
        counts[human_tier(extra)] += 1
        counts["awards"] += int(bool(extra.get("is_award_paper")))
    counts["honored"] = counts["oral"] + counts["spotlight"]
    counts["honoredRate"] = counts["honored"] / counts["papers"]
    return counts


def build_human_comparison(
    manifest: dict[str, dict[str, Any]],
    finalists: list[dict[str, Any]],
    tournament_seed: dict[str, Any],
    ranked: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    main_track_ids = [
        paper_id
        for paper_id, row in manifest.items()
        if track_name(row.get("extra") or {}) == "main"
    ]
    finalist_ids = [str(row["paper_id"]) for row in finalists]
    seed_rows = tournament_seed["ranked_papers"]
    swiss_ids = [
        str(row["paper_id"])
        for row in seed_rows
        if row.get("tournament_pool_reasons")
    ]
    if len(swiss_ids) != tournament_seed["tournament_pool_count"] or len(swiss_ids) != 172:
        raise ValueError("Expected the frozen 172-paper Swiss union")

    ranked_ids = [str(row["paper_id"]) for row in ranked]
    stage_specs = [
        ("main_track", "Main track", main_track_ids, True),
        ("finalists", "AI finalists", finalist_ids, False),
        ("swiss_pool", "Swiss pool", swiss_ids, False),
        ("all_pairs_playoff", "All-pairs playoff", ranked_ids[:60], False),
        ("final_top_20", "AI top 20", ranked_ids[:20], False),
        ("final_top_10", "AI top 10", ranked_ids[:10], False),
    ]
    stages = []
    for key, label, paper_ids, main_track_only in stage_specs:
        stages.append(
            {
                "key": key,
                "label": label,
                **outcome_counts(
                    paper_ids,
                    manifest,
                    main_track_only=main_track_only,
                ),
            }
        )

    expected = {
        "finalists": summary["pipeline_sets"][0],
        "swiss_pool": summary["pipeline_sets"][1],
        "all_pairs_playoff": summary["pipeline_sets"][2],
        "final_top_20": summary["pipeline_sets"][3],
        "final_top_10": summary["pipeline_sets"][4],
    }
    for stage in stages[1:]:
        reference = expected[stage["key"]]
        if stage["papers"] != reference["papers"] or stage["honored"] != reference["honored"]:
            raise ValueError(f"Human-outcome stage mismatch for {stage['key']}")
        if stage["awards"] != reference["award_papers"]:
            raise ValueError(f"Award-paper stage mismatch for {stage['key']}")

    playoff_outcomes = []
    for row in ranked[:60]:
        paper_id = str(row["paper_id"])
        source = manifest[paper_id]
        extra = source.get("extra") or {}
        playoff_outcomes.append(
            {
                "rank": row["rank"],
                "paperId": paper_id,
                "title": row["title"],
                "tier": human_tier(extra),
                "actualAward": bool(extra.get("is_award_paper")),
            }
        )

    award_papers = []
    swiss_id_set = set(swiss_ids)
    finalist_id_set = set(finalist_ids)
    rank_by_id = {paper_id: index for index, paper_id in enumerate(ranked_ids, 1)}
    for paper_id in main_track_ids:
        source = manifest[paper_id]
        extra = source.get("extra") or {}
        if not extra.get("is_award_paper"):
            continue
        rank = rank_by_id.get(paper_id)
        if rank is not None and rank <= 10:
            highest_stage = "top_10"
        elif rank is not None and rank <= 60:
            highest_stage = "playoff"
        elif paper_id in swiss_id_set:
            highest_stage = "swiss_pool"
        elif paper_id in finalist_id_set:
            highest_stage = "finalists"
        else:
            highest_stage = "main_track"
        award_papers.append(
            {
                "paperId": paper_id,
                "title": source["title"],
                "highestStage": highest_stage,
                "aiRank": rank,
            }
        )

    if len(award_papers) != 7:
        raise ValueError(f"Expected seven main-track award papers, found {len(award_papers)}")
    return {
        "stages": stages,
        "playoffOutcomes": playoff_outcomes,
        "officialAwardPapers": award_papers,
    }


def build_method_diagram(tournament: dict[str, Any]) -> dict[str, Any]:
    schedule = tournament["tournament_summary"]["schedule"]
    swiss_pairs = sum(
        row["pair_count"] for row in schedule if row["stage"] == "swiss"
    )
    new_playoff_pairs = sum(
        row["pair_count"] for row in schedule if row["stage"] == "playoff_all_pairs"
    )
    complete_playoff_pairs = 60 * 59 // 2
    if (swiss_pairs, new_playoff_pairs, complete_playoff_pairs) != (860, 1587, 1770):
        raise ValueError("Unexpected production tournament schedule")
    return {
        "stages": [
            {
                "stage": "00-01",
                "count": 6617,
                "title": "Blind and route",
                "models": "Gemini 3.1 Flash Lite",
                "detail": "Identity-redacted first nine pages",
                "tone": "source",
            },
            {
                "stage": "02",
                "count": 6617,
                "title": "Cheap recall ensemble",
                "models": "Nemotron · Gemini Lite · Luna · Grok",
                "detail": "Four judges, two contexts each",
                "tone": "cheap",
            },
            {
                "stage": "03-04",
                "count": 1442,
                "title": "Strong semifinal",
                "models": "Terra + Sonnet",
                "detail": "Two listwise partitions per judge",
                "tone": "strong",
            },
            {
                "stage": "05",
                "count": 250,
                "title": "Frontier PDF panel",
                "models": "Sol + Fable + Gemini Pro",
                "detail": "750 independent judgment cards",
                "tone": "frontier",
            },
            {
                "stage": "06A",
                "count": 172,
                "title": "Swiss pool",
                "models": "Sol pairwise judge",
                "detail": f"10 rounds · {swiss_pairs:,} pairs",
                "tone": "swiss",
            },
            {
                "stage": "06B",
                "count": 60,
                "title": "All-pairs playoff",
                "models": "Sol pairwise judge",
                "detail": f"{complete_playoff_pairs:,} complete pair graph",
                "tone": "playoff",
            },
        ],
        "swissPairs": swiss_pairs,
        "newPlayoffPairs": new_playoff_pairs,
        "completePlayoffPairs": complete_playoff_pairs,
    }


def build() -> dict[str, Any]:
    tournament = json.loads(TOURNAMENT.read_text(encoding="utf-8"))
    tournament_seed = json.loads(TOURNAMENT_SEED.read_text(encoding="utf-8"))
    finalists = read_jsonl(FINALISTS)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = {row["paper_id"]: row for row in read_jsonl(MANIFEST)}
    arxiv_matches = {row["paper_id"]: row for row in read_jsonl(ARXIV_MATCHES)}
    arxiv_overrides = json.loads(ARXIV_OVERRIDES.read_text(encoding="utf-8"))
    ranked = tournament["ranked_papers"]
    playoff = ranked[:60]

    if [row["rank"] for row in playoff] != list(range(1, 61)):
        raise ValueError("Expected contiguous final playoff ranks 1-60")
    if any(
        row.get("tournament_stats", {}).get("ranking_source_stage") != "playoff_all_pairs"
        for row in playoff
    ):
        raise ValueError("Every published AI oral must come from the dense all-pairs playoff")

    papers: list[dict[str, Any]] = []
    for row in playoff:
        paper_id = row["paper_id"]
        source = manifest[paper_id]
        arxiv_match = arxiv_matches.get(paper_id)
        if arxiv_match is None:
            raise ValueError(f"Missing arXiv resolution row for paper {paper_id}")
        if normalized_title(source["title"]) != normalized_title(row["title"]):
            raise ValueError(f"Title mismatch for paper {paper_id}")
        if normalized_title(source["title"]) != normalized_title(arxiv_match["title"]):
            raise ValueError(f"arXiv resolution title mismatch for paper {paper_id}")
        extra = source["extra"]
        review = extra.get("openreview_scores") or {}
        papers.append(
            {
                "rank": row["rank"],
                "paperId": paper_id,
                "title": row["title"],
                "primaryClass": row["primary_contribution_class"],
                "secondaryClasses": row.get("secondary_contribution_classes") or [],
                "whyRankedHere": row["why_ranked_here"],
                "bestCase": row["best_case_for_impact"],
                "mainRisk": row["main_risk"],
                "scores": {
                    "priority": row["overall_priority_score"],
                    "technical": row["technical_soundness_score"],
                    "mlImpact": row["ml_field_impact_score"],
                    "scienceImpact": row["broad_scientific_impact_score"],
                    "evidence": row["evidence_confidence_score"],
                },
                "human": {
                    "tier": human_tier(extra),
                    "reviewerMean": review.get("overall_mean"),
                    "reviewCount": review.get("review_count", 0),
                    "actualAward": bool(extra.get("is_award_paper")),
                    "track": track_name(extra),
                },
                "links": {
                    **primary_link(paper_id, arxiv_match, source, arxiv_overrides),
                    "icml": extra.get("icml_virtual_url"),
                    "openreview": extra.get("paper_url"),
                },
            }
        )

    arxiv_count = sum(paper["links"]["primaryKind"] == "arxiv" for paper in papers)
    if arxiv_count != 59:
        raise ValueError(f"Expected 59 verified arXiv links, found {arxiv_count}")

    return {
        "generatedAt": "2026-07-30",
        "title": "AI as Area Chair: What Science Would AI Send to ICML?",
        "awardRules": {
            "outstandingRanks": [1, 2],
            "honorableMentionRanks": [3, 4, 5, 6, 7],
            "oralRanks": [1, 60],
            "note": (
                "The mock program names ranks 1-2 Outstanding Papers, ranks 3-7 "
                "Honorable Mentions, and the complete 60-paper all-pairs playoff "
                "as AI Orals."
            ),
        },
        "experiment": {
            "scoredPapers": summary["populations"]["all_scored"],
            "mainTrackPapers": summary["populations"]["main_track"]["papers"],
            "finalists": 250,
            "swissPool": 172,
            "playoff": 60,
            "pairwiseComparisons": tournament["tournament_summary"]["pair_count"],
            "finalJudge": tournament["judge"],
            "humanOutcomesJoinedPostHoc": True,
            "arxivLinkedPapers": arxiv_count,
            "officialFallbackLinks": len(papers) - arxiv_count,
        },
        "method": build_method_diagram(tournament),
        "humanComparison": build_human_comparison(
            manifest,
            finalists,
            tournament_seed,
            ranked,
            summary,
        ),
        "results": summary,
        "papers": papers,
    }


def main() -> None:
    payload = json.dumps(build(), ensure_ascii=False, indent=2)
    OUTPUT.write_text(f"window.ICML_AI_AC_DATA = {payload};\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
