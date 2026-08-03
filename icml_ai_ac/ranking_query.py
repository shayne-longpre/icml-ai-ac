from __future__ import annotations

import csv
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from icml_ai_ac.storage import ensure_parent, read_json, read_jsonl


QUERY_PRESETS = ("data", "pretraining", "data-pretraining")
QUERY_SCOPES = ("all", "finalists", "tournament", "playoff")

PRETRAINING_PATTERN = re.compile(r"\bpre[\s-]?train(?:ed|ing)?\b", re.IGNORECASE)
DATA_PRACTICE_PATTERN = re.compile(
    r"\b(?:training data|data (?:curation|selection|filtering|mixture|mixing|quality|"
    r"attribution|deduplication|decontamination|provenance|scaling)|"
    r"(?:corpus|corpora) (?:curation|selection|filtering|mixture|construction))\b",
    re.IGNORECASE,
)

OUTPUT_FIELDS = (
    "final_rank",
    "final_stage",
    "cheap_rank",
    "paper_id",
    "title",
    "topic_cluster",
    "primary_contribution_class",
    "ranking_source",
    "match_reasons",
)


@dataclass(frozen=True, slots=True)
class RankingQuery:
    preset: str | None = None
    query: str | None = None
    topic: str | None = None
    contribution_class: str | None = None
    scope: str = "all"
    limit: int | None = None
    include_abstract: bool = False

    def __post_init__(self) -> None:
        if self.preset is not None and self.preset not in QUERY_PRESETS:
            raise ValueError(f"preset must be one of: {', '.join(QUERY_PRESETS)}")
        if self.scope not in QUERY_SCOPES:
            raise ValueError(f"scope must be one of: {', '.join(QUERY_SCOPES)}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")


def query_ranked_papers(
    *,
    metadata_path: Path,
    cheap_ranking_path: Path,
    final_ranking_path: Path,
    query: RankingQuery,
) -> list[dict[str, Any]]:
    metadata_by_id = index_unique(read_jsonl(metadata_path), source=metadata_path)
    cheap_rows = list(read_jsonl(cheap_ranking_path))
    cheap_by_id = index_unique(cheap_rows, source=cheap_ranking_path)
    final_by_id = index_unique(load_ranked_rows(final_ranking_path), source=final_ranking_path)

    results: list[dict[str, Any]] = []
    for paper_id, cheap_row in cheap_by_id.items():
        metadata = metadata_by_id.get(paper_id, {})
        final_row = final_by_id.get(paper_id)
        row = joined_ranking_row(
            paper_id=paper_id,
            metadata=metadata,
            cheap_row=cheap_row,
            final_row=final_row,
            include_abstract=query.include_abstract,
        )
        reasons = match_reasons(row, metadata=metadata, query=query)
        if reasons is None or not in_scope(row, query.scope):
            continue
        row["match_reasons"] = reasons
        results.append(row)

    results.sort(key=ranking_sort_key)
    if query.limit is not None:
        results = results[: query.limit]
    return results


def index_unique(rows: Any, *, source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError(f"{source}: row missing paper_id")
        if paper_id in indexed:
            raise ValueError(f"{source}: duplicate paper_id {paper_id}")
        indexed[paper_id] = row
    return indexed


def load_ranked_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    payload = read_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("ranked_papers")
    else:
        rows = None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a ranked_papers list or JSONL rows")
    return rows


def joined_ranking_row(
    *,
    paper_id: str,
    metadata: dict[str, Any],
    cheap_row: dict[str, Any],
    final_row: dict[str, Any] | None,
    include_abstract: bool,
) -> dict[str, Any]:
    final_rank = integer(final_row.get("rank")) if final_row else None
    tournament_stats = final_row.get("tournament_stats") if final_row else None
    final_stage = None
    if isinstance(tournament_stats, dict):
        final_stage = tournament_stats.get("ranking_source_stage")
    if final_rank is not None and final_stage is None:
        final_stage = "frontier_finalist_outside_tournament"

    topic = metadata.get("topic_cluster")
    if not topic and isinstance(metadata.get("extra"), dict):
        topic = metadata["extra"].get("topic")
    cheap_contribution_class = (
        cheap_row.get("primary_contribution_class")
        or cheap_row.get("ensemble_contribution_class")
        or cheap_row.get("routing_contribution_class")
    )
    contribution_class = (final_row or {}).get("primary_contribution_class") or cheap_contribution_class
    row: dict[str, Any] = {
        "paper_id": paper_id,
        "title": metadata.get("title") or cheap_row.get("title") or (final_row or {}).get("title"),
        "topic_cluster": topic,
        "primary_contribution_class": contribution_class,
        "cheap_rank": integer(cheap_row.get("aggregate_rank") or cheap_row.get("rank")),
        "final_rank": final_rank,
        "final_stage": final_stage,
        "ranking_source": (final_row or {}).get("ranking_source"),
    }
    if include_abstract:
        row["abstract"] = metadata.get("abstract")
    return row


def match_reasons(
    row: dict[str, Any],
    *,
    metadata: dict[str, Any],
    query: RankingQuery,
) -> list[str] | None:
    reasons: list[str] = []
    if query.preset:
        reasons.extend(preset_match_reasons(row, metadata=metadata, preset=query.preset))
        if not reasons:
            return None

    searchable = "\n".join(
        str(value or "")
        for value in (
            row.get("title"),
            metadata.get("abstract"),
            row.get("topic_cluster"),
            row.get("primary_contribution_class"),
        )
    )
    if query.query:
        if query.query.casefold() not in searchable.casefold():
            return None
        reasons.append(f"query:{query.query}")
    if query.topic:
        topic = str(row.get("topic_cluster") or "")
        if not fnmatch.fnmatch(topic.casefold(), query.topic.casefold()):
            return None
        reasons.append(f"topic:{topic}")
    if query.contribution_class:
        contribution_class = str(row.get("primary_contribution_class") or "")
        if contribution_class != query.contribution_class:
            return None
        reasons.append(f"contribution:{contribution_class}")
    if not any((query.preset, query.query, query.topic, query.contribution_class)):
        reasons.append("all_papers")
    return list(dict.fromkeys(reasons))


def preset_match_reasons(
    row: dict[str, Any],
    *,
    metadata: dict[str, Any],
    preset: str,
) -> list[str]:
    title_abstract = f"{row.get('title') or ''}\n{metadata.get('abstract') or ''}"
    reasons: list[str] = []
    if preset in {"data", "data-pretraining"}:
        if row.get("topic_cluster") == "General Machine Learning->Data":
            reasons.append("official_topic:data")
        if row.get("primary_contribution_class") == "benchmark_dataset":
            reasons.append("contribution:benchmark_dataset")
        if DATA_PRACTICE_PATTERN.search(title_abstract):
            reasons.append("content:data_practice")
    if preset in {"pretraining", "data-pretraining"} and PRETRAINING_PATTERN.search(title_abstract):
        reasons.append("content:pretraining")
    return reasons


def in_scope(row: dict[str, Any], scope: str) -> bool:
    final_rank = row.get("final_rank")
    if scope == "all":
        return True
    if final_rank is None:
        return False
    if scope == "finalists":
        return True
    if scope == "tournament":
        return row.get("final_stage") in {"playoff_all_pairs", "swiss_only"}
    return row.get("final_stage") == "playoff_all_pairs"


def ranking_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    final_rank = row.get("final_rank")
    cheap_rank = row.get("cheap_rank")
    return (
        final_rank is None,
        final_rank if final_rank is not None else 10**9,
        cheap_rank if cheap_rank is not None else 10**9,
        str(row.get("paper_id") or ""),
    )


def write_query_results(rows: list[dict[str, Any]], *, path: Path | None, output_format: str) -> None:
    if path is None:
        write_tsv(rows, handle=sys.stdout)
        return
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if output_format == "csv":
            writer = csv.DictWriter(handle, fieldnames=output_fields(rows))
            writer.writeheader()
            for row in rows:
                writer.writerow(csv_row(row))
        elif output_format == "jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        else:
            write_tsv(rows, handle=handle)


def write_tsv(rows: list[dict[str, Any]], *, handle: TextIO) -> None:
    fields = output_fields(rows)
    writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(csv_row(row))


def output_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = list(OUTPUT_FIELDS)
    if any("abstract" in row for row in rows):
        fields.append("abstract")
    return fields


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "|".join(value) if isinstance(value, list) else value
        for key, value in row.items()
    }


def integer(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
