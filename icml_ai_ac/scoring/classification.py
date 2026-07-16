from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.providers import ChatCompletionClient
from icml_ai_ac.scoring.runner import estimate_tokens, read_text_path, resolve_record_text_path
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, parse_json_response
from icml_ai_ac.storage import read_paper_records, write_json, write_jsonl


CLASSIFY_PROMPT_VERSION = "contribution_classification_v1"


@dataclass(slots=True)
class ContributionClassificationConfig:
    provider: str
    model: str
    prompt_version: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


def run_contribution_classification(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: ContributionClassificationConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_classification_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=config.paper_ids,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_classification_messages(candidates, config=config)
    prompt_path = run_dir / "prompt.json"
    write_json(
        prompt_path,
        {
            "prompt_version": config.prompt_version,
            "provider": config.provider,
            "model": config.model,
            "candidate_count": len(candidates),
            "candidate_ids": [candidate["paper_id"] for candidate in candidates],
            "text_source": config.text_source,
            "messages": messages,
            "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
        },
    )
    base_result = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "prompt_path": str(prompt_path),
    }
    if config.dry_run:
        rows = [
            {
                "status": "dry_run",
                "paper_id": candidate["paper_id"],
                "title": candidate["title"],
                "prompt_path": str(prompt_path),
            }
            for candidate in candidates
        ]
        write_jsonl(out, rows)
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(run_dir / "run.json", result)
        return result
    if client is None:
        raise ValueError("client is required unless dry_run=True")

    try:
        chat = client.complete(
            messages=messages,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        raw_response_path = run_dir / "response.json"
        write_json(raw_response_path, chat.response)
        parsed = parse_json_response(chat.content)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        rows = classification_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
            usage=chat.usage,
        )
        for row in rows:
            row["served_model"] = chat.served_model
        write_jsonl(out, rows)
        result = {
            **base_result,
            "status": "ok",
            "rows": len(rows),
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {**base_result, "status": "failed", "error": repr(exc), "error_path": str(error_path)}
    write_json(run_dir / "run.json", result)
    return result


def select_classification_candidates(
    *,
    manifest: Path,
    text_source: str,
    paper_ids: set[str],
    limit: int | None,
    per_paper_char_budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_paper_records(manifest):
        if paper_ids and record.paper_id not in paper_ids:
            continue
        if record.parse_status != "ok":
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        text = read_text_path(text_path)
        if per_paper_char_budget > 0:
            text = text[:per_paper_char_budget]
        rows.append(
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "text_path": text_path,
                "resolved_text_source": resolved_text_source,
                "text": text,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_classification_messages(candidates: list[dict[str, Any]], *, config: ContributionClassificationConfig) -> list[dict[str, str]]:
    system = """Classify machine learning papers by primary contribution route.

Use only the supplied compact paper text. Do not use author identity, venue prestige, outside memory, retrieval, or web search. Return only valid JSON."""
    user = f"""Classify each paper into one primary contribution class and optional secondary classes.

Allowed classes:
{", ".join(CONTRIBUTION_CLASSES)}

Rules:
- Use the title/abstract/introduction/conclusion-level evidence only.
- Do not judge quality here. This is routing for later ranking batches.
- Return exactly one item per paper_id.

Required JSON:
{{
  "evaluation_mode": "contribution_classification",
  "prompt_version": "{config.prompt_version}",
  "classifications": [
    {{
      "paper_id": "paper id",
      "title": "title",
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "confidence": 1,
      "rationale": "short reason"
    }}
  ]
}}

Papers:
{format_classification_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_classification_candidates(candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"""<PAPER paper_id="{candidate['paper_id']}">
Title: {candidate['title'] or ""}
Text source: {candidate['resolved_text_source']}
<<<PAPER_TEXT
{candidate['text']}
PAPER_TEXT
</PAPER>"""
        )
    return "\n\n".join(blocks)


def classification_response_to_rows(
    parsed: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    config: ContributionClassificationConfig,
    prompt_path: Path,
    raw_response_path: Path,
    parsed_response_path: Path,
    usage: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = {candidate["paper_id"]: candidate for candidate in candidates}
    classifications = parsed.get("classifications")
    if not isinstance(classifications, list):
        raise ValueError("classification response missing classifications array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in classifications:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        if paper_id not in expected:
            continue
        seen.add(paper_id)
        primary = str(item.get("primary_contribution_class") or "other")
        if primary not in CONTRIBUTION_CLASSES:
            primary = "other"
        rows.append(
            {
                "status": "ok",
                "paper_id": paper_id,
                "title": expected[paper_id]["title"],
                "provider": config.provider,
                "model": config.model,
                "prompt_version": config.prompt_version,
                "text_source": config.text_source,
                "resolved_text_source": expected[paper_id]["resolved_text_source"],
                "text_path": expected[paper_id]["text_path"],
                "primary_contribution_class": primary,
                "secondary_contribution_classes": item.get("secondary_contribution_classes") or [],
                "confidence": item.get("confidence"),
                "rationale": item.get("rationale"),
                "prompt_path": str(prompt_path),
                "raw_response_path": str(raw_response_path),
                "parsed_response_path": str(parsed_response_path),
                "usage": usage,
            }
        )
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"classification response missing paper_ids: {missing}")
    return rows
