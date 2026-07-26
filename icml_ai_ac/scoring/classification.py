from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.providers import ChatCompletionClient, is_batch_blocking_provider_error
from icml_ai_ac.scoring.runner import estimate_tokens, read_text_path, resolve_record_text_path
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, parse_json_response
from icml_ai_ac.scoring.usage import sum_usage
from icml_ai_ac.storage import read_json, read_paper_records, write_json, write_jsonl


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
    batch_size: int
    request_delay_seconds: float
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
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = [
        candidates[index : index + config.batch_size]
        for index in range(0, len(candidates), config.batch_size)
    ]
    base_result = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "batch_size": config.batch_size,
        "batch_count": len(batches),
    }
    if not config.dry_run and client is None:
        raise ValueError("client is required unless dry_run=True")

    rows: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []
    failure_count = 0
    resumed_batch_count = 0
    blocked_reason: str | None = None
    for batch_index, batch_candidates in enumerate(batches):
        batch_dir = run_dir / "batches" / f"batch_{batch_index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        messages = build_classification_messages(batch_candidates, config=config)
        prompt_path = batch_dir / "prompt.json"
        prompt_payload = {
            "prompt_version": config.prompt_version,
            "provider": config.provider,
            "model": config.model,
            "batch_index": batch_index,
            "candidate_count": len(batch_candidates),
            "candidate_ids": [candidate["paper_id"] for candidate in batch_candidates],
            "text_source": config.text_source,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "seed": config.seed,
            "messages": messages,
            "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
        }
        fingerprint = prompt_fingerprint(prompt_payload)
        prompt_payload["fingerprint"] = fingerprint
        write_json(prompt_path, prompt_payload)

        if config.dry_run:
            batch_rows = [
                {
                    "status": "dry_run",
                    "paper_id": candidate["paper_id"],
                    "title": candidate["title"],
                    "batch_index": batch_index,
                    "batch_size": len(batch_candidates),
                    "prompt_path": str(prompt_path),
                }
                for candidate in batch_candidates
            ]
            rows.extend(batch_rows)
            batch_results.append(
                {
                    "status": "dry_run",
                    "batch_index": batch_index,
                    "candidate_count": len(batch_candidates),
                    "fingerprint": fingerprint,
                    "prompt_path": str(prompt_path),
                }
            )
            continue

        cached = load_cached_classification_batch(
            batch_dir=batch_dir,
            fingerprint=fingerprint,
            candidates=batch_candidates,
            config=config,
            prompt_path=prompt_path,
        )
        if cached is not None:
            batch_rows, batch_result = cached
            rows.extend(batch_rows)
            batch_results.append(batch_result)
            resumed_batch_count += 1
            continue

        chat = None
        try:
            if client is None:
                raise RuntimeError("live classification requires a provider client")
            chat = client.complete(
                messages=messages,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                seed=config.seed,
                response_format={"type": "json_object"},
            )
            raw_response_path = batch_dir / "response.json"
            write_json(raw_response_path, chat.response)
            parsed = parse_json_response(chat.content)
            parsed_response_path = batch_dir / "parsed.json"
            write_json(parsed_response_path, parsed)
            batch_rows = classification_response_to_rows(
                parsed,
                candidates=batch_candidates,
                config=config,
                prompt_path=prompt_path,
                raw_response_path=raw_response_path,
                parsed_response_path=parsed_response_path,
                usage=chat.usage,
            )
            annotate_classification_rows(
                batch_rows,
                batch_index=batch_index,
                batch_size=len(batch_candidates),
                served_model=chat.served_model,
            )
            batch_result = {
                "status": "ok",
                "batch_index": batch_index,
                "candidate_count": len(batch_candidates),
                "fingerprint": fingerprint,
                "prompt_path": str(prompt_path),
                "raw_response_path": str(raw_response_path),
                "parsed_response_path": str(parsed_response_path),
                "served_model": chat.served_model,
                "usage": chat.usage,
                "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
                "resumed": False,
            }
            write_json(batch_dir / "batch.json", batch_result)
            rows.extend(batch_rows)
            batch_results.append(batch_result)
        except Exception as exc:  # noqa: BLE001 - persist and retry on a later invocation.
            failure_count += 1
            response = getattr(exc, "response", None)
            raw_response_path = batch_dir / "response.json"
            if isinstance(response, dict) and not raw_response_path.exists():
                write_json(raw_response_path, response)
            usage = chat.usage if chat is not None else getattr(exc, "usage", {})
            served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
            error_path = batch_dir / "error.json"
            write_json(error_path, {"error": repr(exc)})
            batch_result = {
                "status": "failed",
                "batch_index": batch_index,
                "candidate_count": len(batch_candidates),
                "fingerprint": fingerprint,
                "prompt_path": str(prompt_path),
                "error_path": str(error_path),
                "error": repr(exc),
            }
            if raw_response_path.exists():
                batch_result["raw_response_path"] = str(raw_response_path)
            if isinstance(usage, dict) and usage:
                batch_result["usage"] = usage
            if served_model:
                batch_result["served_model"] = served_model
            write_json(batch_dir / "batch.json", batch_result)
            batch_results.append(batch_result)
            if is_batch_blocking_provider_error(exc):
                blocked_reason = repr(exc)
        if config.request_delay_seconds:
            time.sleep(config.request_delay_seconds)
        if blocked_reason:
            break

    write_jsonl(out, rows)
    attempted_count = sum(int(result.get("candidate_count") or 0) for result in batch_results)
    result = {
        **base_result,
        "status": (
            "dry_run"
            if config.dry_run
            else ("ok" if failure_count == 0 and attempted_count == len(candidates) else "partial_failed")
        ),
        "rows": len(rows),
        "attempted_candidate_count": attempted_count,
        "attempted_batch_count": len(batch_results),
        "skipped_batch_count": len(batches) - len(batch_results),
        "failure_count": failure_count,
        "resumed_batch_count": resumed_batch_count,
        "blocked_reason": blocked_reason,
        "usage": sum_usage(result.get("usage") for result in batch_results),
        "batch_results": batch_results,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(run_dir / "run.json", result)
    return result


def prompt_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_classification_batch(
    *,
    batch_dir: Path,
    fingerprint: str,
    candidates: list[dict[str, Any]],
    config: ContributionClassificationConfig,
    prompt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    state_path = batch_dir / "batch.json"
    raw_response_path = batch_dir / "response.json"
    parsed_response_path = batch_dir / "parsed.json"
    if not state_path.exists() or not raw_response_path.exists() or not parsed_response_path.exists():
        return None
    try:
        state = read_json(state_path)
        if (
            not isinstance(state, dict)
            or state.get("status") != "ok"
            or state.get("fingerprint") != fingerprint
        ):
            return None
        parsed = read_json(parsed_response_path)
        if not isinstance(parsed, dict):
            return None
        rows = classification_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
            usage=state.get("usage") if isinstance(state.get("usage"), dict) else {},
        )
        annotate_classification_rows(
            rows,
            batch_index=int(state["batch_index"]),
            batch_size=len(candidates),
            served_model=state.get("served_model"),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    resumed_state = {**state, "resumed": True}
    return rows, resumed_state


def annotate_classification_rows(
    rows: list[dict[str, Any]],
    *,
    batch_index: int,
    batch_size: int,
    served_model: Any,
) -> None:
    for row in rows:
        row["batch_index"] = batch_index
        row["batch_size"] = batch_size
        row["served_model"] = served_model


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
    canonical_by_lower = {paper_id.lower(): paper_id for paper_id in expected}
    classifications = parsed.get("classifications")
    if not isinstance(classifications, list):
        raise ValueError("classification response missing classifications array")
    rows: list[dict[str, Any]] = []
    output_ids: list[str] = []
    for item in classifications:
        if not isinstance(item, dict):
            raise ValueError("classification response contains a non-object item")
        raw_paper_id = str(item.get("paper_id") or "")
        paper_id = canonical_by_lower.get(raw_paper_id.lower(), raw_paper_id)
        output_ids.append(paper_id)
        if paper_id not in expected:
            continue
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
    duplicate_ids = sorted({paper_id for paper_id in output_ids if output_ids.count(paper_id) > 1})
    missing_ids = sorted(set(expected) - set(output_ids))
    extra_ids = sorted(set(output_ids) - set(expected))
    if len(classifications) != len(expected) or duplicate_ids or missing_ids or extra_ids:
        raise ValueError(
            "classification response paper_id coverage error: "
            f"expected_count={len(expected)}, actual_count={len(classifications)}, "
            f"duplicates={duplicate_ids}, missing={missing_ids}, extra={extra_ids}"
        )
    return rows
