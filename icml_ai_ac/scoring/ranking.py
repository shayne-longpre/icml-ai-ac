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
from icml_ai_ac.storage import ensure_parent, read_paper_records, write_json, write_jsonl


PASS2_PROMPT_VERSION = "pass2_strong_batch_rank_v3"
PASS2_STAGE1_PROMPT_VERSION = "pass2_stage1_strong_semifinal_v1"
PASS2_FINAL_PROMPT_VERSION = "pass2_stage2_strong_final_v1"
REFERENCE_PROMPT_VERSION = "reference_gold_rank_v1"
REFERENCE_REPAIR_PROMPT_VERSION = "reference_gold_rank_repair_v1"


@dataclass(slots=True)
class RankingCandidate:
    record: PaperRecord
    pass1_row: dict[str, Any]
    text_path: str
    resolved_text_source: str
    paper_text: str


@dataclass(slots=True)
class ReferenceCandidate:
    record: PaperRecord
    text_path: str
    resolved_text_source: str
    paper_text: str


@dataclass(slots=True)
class Pass2RankingConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    text_source: str
    top_fraction: float
    limit: int | None
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class Pass2TwoStageConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    stage1_prompt_version: str
    stage2_prompt_version: str
    text_source: str
    stage1_limit: int | None
    stage2_limit: int
    stage1_per_paper_char_budget: int
    stage2_per_paper_char_budget: int
    temperature: float
    stage1_max_output_tokens: int
    stage2_max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class ReferenceRankingConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class ReferenceRepairConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    text_source: str
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


def run_pass2_ranking(
    *,
    manifest: Path,
    pass1_path: Path,
    out: Path,
    run_dir: Path,
    config: Pass2RankingConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_candidates(
        manifest=manifest,
        pass1_path=pass1_path,
        text_source=config.text_source,
        top_fraction=config.top_fraction,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_pass2_ranking_messages(candidates, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)

    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "pass1_path": str(pass1_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "candidate_titles": {candidate.record.paper_id: candidate.record.title for candidate in candidates},
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
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
        expected_ids = [candidate.record.paper_id for candidate in candidates]
        canonicalize_ranked_paper_ids(parsed, expected_ids)
        validation_errors = validate_ranked_paper_coverage(parsed, expected_ids)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ranking": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - persist the failed ranking attempt.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def run_pass2_two_stage(
    *,
    manifest: Path,
    pass1_path: Path,
    out: Path,
    run_dir: Path,
    config: Pass2TwoStageConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    stage1_out = run_dir / "stage1_ranking.json"
    stage1_rows = run_dir / "stage1_candidates_for_stage2.jsonl"
    stage2_out = run_dir / "stage2_final_ranking.json"
    stage1_result = run_pass2_ranking(
        manifest=manifest,
        pass1_path=pass1_path,
        out=stage1_out,
        run_dir=run_dir / "stage1",
        config=Pass2RankingConfig(
            provider=config.provider,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=config.stage1_prompt_version,
            text_source=config.text_source,
            top_fraction=1.0,
            limit=config.stage1_limit,
            per_paper_char_budget=config.stage1_per_paper_char_budget,
            temperature=config.temperature,
            max_output_tokens=config.stage1_max_output_tokens,
            seed=config.seed,
            dry_run=config.dry_run,
        ),
        client=client,
    )
    if stage1_result["status"] not in {"ok", "dry_run"}:
        result = {
            "status": "stage1_failed",
            "manifest": str(manifest),
            "pass1_path": str(pass1_path),
            "out": str(out),
            "run_dir": str(run_dir),
            "stage1_out": str(stage1_out),
            "stage1_status": stage1_result["status"],
            "stage1_result": stage1_result,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if config.dry_run:
        result = {
            "status": "dry_run",
            "manifest": str(manifest),
            "pass1_path": str(pass1_path),
            "out": str(out),
            "run_dir": str(run_dir),
            "provider": config.provider,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "stage1_prompt_version": config.stage1_prompt_version,
            "stage2_prompt_version": config.stage2_prompt_version,
            "stage1_limit": config.stage1_limit,
            "stage2_limit": config.stage2_limit,
            "stage1_out": str(stage1_out),
            "stage1_status": stage1_result["status"],
            "stage1_result": stage1_result,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result

    converted_count = write_ranking_score_rows(stage1_result, stage1_rows)
    stage2_result = run_pass2_ranking(
        manifest=manifest,
        pass1_path=stage1_rows,
        out=stage2_out,
        run_dir=run_dir / "stage2",
        config=Pass2RankingConfig(
            provider=config.provider,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=config.stage2_prompt_version,
            text_source=config.text_source,
            top_fraction=1.0,
            limit=config.stage2_limit,
            per_paper_char_budget=config.stage2_per_paper_char_budget,
            temperature=config.temperature,
            max_output_tokens=config.stage2_max_output_tokens,
            seed=config.seed,
            dry_run=config.dry_run,
        ),
        client=client,
    )
    result = {
        "status": "ok" if stage2_result["status"] in {"ok", "dry_run"} else "stage2_failed",
        "manifest": str(manifest),
        "pass1_path": str(pass1_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "stage1_prompt_version": config.stage1_prompt_version,
        "stage2_prompt_version": config.stage2_prompt_version,
        "stage1_limit": config.stage1_limit,
        "stage2_limit": config.stage2_limit,
        "stage1_out": str(stage1_out),
        "stage1_rows": str(stage1_rows),
        "stage1_converted_rows": converted_count,
        "stage1_status": stage1_result["status"],
        "stage2_out": str(stage2_out),
        "stage2_status": stage2_result["status"],
        "stage1_result": stage1_result,
        "stage2_result": stage2_result,
        "ranking": stage2_result.get("ranking"),
        "usage": {
            "stage1": stage1_result.get("usage", {}),
            "stage2": stage2_result.get("usage", {}),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def write_ranking_score_rows(result: dict[str, Any], out: Path) -> int:
    ranking = result.get("ranking")
    if not isinstance(ranking, dict):
        ranking = load_ranking_payload(Path(result["out"]))
    ranked = ranking.get("ranked_papers") if isinstance(ranking, dict) else None
    if not isinstance(ranked, list):
        raise ValueError("ranking result missing ranked_papers")
    rows = [
        ranking_item_to_score_row(
            item,
            provider=str(result.get("provider") or ""),
            model=str(result.get("model") or ""),
            prompt_version=str(result.get("prompt_version") or ""),
            source_ranking_path=str(result.get("out") or ""),
        )
        for item in ranked
        if isinstance(item, dict)
    ]
    return write_jsonl(out, rows)


def ranking_item_to_score_row(
    item: dict[str, Any],
    *,
    provider: str,
    model: str,
    prompt_version: str,
    source_ranking_path: str,
) -> dict[str, Any]:
    rank = int(number(item.get("rank")) or 10**9)
    overall = number(item.get("overall_priority_score"))
    if overall <= 0:
        overall = max(0.0, 10.0 - (rank - 1) * 0.25)
    broad = number(item.get("broad_scientific_impact_score"))
    ml = number(item.get("ml_field_impact_score"))
    soundness = number(item.get("technical_soundness_score"))
    contribution_class = str(item.get("primary_contribution_class") or "other")
    if contribution_class not in CONTRIBUTION_CLASSES:
        contribution_class = "other"
    percentile = max(0.0, min(100.0, 100.0 - (rank - 1) * 2.0))
    return {
        "status": "ok",
        "paper_id": str(item.get("paper_id") or ""),
        "title": item.get("title") or "",
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "rank": rank,
        "source_ranking_path": source_ranking_path,
        "scores": {
            "contribution_profile": {
                "primary_contribution_class": contribution_class,
                "secondary_contribution_classes": item.get("secondary_contribution_classes") or [],
            },
            "scores": {
                "executive_ac_priority": overall,
                "broader_science_impact_forecast": broad,
                "ml_field_impact_forecast": ml,
                "overall_significance": overall,
                "technical_soundness": soundness,
            },
            "calibration": {
                "estimated_percentile_among_accepted_papers": percentile,
                "triage_bucket": bucket_from_rank(rank),
                "why_not_higher": item.get("main_risk") or "",
                "why_not_lower": item.get("why_ranked_here") or "",
            },
            "ranking_signals": {
                "should_advance_to_strong_model": bool(item.get("advance_to_final_review")) or rank <= 25,
                "top_paper_case": item.get("top_paper_case") or item.get("best_case_for_impact") or "",
                "broad_scientific_impact_argument": item.get("why_ranked_here") or "",
                "ml_field_impact_argument": item.get("why_ranked_here") or "",
                "dealbreaker_risks": item.get("main_risk") or "",
            },
        },
    }


def bucket_from_rank(rank: int) -> str:
    if rank <= 5:
        return "top_10_percent"
    if rank <= 15:
        return "top_quartile"
    if rank <= 25:
        return "above_average"
    if rank <= 40:
        return "middle"
    return "lower_priority"


def run_reference_ranking(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: ReferenceRankingConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_reference_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=config.paper_ids,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_reference_ranking_messages(candidates, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "paper_set_name": config.paper_set_name,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)

    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "candidate_titles": {candidate.record.paper_id: candidate.record.title for candidate in candidates},
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
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
        canonicalize_reference_ids(parsed, candidates)
        validation_errors = validate_reference_ranking(parsed, candidates)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ranking": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - persist the failed ranking attempt.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def run_reference_repair(
    *,
    manifest: Path,
    reference_path: Path,
    out: Path,
    run_dir: Path,
    config: ReferenceRepairConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_reference_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=set(),
        limit=None,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    source_payload = load_ranking_payload(reference_path)
    canonicalize_reference_ids(source_payload, candidates)
    source_validation_errors = validate_reference_ranking(source_payload, candidates)
    messages = build_reference_repair_messages(
        source_payload,
        candidates=candidates,
        validation_errors=source_validation_errors,
        config=config,
    )
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "paper_set_name": config.paper_set_name,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "reference_path": str(reference_path),
        "source_validation_errors": source_validation_errors,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "manifest": str(manifest),
        "reference_path": str(reference_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "source_validation_errors": source_validation_errors,
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
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
        canonicalize_reference_ids(parsed, candidates)
        validation_errors = validate_reference_ranking(parsed, candidates)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ranking": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - persist failed repair attempts.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def prepare_ranking_run_dir(run_dir: Path) -> None:
    ensure_parent(run_dir / ".keep")
    run_dir.mkdir(parents=True, exist_ok=True)


def select_candidates(
    *,
    manifest: Path,
    pass1_path: Path,
    text_source: str,
    top_fraction: float,
    limit: int | None,
    per_paper_char_budget: int,
) -> list[RankingCandidate]:
    records_by_id = {record.paper_id: record for record in read_paper_records(manifest)}
    rows = [json.loads(line) for line in pass1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok_rows = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("scores"), dict)]
    ok_rows.sort(key=pass1_sort_key, reverse=True)
    if limit is None:
        limit = max(1, round(len(ok_rows) * top_fraction))
    selected_rows = ok_rows[:limit]
    candidates: list[RankingCandidate] = []
    for row in selected_rows:
        paper_id = str(row.get("paper_id") or "")
        record = records_by_id.get(paper_id)
        if record is None:
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        paper_text = truncate_candidate_text(read_text_path(text_path), per_paper_char_budget)
        candidates.append(
            RankingCandidate(
                record=record,
                pass1_row=row,
                text_path=text_path,
                resolved_text_source=resolved_text_source,
                paper_text=paper_text,
            )
        )
    return candidates


def select_reference_candidates(
    *,
    manifest: Path,
    text_source: str,
    paper_ids: set[str],
    limit: int | None,
    per_paper_char_budget: int,
) -> list[ReferenceCandidate]:
    candidates: list[ReferenceCandidate] = []
    for record in read_paper_records(manifest):
        if paper_ids and record.paper_id not in paper_ids:
            continue
        if record.parse_status != "ok":
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        paper_text = truncate_reference_text(read_text_path(text_path), per_paper_char_budget)
        candidates.append(ReferenceCandidate(record, text_path, resolved_text_source, paper_text))
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def pass1_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    return (
        number(core.get("executive_ac_priority")),
        number(calibration.get("estimated_percentile_among_accepted_papers")) / 10.0,
        number(core.get("broader_science_impact_forecast")),
        number(core.get("ml_field_impact_forecast")),
        number(core.get("overall_significance")),
    )


def number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def truncate_candidate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for pass-2 ranking at {limit} characters]"


def truncate_reference_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for reference ranking at {limit} characters]"


def build_pass2_ranking_messages(
    candidates: list[RankingCandidate],
    *,
    config: Pass2RankingConfig,
) -> list[dict[str, str]]:
    system = """You are the strong-model second-stage judge for a machine learning conference paper study.

Rank candidate papers by likely broad scientific and machine-learning impact. Use only the supplied paper text and first-pass outputs. Ignore author identity, institution, venue prestige, and outside knowledge.

This run is text-only and retrieval-free: you do not have PDF page images, figure images, table images, file search, web search, vector stores, or external tools. Treat missing visual/table detail as an uncertainty, not as a factual flaw.

Your job is comparative ranking, not independent accept/reject reviewing. A technically clean but incremental paper should rank below a paper with a plausible path to broad scientific influence, even if both are acceptable ICML papers.

Return only valid JSON."""
    user = f"""Rank the candidate papers selected by the prior pipeline stage.

Primary objective:
- Identify which papers best deserve expensive human/strong-model attention for broad scientific impact, sweeping ML-field importance, and long-run contribution.
- Return exactly {len(candidates)} `ranked_papers` entries: one for every candidate paper_id, with unique ranks 1..{len(candidates)}.
- Candidate paper_ids that must appear exactly once: {", ".join(candidate.record.paper_id for candidate in candidates)}

Secondary objective:
- Audit the prior-stage shortlist. Say whether it surfaced genuinely strong broad-impact candidates or mostly reviewer-polished incremental work.
- Note any paper that the prior stage may have overpromoted or underpromoted from the evidence provided here.

Ranking rubric:
- broad_scientific_impact: could this enable scientific discovery, measurement, modeling, safety, or methodology beyond a narrow subcommunity?
- ml_field_impact: could this reshape common ML practice, theory, evaluation, systems, or benchmarks?
- soundness_gate: if core evidence is weak, cap rank even when the idea is attractive.
- novelty_gate: incremental combinations should not outrank field-shaping ideas unless they are exceptionally useful.
- evidence_gate: distinguish visible evidence from merely plausible impact.
- contribution_class: classify the main impact route. Benchmarks, datasets, infrastructure, safety/evals, scientific tools, theory, and algorithms can all be high-impact, but should also be compared within class.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "pass2_strong_batch_ranking",
  "prompt_version": "{config.prompt_version}",
  "ranked_papers": [
    {{
      "rank": 1,
      "paper_id": "paper id",
      "title": "title",
      "broad_scientific_impact_score": 1,
      "ml_field_impact_score": 1,
      "technical_soundness_score": 1,
      "overall_priority_score": 1,
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "advance_to_final_review": true,
      "why_ranked_here": "evidence-grounded comparative reason",
      "top_paper_case": "best case for broad importance",
      "main_risk": "main reason this might not matter"
    }}
  ],
  "first_pass_quality_assessment": {{
    "did_prior_stage_surface_broad_impact_candidates": "yes/no/mixed plus explanation",
    "prior_stage_overpromoted": ["paper_id with reason"],
    "prior_stage_underpromoted": ["paper_id with reason"],
    "recommended_first_pass_changes": ["prompt or representation changes"]
  }},
  "category_rankings": {{
    "core_ml_algorithm": ["paper_id", "paper_id"],
    "benchmark_dataset": ["paper_id"],
    "infrastructure_systems": ["paper_id"],
    "scientific_modeling_tool": ["paper_id"],
    "safety_governance_eval": ["paper_id"]
  }},
  "modality_limitations": {{
    "text_only_risks": ["what may be missed without PDF page images, figures, or table images"],
    "would_pdf_visual_review_change_any_rankings": "yes/no/maybe plus explanation"
  }},
  "overall_assessment": "short comparative assessment"
}}

Candidates:
{format_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reference_ranking_messages(
    candidates: list[ReferenceCandidate],
    *,
    config: ReferenceRankingConfig,
) -> list[dict[str, str]]:
    n = len(candidates)
    system = """You are the strongest available reference judge for a machine learning conference paper study.

Create an independent reference ranking for evaluation of a cheaper first-pass -> strong-model pipeline. Use only the supplied extracted paper text. Do not use web search, retrieval, citation memory, author identity, institution, venue prestige, or outside knowledge.

This run is text-only and retrieval-free: you do not have PDF page images, figure images, table images, file search, vector stores, or external tools. Treat missing visual/table detail as an uncertainty, not as a factual flaw.

Rank by likely broad scientific and machine-learning impact, with technical soundness as a gate. Your job is comparative prioritization among accepted papers, not accept/reject reviewing.

Return only valid JSON."""
    user = f"""Create a reference gold ranking for these {n} ICML accepted papers.

This artifact will be used to evaluate whether cheaper batch triage and strong-model reranking preserve the papers a stronger judge would prioritize. It is not a claim of objective truth; it is a reproducible strong-judge reference.

Primary objective:
- Rank every candidate from 1 to {n}; no ties, no missing papers.
- Identify papers with the strongest expected broad scientific impact and ML-field impact.
- Give higher ranks to work that could reshape methods, evaluation, theory, scientific practice, safety/evals, infrastructure, or durable community resources.

Ranking rules:
- Use only the text provided here.
- Do not infer merit from author identity, institution, citation memory, web knowledge, or ICML acceptance itself.
- Apply a soundness gate: weak or under-evidenced technical claims should cap the rank even when the idea is attractive.
- Apply an impact gate: polished but incremental papers should not outrank papers with a plausible path to field-shaping influence.
- Score spread matters: use the full 1-10 scale where justified, and make ranks meaningfully comparative.
- Benchmarks, datasets, infrastructure, safety/eval work, scientific tools, theory, and algorithms can all be high-impact, but also rank each paper within its primary contribution class.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "reference_gold_ranking",
  "prompt_version": "{config.prompt_version}",
  "judge": "{config.provider}:{config.model}",
  "paper_set": "{config.paper_set_name}",
  "modality": "text_only_extracted_main_paper",
  "retrieval": "off",
  "ranked_papers": [
    {{
      "rank": 1,
      "paper_id": "paper id",
      "title": "title",
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "category_rank": 1,
      "overall_priority_score": 10,
      "broad_scientific_impact_score": 10,
      "ml_field_impact_score": 10,
      "technical_soundness_score": 9,
      "novelty_score": 9,
      "evidence_confidence_score": 8,
      "advance_to_final_review": true,
      "two_year_adoption_path": "who uses it and what changes",
      "why_ranked_here": "evidence-grounded comparative reason",
      "best_case_for_impact": "strongest plausible impact route",
      "main_risk": "main reason this may not matter or may not hold up"
    }}
  ],
  "category_rankings": {{
    "core_ml_algorithm": ["paper_id"],
    "theory": ["paper_id"],
    "benchmark_dataset": ["paper_id"],
    "infrastructure_systems": ["paper_id"],
    "scientific_modeling_tool": ["paper_id"],
    "safety_governance_eval": ["paper_id"],
    "application_method": ["paper_id"],
    "analysis_position": ["paper_id"],
    "other": ["paper_id"]
  }},
  "calibration_notes": {{
    "top_10_threshold": "what distinguished the top 10",
    "top_20_threshold": "what distinguished the top 20",
    "common_overranking_risks": ["risks that cheaper models may overvalue"],
    "common_underranking_risks": ["risks that cheaper models may miss"]
  }},
  "modality_limitations": {{
    "text_only_risks": ["what may be missed without PDF page images, figures, or table images"],
    "would_pdf_visual_review_change_any_rankings": "yes/no/maybe plus explanation"
  }},
  "overall_assessment": "short comparative assessment of this 50-paper set"
}}

Candidate paper_ids that must appear exactly once:
{", ".join(candidate.record.paper_id for candidate in candidates)}

Papers:
{format_reference_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reference_repair_messages(
    source_payload: dict[str, Any],
    *,
    candidates: list[ReferenceCandidate],
    validation_errors: list[str],
    config: ReferenceRepairConfig,
) -> list[dict[str, str]]:
    expected_ids = [candidate.record.paper_id for candidate in candidates]
    rows = source_payload.get("ranked_papers")
    rows = rows if isinstance(rows, list) else []
    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    expected_id_set = set(expected_ids)
    missing_ids = sorted(expected_id_set - row_id_set)
    duplicate_ids = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    source_excerpt = {
        "evaluation_mode": source_payload.get("evaluation_mode"),
        "prompt_version": source_payload.get("prompt_version"),
        "judge": source_payload.get("judge"),
        "paper_set": source_payload.get("paper_set"),
        "modality": source_payload.get("modality"),
        "retrieval": source_payload.get("retrieval"),
        "ranked_papers": rows,
        "category_rankings": source_payload.get("category_rankings"),
        "calibration_notes": source_payload.get("calibration_notes"),
        "modality_limitations": source_payload.get("modality_limitations"),
        "overall_assessment": source_payload.get("overall_assessment"),
    }
    missing_texts = [
        format_reference_candidate(candidate)
        for candidate in candidates
        if candidate.record.paper_id in missing_ids
    ]
    duplicate_texts = [
        format_reference_candidate(candidate)
        for candidate in candidates
        if candidate.record.paper_id in duplicate_ids
    ]
    system = """You are repairing a structurally invalid reference ranking for a machine learning conference paper study.

Use only the supplied existing ranking and supplied paper text. Do not use web search, retrieval, citation memory, author identity, institution, venue prestige, or outside knowledge.

Your task is to fix exact paper coverage while preserving the ranking's comparative intent where possible. Return only valid JSON."""
    user = f"""Repair this reference ranking for {config.paper_set_name}.

The previous model response failed structural validation:
{json.dumps(validation_errors, ensure_ascii=False, indent=2)}

Expected candidate paper_ids, each exactly once:
{", ".join(expected_ids)}

Repair rules:
- Return the full reference ranking JSON, not a patch.
- `ranked_papers` must contain exactly {len(candidates)} entries.
- Every expected paper_id must appear exactly once; no extras, no duplicates.
- Ranks must be exactly 1..{len(candidates)} with no ties.
- Insert each missing paper at the rank justified by its text and the current ranking ladder.
- Remove duplicate rows. If a duplicated paper appears twice, keep the entry whose rank/reason is better supported by the current ranking context unless the lower entry clearly belongs to a missing paper.
- Preserve the existing order of unrelated papers unless the missing paper's placement requires renumbering.
- Keep the same schema as the source ranking and update category rankings to match the repaired `ranked_papers`.
- Mark `"prompt_version": "{config.prompt_version}"`, `"judge": "{config.provider}:{config.model}"`, `"paper_set": "{config.paper_set_name}"`, `"modality": "text_only_extracted_main_paper"`, and `"retrieval": "off"`.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Current invalid ranking:
<<<CURRENT_RANKING_JSON
{json.dumps(source_excerpt, ensure_ascii=False, indent=2)}
CURRENT_RANKING_JSON

Missing paper text:
{chr(10).join(missing_texts) if missing_texts else "[none]"}

Duplicated paper text:
{chr(10).join(duplicate_texts) if duplicate_texts else "[none]"}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_candidates(candidates: list[RankingCandidate]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(format_candidate(candidate))
    return "\n\n".join(blocks)


def format_candidate(candidate: RankingCandidate) -> str:
    first_pass = compact_first_pass(candidate.pass1_row)
    return f"""<CANDIDATE paper_id="{candidate.record.paper_id}">
Title: {candidate.record.title or ""}
Decision label: {candidate.record.decision_label or ""}
Text source: {candidate.resolved_text_source}
Text path: {candidate.text_path}

First-pass cheap-model output summary:
{json.dumps(first_pass, ensure_ascii=False, indent=2, sort_keys=True)}

Paper text:
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
</CANDIDATE>"""


def format_reference_candidates(candidates: list[ReferenceCandidate]) -> str:
    return "\n\n".join(format_reference_candidate(candidate) for candidate in candidates)


def format_reference_candidate(candidate: ReferenceCandidate) -> str:
    return f"""<PAPER paper_id="{candidate.record.paper_id}">
Title: {candidate.record.title or ""}
Decision label: {candidate.record.decision_label or ""}
Text source: {candidate.resolved_text_source}
Text path: {candidate.text_path}

Paper text:
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
</PAPER>"""


def compact_first_pass(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    keep = {
        "summary": scores.get("summary"),
        "scores": scores.get("scores"),
        "impact_axes": scores.get("impact_axes"),
        "calibration": scores.get("calibration"),
        "ranking_signals": scores.get("ranking_signals"),
        "evidence": scores.get("evidence"),
        "evidence_audit": scores.get("evidence_audit"),
        "uncertainty": scores.get("uncertainty"),
    }
    return {key: value for key, value in keep.items() if value is not None}


def canonicalize_ranked_paper_ids(payload: dict[str, Any], expected_ids: list[str]) -> None:
    canonical_by_lower = {paper_id.lower(): paper_id for paper_id in expected_ids}
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str):
            canonical = canonical_by_lower.get(paper_id.lower())
            if canonical is not None:
                row["paper_id"] = canonical


def validate_ranked_paper_coverage(payload: dict[str, Any], expected_ids: list[str]) -> list[str]:
    errors: list[str] = []
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return ["ranked_papers must be a list"]
    expected_id_set = set(expected_ids)
    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    duplicates = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    missing = sorted(expected_id_set - row_id_set)
    extra = sorted(row_id_set - expected_id_set)
    if len(rows) != len(expected_ids):
        errors.append(f"ranked_papers length {len(rows)} does not match candidate count {len(expected_ids)}")
    if duplicates:
        errors.append("duplicate paper_ids: " + ", ".join(duplicates))
    if missing:
        errors.append("missing paper_ids: " + ", ".join(missing))
    if extra:
        errors.append("extra paper_ids: " + ", ".join(extra))
    ranks: list[int] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"ranked_papers[{idx}] must be an object")
            continue
        rank = row.get("rank")
        if not isinstance(rank, int):
            errors.append(f"ranked_papers[{idx}].rank must be an integer")
        else:
            ranks.append(rank)
    expected_ranks = list(range(1, len(expected_ids) + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(f"ranks must be exactly 1..{len(expected_ids)} with no ties")
    return errors


def canonicalize_reference_ids(payload: dict[str, Any], candidates: list[ReferenceCandidate]) -> None:
    canonical_by_lower = {candidate.record.paper_id.lower(): candidate.record.paper_id for candidate in candidates}
    rows = payload.get("ranked_papers")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            paper_id = row.get("paper_id")
            if isinstance(paper_id, str):
                canonical = canonical_by_lower.get(paper_id.lower())
                if canonical is not None:
                    row["paper_id"] = canonical
    category_rankings = payload.get("category_rankings")
    if isinstance(category_rankings, dict):
        for key, values in category_rankings.items():
            if not isinstance(values, list):
                continue
            category_rankings[key] = [
                canonical_by_lower.get(value.lower(), value) if isinstance(value, str) else value
                for value in values
            ]


def validate_reference_ranking(payload: dict[str, Any], candidates: list[ReferenceCandidate]) -> list[str]:
    errors: list[str] = []
    expected_ids = [candidate.record.paper_id for candidate in candidates]
    expected_id_set = set(expected_ids)
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return ["ranked_papers must be a list"]
    if len(rows) != len(candidates):
        errors.append(f"ranked_papers length {len(rows)} does not match candidate count {len(candidates)}")

    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    duplicates = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    missing = sorted(expected_id_set - row_id_set)
    extra = sorted(row_id_set - expected_id_set)
    if duplicates:
        errors.append("duplicate paper_ids: " + ", ".join(duplicates))
    if missing:
        errors.append("missing paper_ids: " + ", ".join(missing))
    if extra:
        errors.append("extra paper_ids: " + ", ".join(extra))

    ranks: list[int] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"ranked_papers[{idx}] must be an object")
            continue
        rank = row.get("rank")
        if not isinstance(rank, int):
            errors.append(f"ranked_papers[{idx}].rank must be an integer")
        else:
            ranks.append(rank)
        contribution_class = row.get("primary_contribution_class")
        if contribution_class is not None and contribution_class not in CONTRIBUTION_CLASSES:
            errors.append(
                f"ranked_papers[{idx}].primary_contribution_class must be one of: "
                + ", ".join(CONTRIBUTION_CLASSES)
            )
    expected_ranks = list(range(1, len(candidates) + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(f"ranks must be exactly 1..{len(candidates)} with no ties")
    return errors


def load_ranking_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("ranking"), dict):
        return payload["ranking"]
    if not isinstance(payload, dict):
        raise ValueError(f"ranking payload must be an object: {path}")
    return payload
