from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.storage import append_jsonl, read_jsonl_if_exists, write_json, write_jsonl


ANONYMIZATION_VERSION = "direct_identity_redaction_v13"
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
URL_PATTERN = re.compile(
    r"(?i)(?:"
    r"\bhttps?://[^\s<>()\[\]{}]*|"
    r"\bwww\.[^\s<>()\[\]{}]+|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|ai|io|dev|co)"
    r"(?:/[^\s<>()\[\]{}]*)?"
    r")"
)
ACKNOWLEDGEMENTS_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*\.?|[A-Z])\s+)?"
    r"(?:acknowledg(?:e)?ments?|acknowledgement)[ \t]*$"
)
IDENTITY_LINE_PATTERN = re.compile(
    r"(?i)(?:\bcorrespondence\s+to\b|\bequal\s+contribution\b|"
    r"\bdepartment\s+of\b|\buniversity\b|\binstitute\b|\blaboratory\b|"
    r"\baffiliation\b|\bproceedings\s+of\s+the\b|\bpmlr\s+\d+\b|"
    r"\bcopyright\s+\d{4}\b|\bby\s+the\s+author(?:s)?\b|@)"
)
TEXT_IDENTITY_LINE_PATTERN = re.compile(
    r"(?i)(?:"
    r"^\s*[*†‡]?\s*(?:equal\s+contribution|correspondence\s+to|department\b|"
    r"school\s+of\b|faculty\s+of\b|laboratory\b|institute\b|"
    r"international\s+conference\s+on\s+machine\b)|"
    r"^\s*of\b.{0,120}\buniversity\b|"
    r"\bproceedings\s+of\s+the\b|\bpmlr\s+\d+\b|"
    r"\bcopyright\s+\d{4}\b|\bby\s+the\s+author(?:s)?\b|"
    r"^\s*.{0,90}\buniversity\s*,.{0,80}$"
    r")"
)
AFFILIATION_FRAGMENT_PATTERN = re.compile(
    r"(?i)(?:"
    r"\buniversity\b|\bdepartment\b|\binstitute\b|\blaborator(?:y|ies)\b|"
    r"\bschool\s+of\b|\bfaculty\s+of\b|\bcorrespondence\b|"
    r"\bauthors?\s+are\s+ordered\b|\binternship\s+at\b|"
    r"\b(?:USA|U\.S\.A\.|UK|U\.K\.|Canada|China|France|Germany|"
    r"Switzerland|Belgium|Australia|Japan|Korea|Singapore)\b"
    r")"
)


@dataclass(frozen=True, slots=True)
class AnonymizationConfig:
    pdf_dir: Path
    text_dir: Path
    max_pages: int = 9
    limit: int | None = None
    overwrite: bool = False


def anonymize_manifest(
    *,
    records: list[PaperRecord],
    out: Path,
    report_path: Path,
    config: AnonymizationConfig,
) -> dict[str, Any]:
    if config.max_pages <= 0:
        raise ValueError("max_pages must be positive")
    journal_path = out.with_suffix(out.suffix + ".journal.jsonl")
    journal_rows = {
        str(row.get("paper_id")): row
        for row in read_jsonl_if_exists(journal_path)
        if row.get("paper_id")
    }
    selected = records[: config.limit] if config.limit is not None else records
    completed: list[PaperRecord] = []
    audit_rows: list[dict[str, Any]] = []
    resumed = 0
    for record in selected:
        fingerprint = anonymization_fingerprint(record, max_pages=config.max_pages)
        prior = journal_rows.get(record.paper_id)
        if (
            prior
            and prior.get("fingerprint") == fingerprint
            and prior.get("status") == "ok"
            and not config.overwrite
        ):
            resumed_record = PaperRecord.from_dict(prior["record"])
            if scoring_artifacts_exist(resumed_record):
                completed.append(resumed_record)
                audit_rows.append(prior["audit"])
                resumed += 1
                continue
        anonymized_record, audit = anonymize_record(
            record,
            config=config,
            fingerprint=fingerprint,
        )
        journal_row = {
            "paper_id": record.paper_id,
            "fingerprint": fingerprint,
            "status": audit["status"],
            "audit": audit,
            "record": anonymized_record.to_dict(),
        }
        append_jsonl(journal_path, [journal_row])
        journal_rows[record.paper_id] = journal_row
        audit_rows.append(audit)
        if audit["status"] == "ok":
            completed.append(anonymized_record)

    write_jsonl(out, (record.to_dict() for record in completed))
    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "anonymization_version": ANONYMIZATION_VERSION,
        "input_records": len(selected),
        "output_records": len(completed),
        "excluded_records": len(selected) - len(completed),
        "resumed_records": resumed,
        "max_pages": config.max_pages,
        "pdf_dir": str(config.pdf_dir),
        "text_dir": str(config.text_dir),
        "journal": str(journal_path),
        "status_counts": status_counts,
        "excluded_paper_ids": [
            row["paper_id"] for row in audit_rows if row.get("status") != "ok"
        ],
        "records": audit_rows,
    }
    write_json(report_path, report)
    return report


def anonymize_record(
    record: PaperRecord,
    *,
    config: AnonymizationConfig,
    fingerprint: str | None = None,
) -> tuple[PaperRecord, dict[str, Any]]:
    source_pdf = Path(record.pdf_path) if record.pdf_path else None
    audit: dict[str, Any] = {
        "paper_id": record.paper_id,
        "status": "failed",
        "anonymization_version": ANONYMIZATION_VERSION,
        "fingerprint": fingerprint,
        "authors_supplied": len(record.authors),
        "error": None,
    }
    if source_pdf is None or not source_pdf.exists():
        audit["error"] = "missing source PDF"
        return record, audit

    source_text_paths = representation_paths(record)
    missing_text = sorted(name for name, path in source_text_paths.items() if path is None or not path.exists())
    if missing_text:
        audit["error"] = f"missing text representations: {', '.join(missing_text)}"
        return record, audit

    config.pdf_dir.mkdir(parents=True, exist_ok=True)
    paper_text_dir = config.text_dir / record.paper_id
    paper_text_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = config.pdf_dir / f"{record.paper_id}.first{config.max_pages}.anonymized.pdf"
    out_text_paths = {
        name: paper_text_dir / f"{name}.anonymized.txt"
        for name in ("compact_repr", "full_repr", "scoring_repr")
    }
    try:
        pdf_audit = anonymize_pdf(
            source_pdf=source_pdf,
            out_pdf=out_pdf,
            authors=record.authors,
            max_pages=config.max_pages,
        )
        text_audits: dict[str, Any] = {}
        for name, source_path in source_text_paths.items():
            assert source_path is not None
            source_text = source_path.read_text(encoding="utf-8")
            sanitized, text_audit = anonymize_representation(
                source_text,
                authors=record.authors,
                strip_front_matter=name in {"full_repr", "scoring_repr"},
            )
            out_text_paths[name].write_text(sanitized, encoding="utf-8")
            text_audits[name] = text_audit
        residual_text_authors = sorted(
            {
                author
                for text_audit in text_audits.values()
                for author in text_audit["residual_authors"]
            }
        )
        status = (
            "ok"
            if pdf_audit["status"] == "ok" and not residual_text_authors
            else "needs_review"
        )
        audit.update(
            {
                "status": status,
                "source_pdf": str(source_pdf),
                "source_pdf_sha256": sha256_file(source_pdf),
                "anonymized_pdf": str(out_pdf),
                "anonymized_pdf_sha256": sha256_file(out_pdf),
                "pdf": pdf_audit,
                "text": text_audits,
                "residual_text_authors": residual_text_authors,
            }
        )
        if status == "ok":
            record = build_blinded_record(
                source=record,
                out_pdf=out_pdf,
                out_text_paths=out_text_paths,
                audit=audit,
            )
    except Exception as exc:  # noqa: BLE001 - preserve row-local failures for review.
        audit["error"] = repr(exc)
    return record, audit


def build_blinded_record(
    *,
    source: PaperRecord,
    out_pdf: Path,
    out_text_paths: dict[str, Path],
    audit: dict[str, Any],
) -> PaperRecord:
    scoring_artifacts = {
        "artifact_kind": "anonymized_derivative",
        "anonymization_version": ANONYMIZATION_VERSION,
        "pdf": str(out_pdf),
        **{name: str(path) for name, path in out_text_paths.items()},
    }
    safe_anonymization = {
        key: audit[key]
        for key in (
            "paper_id",
            "status",
            "anonymization_version",
            "fingerprint",
            "source_pdf_sha256",
            "anonymized_pdf_sha256",
        )
    }
    return PaperRecord(
        paper_id=source.paper_id,
        source="blinded_scoring",
        title=source.title,
        pdf_path=str(out_pdf),
        text_full=str(out_text_paths["full_repr"]),
        text_compact=str(out_text_paths["compact_repr"]),
        text_scoring=str(out_text_paths["scoring_repr"]),
        parse_status=source.parse_status,
        extra={
            "anonymization": safe_anonymization,
            "scoring_artifacts": scoring_artifacts,
            "blind_manifest": {
                "human_outcomes_available_to_models": False,
                "identity_metadata_available_to_models": False,
            },
        },
    )


def anonymize_pdf(
    *,
    source_pdf: Path,
    out_pdf: Path,
    authors: list[str],
    max_pages: int,
) -> dict[str, Any]:
    previous_error_display = pymupdf.TOOLS.mupdf_display_errors()
    pymupdf.TOOLS.mupdf_warnings(reset=1)
    pymupdf.TOOLS.mupdf_display_errors(False)
    output: pymupdf.Document | None = None
    first_page_author_band_redacted = False
    try:
        source = pymupdf.open(source_pdf)
        try:
            if len(source) == 0:
                raise ValueError(f"source PDF has no pages: {source_pdf}")
            output = pymupdf.open()
            output.insert_pdf(source, from_page=0, to_page=min(len(source), max_pages) - 1)
        finally:
            source.close()

        author_hits: dict[str, int] = {author: 0 for author in authors}
        redaction_count = 0
        identity_line_count = 0
        mail_link_count = 0
        external_link_count = 0
        url_redaction_count = 0
        for page_index, page in enumerate(output):
            page_rects: list[pymupdf.Rect] = []
            first_page_author_rects: list[pymupdf.Rect] = []
            words = page.get_text("words", sort=True)
            for author in authors:
                rects: list[pymupdf.Rect] = []
                for variant in author_name_variants(author):
                    rects.extend(find_name_rects(words, variant))
                if not rects:
                    rects = list(page.search_for(author))
                rects = dedupe_rects(rects)
                if rects:
                    author_hits[author] += len(rects)
                    page_rects.extend(rects)
                    if page_index == 0:
                        first_page_author_rects.extend(
                            rect for rect in rects if rect.y0 < page.rect.height * 0.42
                        )
            if first_page_author_rects:
                y0 = max(0.0, min(rect.y0 for rect in first_page_author_rects) - 2.0)
                y1 = min(page.rect.height, max(rect.y1 for rect in first_page_author_rects) + 2.0)
                page_rects.append(pymupdf.Rect(0.0, y0, page.rect.width, y1))
                first_page_author_band_redacted = True
            author_band_bottom = (
                max(rect.y1 for rect in first_page_author_rects)
                if first_page_author_rects
                else None
            )
            abstract_hits = list(page.search_for("Abstract")) if page_index == 0 else []
            abstract_top = min((rect.y0 for rect in abstract_hits), default=None)

            for line_text, line_rect in iter_page_lines(page):
                has_author_identity = any(
                    contains_author_identity(line_text, author)
                    for author in authors
                )
                if has_author_identity or EMAIL_PATTERN.search(line_text):
                    page_rects.append(
                        expand_rect(line_rect, page=page, x_padding=2.0, y_padding=2.0)
                    )
                    identity_line_count += 1
                elif (
                    page_index == 0
                    and (
                        line_rect.y0 > page.rect.height * 0.72
                        or (
                            author_band_bottom is not None
                            and abstract_top is not None
                            and line_rect.y0 >= author_band_bottom
                            and line_rect.y1 <= abstract_top
                        )
                    )
                    and IDENTITY_LINE_PATTERN.search(line_text)
                ):
                    page_rects.append(
                        expand_rect(line_rect, page=page, x_padding=2.0, y_padding=9.0)
                    )
                    identity_line_count += 1

            for link in page.get_links():
                uri = str(link.get("uri") or "")
                if uri.lower().startswith("mailto:"):
                    page.delete_link(link)
                    mail_link_count += 1
                elif uri:
                    page.delete_link(link)
                    external_link_count += 1

            for url_match in URL_PATTERN.finditer(page.get_text()):
                url = url_match.group(0).rstrip(".,;:!?)]}")
                url_rects = list(page.search_for(url))
                page_rects.extend(url_rects)
                url_redaction_count += len(url_rects)

            for rect in dedupe_rects(page_rects):
                page.add_redact_annot(rect, fill=(1, 1, 1), cross_out=False)
                redaction_count += 1
            if page_rects:
                page.apply_redactions(
                    images=pymupdf.PDF_REDACT_IMAGE_NONE,
                    graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                    text=pymupdf.PDF_REDACT_TEXT_REMOVE,
                )

        output.set_metadata({})
        if hasattr(output, "del_xml_metadata"):
            output.del_xml_metadata()
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        output.save(out_pdf, garbage=4, clean=True, deflate=True)
    finally:
        try:
            if output is not None:
                output.close()
        finally:
            diagnostics_text = pymupdf.TOOLS.mupdf_warnings(reset=1)
            pymupdf.TOOLS.mupdf_display_errors(previous_error_display)

    diagnostics = list(
        dict.fromkeys(
            line.strip()
            for line in str(diagnostics_text or "").splitlines()
            if line.strip()
        )
    )

    validation = validate_anonymized_pdf(out_pdf, authors=authors)
    authors_not_located = sorted(author for author, count in author_hits.items() if count == 0)
    status = (
        "ok"
        if first_page_author_band_redacted
        and not validation["residual_authors"]
        and validation["residual_email_count"] == 0
        and validation["residual_url_count"] == 0
        and not validation["metadata_author"]
        and not validation["xml_metadata_present"]
        else "needs_review"
    )
    return {
        "status": status,
        "pages_written": validation["page_count"],
        "redaction_count": redaction_count,
        "identity_line_count": identity_line_count,
        "mail_link_count": mail_link_count,
        "external_link_count": external_link_count,
        "url_redaction_count": url_redaction_count,
        "author_hits": author_hits,
        "authors_not_located": authors_not_located,
        "first_page_author_band_redacted": first_page_author_band_redacted,
        "source_repair_diagnostic_count": len(diagnostics),
        "source_repair_diagnostics": diagnostics[:20],
        **validation,
    }


def validate_anonymized_pdf(path: Path, *, authors: list[str]) -> dict[str, Any]:
    document = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in document)
    metadata_author = str(document.metadata.get("author") or "")
    xml_metadata_present = bool(document.get_xml_metadata())
    page_count = len(document)
    document.close()
    residual_authors = [
        author
        for author in authors
        if contains_author_identity(text, author)
    ]
    return {
        "page_count": page_count,
        "residual_authors": residual_authors,
        "residual_email_count": len(EMAIL_PATTERN.findall(text)),
        "residual_url_count": len(URL_PATTERN.findall(text)),
        "metadata_author": metadata_author,
        "xml_metadata_present": xml_metadata_present,
    }


def anonymize_representation(
    text: str,
    *,
    authors: list[str],
    strip_front_matter: bool,
) -> tuple[str, dict[str, Any]]:
    sanitized = text
    front_matter_removed = False
    if strip_front_matter:
        sanitized, front_matter_removed = remove_front_matter(sanitized)
    sanitized, acknowledgements_removed = remove_acknowledgements(sanitized)
    replacement_count = 0
    identity_line_replacement_count = 0
    lines = sanitized.splitlines()
    author_flags = [
        any(contains_author_identity(line, author) for author in authors)
        for line in lines
    ]
    identity_flags = [
        is_text_identity_line(line)
        for line in lines
    ]
    identity_block_lines = find_identity_block_lines(
        lines,
        direct_flags=[
            author_flag or identity_flag
            for author_flag, identity_flag in zip(author_flags, identity_flags, strict=True)
        ],
    )
    sanitized_lines: list[str] = []
    for index, line in enumerate(lines):
        if author_flags[index]:
            sanitized_lines.append("[IDENTITY REDACTED]")
            replacement_count += 1
        elif identity_flags[index] or index in identity_block_lines:
            sanitized_lines.append("[IDENTITY REDACTED]")
            identity_line_replacement_count += 1
        else:
            sanitized_lines.append(line)
    sanitized = "\n".join(sanitized_lines)
    sanitized, email_count = EMAIL_PATTERN.subn("[EMAIL REDACTED]", sanitized)
    sanitized, url_count = URL_PATTERN.subn("[URL REDACTED]", sanitized)
    residual_authors = [
        author
        for author in authors
        if contains_author_identity(sanitized, author)
    ]
    return sanitized, {
        "front_matter_removed": front_matter_removed,
        "acknowledgements_removed": acknowledgements_removed,
        "author_replacement_count": replacement_count,
        "identity_line_replacement_count": identity_line_replacement_count,
        "email_replacement_count": email_count,
        "url_replacement_count": url_count,
        "residual_authors": residual_authors,
    }


def is_text_identity_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and len(stripped) <= 220 and TEXT_IDENTITY_LINE_PATTERN.search(stripped))


def find_identity_block_lines(
    lines: list[str],
    *,
    direct_flags: list[bool],
) -> set[int]:
    block_lines: set[int] = set()
    for start, line in enumerate(lines):
        if not re.fullmatch(r"\s*[*†‡]+\s*", line):
            continue
        lookahead_end = min(len(lines), start + 13)
        if not any(direct_flags[start + 1 : lookahead_end]):
            continue
        window_end = min(len(lines), start + 51)
        direct_indices = [
            index
            for index in range(start + 1, window_end)
            if direct_flags[index]
        ]
        if direct_indices:
            block_lines.update(range(start, direct_indices[-1] + 1))
    early_limit = min(len(lines), 120)
    for direct_index in range(early_limit):
        if not direct_flags[direct_index]:
            continue
        window_start = max(0, direct_index - 15)
        window_end = min(early_limit, direct_index + 21)
        cue_indices = [
            index
            for index in range(window_start, window_end)
            if AFFILIATION_FRAGMENT_PATTERN.search(lines[index])
        ]
        if not cue_indices:
            continue
        related_indices = cue_indices + [
            index
            for index in range(window_start, window_end)
            if direct_flags[index]
        ]
        block_start = min(related_indices)
        if block_start > 0 and re.fullmatch(r"\s*(?:\d+|[*†‡]+)\s*", lines[block_start - 1]):
            block_start -= 1
        block_end = max(related_indices)
        block_lines.update(range(block_start, block_end + 1))
    return block_lines


def remove_front_matter(text: str) -> tuple[str, bool]:
    header_match = re.match(
        r"(?s)\A(?P<header>Title:[^\n]*\n\n(?:Representation:.*?\n\n)?)",
        text,
    )
    header = header_match.group("header") if header_match else ""
    body = text[len(header) :]
    abstract = re.search(r"(?im)^[ \t]*abstract[ \t]*$", body)
    if not abstract:
        return text, False
    return header + body[abstract.start() :], bool(body[: abstract.start()].strip())


def remove_acknowledgements(text: str) -> tuple[str, bool]:
    match = ACKNOWLEDGEMENTS_PATTERN.search(text)
    if not match:
        return text, False
    return text[: match.start()].rstrip() + "\n", True


def find_name_rects(words: list[tuple[Any, ...]], name: str) -> list[pymupdf.Rect]:
    target = normalize_identity_text(name)
    if not target:
        return []
    normalized_words = [normalize_identity_token(str(word[4])) for word in words]
    results: list[pymupdf.Rect] = []
    for start, token in enumerate(normalized_words):
        if not token or not target.startswith(token):
            continue
        matched_indices = [start]
        accumulated = token
        word_index = start + 1
        while word_index < len(words) and accumulated != target:
            candidate = normalized_words[word_index]
            proposed = accumulated + candidate
            if candidate and target.startswith(proposed):
                matched_indices.append(word_index)
                accumulated = proposed
            elif not candidate or candidate.isdigit() or len(candidate) == 1:
                matched_indices.append(word_index)
            else:
                break
            word_index += 1
        if accumulated == target:
            rect = pymupdf.Rect(words[matched_indices[0]][:4])
            for index in matched_indices[1:]:
                rect.include_rect(pymupdf.Rect(words[index][:4]))
            results.append(rect)
    return dedupe_rects(results)


def iter_page_lines(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    lines: list[tuple[str, pymupdf.Rect]] = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if text:
                lines.append((text, pymupdf.Rect(line["bbox"])))
    return lines


def dedupe_rects(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    unique: list[pymupdf.Rect] = []
    seen: set[tuple[int, int, int, int]] = set()
    for rect in rects:
        key = tuple(round(value * 10) for value in (rect.x0, rect.y0, rect.x1, rect.y1))
        if key not in seen and rect.width > 0 and rect.height > 0:
            seen.add(key)
            unique.append(rect)
    return unique


def expand_rect(
    rect: pymupdf.Rect,
    *,
    page: pymupdf.Page,
    x_padding: float,
    y_padding: float,
) -> pymupdf.Rect:
    return pymupdf.Rect(
        max(0.0, rect.x0 - x_padding),
        max(0.0, rect.y0 - y_padding),
        min(page.rect.width, rect.x1 + x_padding),
        min(page.rect.height, rect.y1 + y_padding),
    )


def representation_paths(record: PaperRecord) -> dict[str, Path | None]:
    artifacts = record.extra.get("text_artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    values = {
        "compact_repr": record.text_compact or artifacts.get("compact_repr"),
        "full_repr": record.text_full or artifacts.get("full_repr"),
        "scoring_repr": record.text_scoring or artifacts.get("scoring_repr"),
    }
    return {name: Path(value) if value else None for name, value in values.items()}


def scoring_artifacts_exist(record: PaperRecord) -> bool:
    artifacts = record.extra.get("scoring_artifacts")
    if not isinstance(artifacts, dict):
        return False
    return all(
        isinstance(artifacts.get(name), str) and Path(artifacts[name]).exists()
        for name in ("pdf", "compact_repr", "full_repr", "scoring_repr")
    )


def anonymization_fingerprint(record: PaperRecord, *, max_pages: int) -> str:
    text_paths = representation_paths(record)
    payload = "\n".join(
        [
            ANONYMIZATION_VERSION,
            str(max_pages),
            record.paper_id,
            record.title or "",
            "\x1f".join(record.authors),
            str(record.pdf_path or ""),
            sha256_file(Path(record.pdf_path)) if record.pdf_path and Path(record.pdf_path).exists() else "",
            *[
                f"{name}:{sha256_file(path) if path is not None and path.exists() else ''}"
                for name, path in sorted(text_paths.items())
            ],
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_identity_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    normalized = "".join(character for character in decomposed.casefold() if character.isalnum())
    return normalize_transliteration(normalized)


def normalize_identity_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    normalized = "".join(character for character in decomposed.casefold() if character.isalnum())
    return normalize_transliteration(normalized)


def normalize_transliteration(value: str) -> str:
    return value.replace("ae", "a").replace("oe", "o").replace("ue", "u")


def author_name_variants(author: str) -> list[str]:
    variants = [author]
    tokens = re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", author, flags=re.UNICODE)
    if len(tokens) > 2:
        variants.append(f"{tokens[0]} {tokens[-1]}")
    return list(dict.fromkeys(variants))


def author_identity_keys(author: str) -> list[str]:
    return [
        key
        for variant in author_name_variants(author)
        if (key := normalize_identity_text(variant))
    ]


def contains_author_identity(text: str, author: str) -> bool:
    return any(
        line_contains_name_variant(line, variant)
        for line in text.splitlines()
        for variant in author_name_variants(author)
    )


def line_contains_name_variant(line: str, variant: str) -> bool:
    target = normalize_identity_text(variant)
    tokens = [
        normalize_identity_token(token)
        for token in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", line, flags=re.UNICODE)
    ]
    for start, token in enumerate(tokens):
        if not token or not target.startswith(token):
            continue
        accumulated = token
        for candidate in tokens[start + 1 :]:
            if accumulated == target:
                return True
            proposed = accumulated + candidate
            if candidate and target.startswith(proposed):
                accumulated = proposed
            elif candidate.isdigit() or len(candidate) == 1:
                continue
            else:
                break
        if accumulated == target:
            return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
