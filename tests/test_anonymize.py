import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from icml_ai_ac.anonymize import (
    AnonymizationConfig,
    anonymize_manifest,
    anonymize_pdf,
    anonymize_representation,
    contains_author_identity,
    filter_author_identities,
)
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.runner import resolve_record_pdf_path, resolve_record_text_path


class AnonymizationTests(unittest.TestCase):
    def test_identity_matching_handles_camera_ready_name_variants(self) -> None:
        self.assertTrue(contains_author_identity("Leello Dadi", "Leello Tadesse Dadi"))
        self.assertTrue(contains_author_identity("Christoph Schnörr", "Christoph Schnoerr"))
        self.assertTrue(contains_author_identity("Mark N. Müller", "Mark Niklas Mueller"))
        self.assertTrue(contains_author_identity("O˘guz Kaan Y¨uksel 1", "Oğuz Yüksel"))
        self.assertFalse(contains_author_identity("Mueller et al. provide a baseline.", "Mark Niklas Mueller"))

    def test_pdf_redaction_removes_identity_but_preserves_paper_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            output = root / "anonymized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Machine Learning Paper", fontsize=16)
            page.insert_text((72, 108), "Ada Lovelace  Grace Hopper", fontsize=11)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            page.insert_text((72, 165), "This paper presents a useful scientific result.", fontsize=10)
            page.insert_text((72, 185), "Code: https://example.edu/ada/project", fontsize=10)
            page.insert_text((72, 700), "Department of Computing, Example University", fontsize=8)
            page.insert_text((72, 714), "Correspondence to: ada@example.edu", fontsize=8)
            document.set_metadata({"author": "Ada Lovelace; Grace Hopper"})
            document.set_xml_metadata(
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            )
            document.save(source)
            document.close()

            audit = anonymize_pdf(
                source_pdf=source,
                out_pdf=output,
                title="A Useful Machine Learning Paper",
                authors=["Ada Lovelace", "Grace Hopper"],
                max_pages=9,
            )

            anonymized = pymupdf.open(output)
            text = "\n".join(page.get_text() for page in anonymized)
            metadata = anonymized.metadata
            anonymized.close()
            self.assertEqual(audit["status"], "ok")
            self.assertNotIn("Ada Lovelace", text)
            self.assertNotIn("Grace Hopper", text)
            self.assertNotIn("Example University", text)
            self.assertNotIn("ada@example.edu", text)
            self.assertNotIn("example.edu/ada", text)
            self.assertIn("A Useful Machine Learning Paper", text)
            self.assertIn("useful scientific result", text)
            self.assertFalse(metadata.get("author"))
            self.assertFalse(audit["xml_metadata_present"])

    def test_pdf_redaction_tolerates_stale_metadata_when_author_band_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            output = root / "anonymized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Machine Learning Paper", fontsize=16)
            page.insert_text((72, 108), "Ada Lovelace  New Camera Ready Author", fontsize=11)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            page.insert_text((72, 165), "This paper presents a useful scientific result.", fontsize=10)
            document.save(source)
            document.close()

            audit = anonymize_pdf(
                source_pdf=source,
                out_pdf=output,
                title="A Useful Machine Learning Paper",
                authors=["Ada Lovelace", "Stale Metadata Author"],
                max_pages=9,
            )

            anonymized = pymupdf.open(output)
            text = "\n".join(page.get_text() for page in anonymized)
            anonymized.close()
            self.assertEqual(audit["status"], "ok")
            self.assertTrue(audit["first_page_author_band_redacted"])
            self.assertEqual(audit["authors_not_located"], ["Stale Metadata Author"])
            self.assertNotIn("Ada Lovelace", text)
            self.assertNotIn("New Camera Ready Author", text)
            self.assertIn("useful scientific result", text)

    def test_pdf_redaction_fails_closed_without_a_detected_author_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            output = root / "anonymized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Machine Learning Paper", fontsize=16)
            page.insert_text((72, 108), "New Camera Ready Author", fontsize=11)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            document.save(source)
            document.close()

            audit = anonymize_pdf(
                source_pdf=source,
                out_pdf=output,
                title="A Useful Machine Learning Paper",
                authors=["Stale Metadata Author"],
                max_pages=9,
            )

            self.assertEqual(audit["status"], "needs_review")
            self.assertFalse(audit["first_page_author_band_redacted"])

    def test_pdf_redaction_removes_entire_superscripted_identity_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            output = root / "anonymized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Machine Learning Paper", fontsize=16)
            page.insert_text((72, 108), "Laura Lutzow1,2", fontsize=11)
            page.insert_text((350, 108), "laura@example.edu", fontsize=9)
            page.insert_text((72, 122), "1 Example University", fontsize=9)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            page.insert_text((72, 165), "Scientific content.", fontsize=10)
            document.save(source)
            document.close()

            audit = anonymize_pdf(
                source_pdf=source,
                out_pdf=output,
                title="A Useful Machine Learning Paper",
                authors=["Laura Lützow"],
                max_pages=9,
            )

            anonymized = pymupdf.open(output)
            text = "\n".join(page.get_text() for page in anonymized)
            anonymized.close()
            self.assertEqual(audit["status"], "ok")
            self.assertIn("known_author", audit["first_page_identity_band_reason"])
            self.assertIn("email", audit["first_page_identity_band_reason"])
            self.assertNotIn("Laura", text)
            self.assertNotIn("Example University", text)
            self.assertIn("Scientific content.", text)

    def test_pdf_redaction_accepts_an_already_anonymous_byline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pdf"
            output = root / "anonymized.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Machine Learning Paper", fontsize=16)
            page.insert_text((220, 108), "Anonymous Author(s)", fontsize=11)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            page.insert_text((72, 165), "Scientific content.", fontsize=10)
            page.insert_text(
                (72, 700),
                "For ICML 2026 reviewers: follow the Reviewer Console policy.",
                fontsize=8,
            )
            page.insert_text(
                (72, 714),
                "The assigned LLM policy might differ from registration.",
                fontsize=8,
            )
            document.save(source)
            document.close()

            audit = anonymize_pdf(
                source_pdf=source,
                out_pdf=output,
                title="A Useful Machine Learning Paper",
                authors=["Camera Ready Author"],
                max_pages=9,
            )

            anonymized = pymupdf.open(output)
            text = "\n".join(page.get_text() for page in anonymized)
            anonymized.close()
            self.assertEqual(audit["status"], "ok")
            self.assertEqual(
                audit["first_page_identity_band_reason"],
                ["anonymous_author"],
            )
            self.assertNotIn("Anonymous Author", text)
            self.assertNotIn("Reviewer Console", text)
            self.assertNotIn("assigned LLM policy", text)
            self.assertIn("Scientific content.", text)

    def test_short_uppercase_metadata_identity_is_ignored(self) -> None:
        active, ignored = filter_author_identities(
            ["Lei Wei", "TT", "Xi"],
        )

        self.assertEqual(active, ["Lei Wei", "Xi"])
        self.assertEqual(ignored, ["TT"])

    def test_text_anonymization_removes_front_matter_and_acknowledgements(self) -> None:
        text = """Title: A Useful Paper

Representation: main paper body extracted from the PDF.

A Useful Paper
Ada Lovelace and Grace Hopper
Example University

Abstract
The method is useful.

1 Introduction
The method is described here.
Ada Lovelace
Example Research Center, London
Department of Computing, Example University
Code is available at https://example.edu/ada/project.
The scientific discussion continues after the affiliation block.

Acknowledgements
Ada Lovelace thanks Example University.
"""

        sanitized, audit = anonymize_representation(
            text,
            authors=["Ada Lovelace", "Grace Hopper"],
            strip_front_matter=True,
        )

        self.assertTrue(audit["front_matter_removed"])
        self.assertTrue(audit["acknowledgements_removed"])
        self.assertNotIn("Ada Lovelace", sanitized)
        self.assertNotIn("Grace Hopper", sanitized)
        self.assertNotIn("Example University", sanitized)
        self.assertNotIn("Example Research Center", sanitized)
        self.assertNotIn("example.edu/ada", sanitized)
        self.assertIn("Title: A Useful Paper", sanitized)
        self.assertIn("The method is described here.", sanitized)
        self.assertIn("scientific discussion continues", sanitized)
        self.assertGreaterEqual(audit["identity_line_replacement_count"], 1)
        self.assertEqual(audit["url_replacement_count"], 1)

    def test_manifest_is_resumable_and_scoring_prefers_anonymized_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_pdf = root / "source.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "A Useful Paper", fontsize=16)
            page.insert_text((72, 108), "Ada Lovelace", fontsize=11)
            page.insert_text((72, 145), "Abstract", fontsize=12)
            page.insert_text((72, 165), "Scientific content.", fontsize=10)
            document.save(source_pdf)
            document.close()
            source_text_dir = root / "source_text"
            source_text_dir.mkdir()
            paths = {}
            for name in ("compact_repr", "full_repr", "scoring_repr"):
                path = source_text_dir / f"{name}.txt"
                path.write_text(
                    "Title: A Useful Paper\n\nAda Lovelace\n\nAbstract\nScientific content.\n",
                    encoding="utf-8",
                )
                paths[name] = str(path)
            record = PaperRecord(
                paper_id="paper-1",
                source="accepted",
                decision_label="SENTINEL_HUMAN_DECISION",
                title="A Useful Paper",
                authors=["Ada Lovelace"],
                forum_url="https://openreview.net/forum?id=paper-1",
                pdf_url="https://openreview.net/pdf?id=paper-1",
                pdf_path=str(source_pdf),
                text_compact=paths["compact_repr"],
                text_full=paths["full_repr"],
                text_scoring=paths["scoring_repr"],
                parse_status="ok",
                extra={
                    "openreview_scores": {"overall_mean": 5.0},
                    "presentation_type": "oral",
                    "is_award_paper": True,
                },
            )
            config = AnonymizationConfig(
                pdf_dir=root / "pdfs",
                text_dir=root / "text",
                max_pages=9,
            )
            out = root / "manifest.jsonl"
            report_path = root / "report.json"

            first = anonymize_manifest(records=[record], out=out, report_path=report_path, config=config)
            second = anonymize_manifest(records=[record], out=out, report_path=report_path, config=config)

            self.assertEqual(first["output_records"], 1)
            self.assertEqual(second["resumed_records"], 1)
            anonymized_record = PaperRecord.from_dict(
                json.loads(out.read_text(encoding="utf-8"))
            )
            text_path, text_source = resolve_record_text_path(anonymized_record, "scoring")
            pdf_path, pdf_source = resolve_record_pdf_path(anonymized_record)
            self.assertEqual(text_source, "anonymized_scoring")
            self.assertEqual(pdf_source, "anonymized_pdf")
            self.assertNotIn("Ada Lovelace", Path(text_path).read_text(encoding="utf-8"))
            self.assertTrue(pdf_path.exists())
            self.assertEqual(anonymized_record.pdf_path, str(pdf_path))
            self.assertEqual(anonymized_record.source, "blinded_scoring")
            self.assertIsNone(anonymized_record.decision_label)
            self.assertEqual(anonymized_record.authors, [])
            self.assertIsNone(anonymized_record.forum_url)
            self.assertIsNone(anonymized_record.pdf_url)
            self.assertEqual(
                set(anonymized_record.extra),
                {"anonymization", "scoring_artifacts", "blind_manifest"},
            )
            self.assertNotIn(
                "author_hits",
                anonymized_record.extra["anonymization"],
            )
            serialized = json.dumps(anonymized_record.to_dict())
            self.assertNotIn("Ada Lovelace", serialized)
            self.assertNotIn("accepted", serialized)


if __name__ == "__main__":
    unittest.main()
