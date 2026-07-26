import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from icml_ai_ac.scoring.frontier_gold import (
    FrontierCardConfig,
    FrontierCardCandidate,
    FrontierTournamentConfig,
    build_frontier_card_messages,
    frontier_card_request_fingerprint,
    run_frontier_pdf_cards,
    run_pair_batches,
)
from icml_ai_ac.models import PaperRecord


class FrontierCardFallbackTests(unittest.TestCase):
    def test_card_fingerprint_covers_model_and_content_but_not_human_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            record = PaperRecord(
                paper_id="p1",
                source="SENTINEL_HUMAN_SOURCE",
                decision_label="SENTINEL_HUMAN_DECISION",
                title="Paper",
                extra={"pdf_sha256": "abc123"},
            )
            candidate = FrontierCardCandidate(
                record=record,
                text_path="paper.txt",
                resolved_text_source="scoring",
                paper_text="Paper content.",
                pdf_path=pdf_path,
                pdf_excerpt_path=root / "excerpt.pdf",
            )
            candidate.pdf_excerpt_path.write_bytes(b"%PDF-1.4 excerpt")
            config = FrontierCardConfig(
                provider="openrouter",
                model="model-a",
                reasoning_effort="high",
                prompt_version="test",
                paper_set_name="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                pdf_excerpt_pages=9,
                pdf_excerpt_dir=root,
                pdf_optimize_threshold_bytes=None,
                pdf_settings="/ebook",
                temperature=0,
                max_output_tokens=1000,
                seed=None,
                dry_run=False,
                overwrite=False,
                openrouter_pdf_engine="native",
                fallback_model=None,
                fallback_reasoning_effort=None,
            )

            fingerprint = frontier_card_request_fingerprint(candidate, config=config)
            messages = build_frontier_card_messages(candidate, config=config)
            prompt_text = json.dumps(messages)
            record.source = "different human source"
            record.decision_label = "different human decision"

            self.assertNotIn("SENTINEL_HUMAN_SOURCE", prompt_text)
            self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt_text)
            self.assertIn("No reviews, reviewer scores", prompt_text)
            self.assertEqual(fingerprint, frontier_card_request_fingerprint(candidate, config=config))
            self.assertNotEqual(
                fingerprint,
                frontier_card_request_fingerprint(
                    candidate,
                    config=replace(config, model="model-b"),
                ),
            )

    def test_failed_primary_uses_and_records_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = SimpleNamespace(record=SimpleNamespace(paper_id="p1"))
            config = FrontierCardConfig(
                provider="openrouter",
                model="primary",
                reasoning_effort="high",
                prompt_version="test",
                paper_set_name="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                pdf_excerpt_pages=9,
                pdf_excerpt_dir=root / "excerpts",
                pdf_optimize_threshold_bytes=None,
                pdf_settings="/ebook",
                temperature=0,
                max_output_tokens=1000,
                seed=None,
                dry_run=False,
                overwrite=False,
                openrouter_pdf_engine="native",
                fallback_model="fallback",
                fallback_reasoning_effort="high",
            )

            def fake_card(candidate, *, run_dir, config, **kwargs):
                if config.model == "primary":
                    return {
                        "status": "failed",
                        "paper_id": "p1",
                        "model": "primary",
                        "error": "refused",
                        "usage": {"cost": 0.2, "prompt_tokens": 100},
                        "fingerprint": "fp-primary",
                    }
                return {
                    "status": "ok",
                    "paper_id": "p1",
                    "model": "fallback",
                    "served_model": "fallback",
                    "usage": {"cost": 0.5},
                    "card": {"paper_id": "p1"},
                    "fingerprint": "fp-fallback",
                }

            with (
                patch(
                    "icml_ai_ac.scoring.frontier_gold.select_frontier_card_candidates",
                    return_value=[candidate],
                ),
                patch(
                    "icml_ai_ac.scoring.frontier_gold.run_frontier_pdf_card",
                    side_effect=fake_card,
                ) as card_call,
                patch(
                    "icml_ai_ac.scoring.frontier_gold.frontier_card_request_fingerprint",
                    side_effect=lambda candidate, *, config: f"fp-{config.model}",
                ),
            ):
                result = run_frontier_pdf_cards(
                    manifest=root / "manifest.jsonl",
                    out=root / "cards.jsonl",
                    run_dir=root / "run",
                    config=config,
                    client=object(),
                    fallback_client=object(),
                )
                resumed = run_frontier_pdf_cards(
                    manifest=root / "manifest.jsonl",
                    out=root / "cards.jsonl",
                    run_dir=root / "run",
                    config=config,
                    client=object(),
                    fallback_client=object(),
                )

            row = json.loads((root / "cards.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["fallback_count"], 1)
            self.assertEqual(result["usage"], {"cost": 0.7, "prompt_tokens": 100})
            self.assertEqual(row["fallback_from_model"], "primary")
            self.assertEqual(row["primary_attempt"]["error"], "refused")
            self.assertEqual(row["attempt_usage"], {"cost": 0.7, "prompt_tokens": 100})
            self.assertEqual(card_call.call_count, 2)
            self.assertEqual(resumed["ok_count"], 1)
            self.assertEqual(resumed["fallback_count"], 1)

    def test_billing_error_stops_cards_without_fallback_and_resumes_later(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [
                SimpleNamespace(record=SimpleNamespace(paper_id="p1")),
                SimpleNamespace(record=SimpleNamespace(paper_id="p2")),
            ]
            config = FrontierCardConfig(
                provider="openrouter",
                model="primary",
                reasoning_effort="high",
                prompt_version="test",
                paper_set_name="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                pdf_excerpt_pages=9,
                pdf_excerpt_dir=root / "excerpts",
                pdf_optimize_threshold_bytes=None,
                pdf_settings="/ebook",
                temperature=0,
                max_output_tokens=1000,
                seed=None,
                dry_run=False,
                overwrite=False,
                openrouter_pdf_engine="native",
                fallback_model="fallback",
                fallback_reasoning_effort="high",
            )
            blocked = {
                "status": "failed",
                "paper_id": "p1",
                "model": "primary",
                "error": "HTTP 402 from provider: insufficient credits",
                "blocking_provider_error": True,
                "blocked_reason": "HTTP 402 from provider: insufficient credits",
            }

            with (
                patch(
                    "icml_ai_ac.scoring.frontier_gold.select_frontier_card_candidates",
                    return_value=candidates,
                ),
                patch(
                    "icml_ai_ac.scoring.frontier_gold.run_frontier_pdf_card",
                    return_value=blocked,
                ) as card_call,
            ):
                result = run_frontier_pdf_cards(
                    manifest=root / "manifest.jsonl",
                    out=root / "cards.jsonl",
                    run_dir=root / "run",
                    config=config,
                    client=object(),
                    fallback_client=object(),
                )

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["attempted_count"], 1)
            self.assertEqual(result["skipped_count"], 1)
            self.assertIn("402", result["blocked_reason"])
            self.assertEqual(card_call.call_count, 1)

    def test_billing_error_stops_remaining_tournament_batches(self) -> None:
        config = FrontierTournamentConfig(
            provider="openrouter",
            model="test",
            reasoning_effort="high",
            prompt_version="test",
            paper_set_name="test",
            strategy="all_pairs",
            top_n=3,
            pairs_per_batch=1,
            swiss_rounds=1,
            playoff_top_n=None,
            temperature=0,
            max_output_tokens=1000,
            seed=7,
            dry_run=False,
            overwrite=False,
        )
        blocked = {
            "status": "failed",
            "blocking_provider_error": True,
            "blocked_reason": "HTTP 402 from provider: insufficient credits",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "icml_ai_ac.scoring.frontier_gold.run_frontier_tournament_batch",
                return_value=blocked,
            ) as batch_call,
        ):
            results, matches = run_pair_batches(
                pair_batches=[[("p1", "p2")], [("p1", "p3")]],
                rows_by_paper={},
                seed_ranked=[],
                start_batch_index=0,
                batch_count=2,
                run_dir=Path(tmp),
                config=config,
                client=object(),
            )

        self.assertEqual(results, [blocked])
        self.assertEqual(matches, [])
        self.assertEqual(batch_call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
