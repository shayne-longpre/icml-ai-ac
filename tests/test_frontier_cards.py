import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from icml_ai_ac.scoring.frontier_gold import FrontierCardConfig, run_frontier_pdf_cards


class FrontierCardFallbackTests(unittest.TestCase):
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
                    }
                return {
                    "status": "ok",
                    "paper_id": "p1",
                    "model": "fallback",
                    "served_model": "fallback",
                    "usage": {"cost": 0.5},
                    "card": {"paper_id": "p1"},
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
            ):
                result = run_frontier_pdf_cards(
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


if __name__ == "__main__":
    unittest.main()
