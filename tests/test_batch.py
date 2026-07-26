import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.batch import (
    BatchCandidate,
    Pass1BatchConfig,
    Pass1BatchSuiteConfig,
    build_pass1_batch_messages,
    run_pass1_batch_suite,
)
from icml_ai_ac.storage import write_jsonl


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, temperature, max_output_tokens, seed=None, response_format=None):
        self.calls += 1
        paper_ids = re.findall(r'<PAPER paper_id="(p\d+)">', messages[-1]["content"])
        content = json.dumps(
            {
                "ranked_papers": [
                    {
                        "rank": rank,
                        "paper_id": paper_id,
                        "forced_bucket": "top_10_percent" if rank == 1 else "middle",
                        "estimated_percentile_among_batch": 100 - rank * 10,
                        "primary_contribution_class": "theory",
                        "executive_ac_priority": 10 - rank,
                        "broader_science_impact_forecast": 9 - rank,
                        "ml_field_impact_forecast": 8 - rank,
                        "technical_soundness": 7,
                        "overall_significance": 8,
                        "should_advance_to_strong_model": rank == 1,
                        "category_standout": rank == 1,
                    }
                    for rank, paper_id in enumerate(paper_ids, start=1)
                ]
            }
        )
        return SimpleNamespace(
            response={"model": "served"},
            content=content,
            usage={"prompt_tokens": 20, "cost": 0.02},
            served_model="served",
            elapsed_seconds=0.01,
        )


class Pass1BatchResumeTests(unittest.TestCase):
    def test_batch_prompt_excludes_human_outcome_metadata(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="SENTINEL_HUMAN_SOURCE",
            title="Paper",
            decision_label="SENTINEL_HUMAN_DECISION",
        )
        candidate = BatchCandidate(record, "paper.txt", "scoring", "Paper content.")
        config = Pass1BatchConfig(
            provider="openrouter",
            model="test",
            prompt_version="test",
            text_source="scoring",
            limit=None,
            paper_ids=set(),
            per_paper_char_budget=1000,
            temperature=0,
            max_output_tokens=1000,
            seed=None,
            dry_run=True,
        )

        messages = build_pass1_batch_messages([candidate], config=config)
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertNotIn("SENTINEL_HUMAN_SOURCE", prompt_text)
        self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt_text)
        self.assertIn("No reviews, reviewer scores", prompt_text)

    def test_suite_resumes_only_matching_valid_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(4):
                text_path = root / f"p{index}.txt"
                text_path.write_text(f"Paper {index}", encoding="utf-8")
                records.append(
                    PaperRecord(
                        paper_id=f"p{index}",
                        source="accepted",
                        title=f"Paper {index}",
                        text_scoring=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                )
            manifest = root / "manifest.jsonl"
            out = root / "scores.jsonl"
            run_dir = root / "run"
            write_jsonl(manifest, records)
            config = Pass1BatchSuiteConfig(
                provider="openrouter",
                model="test/model",
                reasoning_effort="none",
                prompt_version="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                batch_size=2,
                partitions=2,
                strategy="shuffled",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=1000,
                seed=3,
                dry_run=False,
            )
            client = FakeClient()

            first = run_pass1_batch_suite(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )
            second = run_pass1_batch_suite(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )

            self.assertEqual(first["status"], "ok", first)
            self.assertEqual(first["rows"], 8)
            self.assertEqual(first["usage"]["cost"], 0.08)
            self.assertEqual(client.calls, 4)
            self.assertEqual(second["resumed_batch_count"], 4)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
