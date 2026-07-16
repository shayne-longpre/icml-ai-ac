import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.scoring.ranking import write_ranking_score_rows
from icml_ai_ac.eval import read_ranking


class RankingTests(unittest.TestCase):
    def test_write_ranking_score_rows_converts_pass2_json_for_stage2(self) -> None:
        result = {
            "provider": "openai",
            "model": "gpt-5.4",
            "prompt_version": "test_prompt",
            "out": "stage1.json",
            "ranking": {
                "ranked_papers": [
                    {
                        "rank": 1,
                        "paper_id": "p1",
                        "title": "Paper 1",
                        "primary_contribution_class": "theory",
                        "overall_priority_score": 9,
                        "broad_scientific_impact_score": 8,
                        "ml_field_impact_score": 9,
                        "technical_soundness_score": 8,
                        "advance_to_final_review": True,
                        "why_ranked_here": "strong",
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            count = write_ranking_score_rows(result, out)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["paper_id"], "p1")
        self.assertEqual(rows[0]["scores"]["contribution_profile"]["primary_contribution_class"], "theory")
        self.assertEqual(rows[0]["scores"]["scores"]["executive_ac_priority"], 9.0)

    def test_eval_ranking_rejects_duplicate_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.json"
            path.write_text(
                json.dumps(
                    {
                        "ranked_papers": [
                            {"rank": 1, "paper_id": "p1"},
                            {"rank": 2, "paper_id": "p1"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate paper_id"):
                read_ranking(path)


if __name__ == "__main__":
    unittest.main()
