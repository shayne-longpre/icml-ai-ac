import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.shortlist import build_shortlist, select_ranked_rows


def score_row(paper_id: str, model: str, priority: float, *, advance: bool = True) -> dict:
    return {
        "status": "ok",
        "paper_id": paper_id,
        "title": paper_id,
        "provider": "openrouter",
        "model": model,
        "batch_local_priority": priority,
        "scores": {
            "contribution_profile": {"primary_contribution_class": "core_ml_algorithm"},
            "scores": {
                "executive_ac_priority": priority * 10,
                "broader_science_impact_forecast": priority * 10,
                "ml_field_impact_forecast": priority * 10,
                "overall_significance": priority * 10,
                "technical_soundness": 8,
            },
            "ranking_signals": {"should_advance_to_strong_model": advance},
        },
    }


class ShortlistTests(unittest.TestCase):
    def test_build_shortlist_accepts_multiple_score_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        score_row("p1", "model-a", 0.9),
                        score_row("p2", "model-a", 0.5, advance=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        score_row("p1", "model-b", 0.8),
                        score_row("p3", "model-b", 0.7),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "shortlist.jsonl"
            report = root / "report.json"

            payload = build_shortlist(
                scores_path=[first, second],
                out=out,
                report=report,
                limit=None,
                min_per_class=0,
                class_path=None,
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(payload["source_count"], 2)
            self.assertEqual(payload["unique_papers"], 3)
            self.assertEqual(rows[0]["paper_id"], "p1")
            self.assertEqual(rows[0]["shortlist_metrics"]["source_model_count"], 2)
            self.assertEqual(rows[0]["shortlist_metrics"]["source_rows"][0]["model"], "model-a")

    def test_class_balance_uses_aggregate_rank_not_alphabetical_class_order(self) -> None:
        ranked = [
            {"paper_id": "p1", "primary_contribution_class": "z_class", "aggregate_rank": 1},
            {"paper_id": "p2", "primary_contribution_class": "a_class", "aggregate_rank": 2},
            {"paper_id": "p3", "primary_contribution_class": "m_class", "aggregate_rank": 3},
        ]

        selected = select_ranked_rows(ranked, limit=2, min_per_class=1)

        self.assertEqual([row["paper_id"] for row in selected], ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
