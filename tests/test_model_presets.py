import unittest

from icml_ai_ac.cli import build_parser
from icml_ai_ac.model_presets import PRODUCTION_SEMIFINAL_JUDGES


class ModelPresetTests(unittest.TestCase):
    def test_cli_defaults_match_production_model_configuration(self) -> None:
        parser = build_parser()
        pass1 = parser.parse_args(
            ["score-pass1-ensemble", "--manifest", "manifest.jsonl", "--out-dir", "runs"]
        )
        pass2 = parser.parse_args(
            [
                "rank-pass2",
                "--manifest",
                "manifest.jsonl",
                "--pass1",
                "pass1.jsonl",
                "--out",
                "pass2.json",
                "--run-dir",
                "runs",
            ]
        )
        tournament = parser.parse_args(
            [
                "rank-frontier-card-tournament",
                "--cards",
                "cards.jsonl",
                "--seed-ranking",
                "seed.json",
                "--out",
                "ranking.json",
                "--run-dir",
                "runs",
            ]
        )

        self.assertEqual(pass1.model_preset, "production_2026_v2")
        self.assertEqual(pass1.reasoning_effort, "none")
        self.assertEqual(pass2.model, "gpt-5.6-terra")
        self.assertEqual(pass2.reasoning_effort, "high")
        self.assertEqual(tournament.model, "openai/gpt-5.6-sol")
        self.assertEqual(tournament.reasoning_effort, "xhigh")
        self.assertEqual(
            [(judge.provider, judge.model, judge.reasoning_effort) for judge in PRODUCTION_SEMIFINAL_JUDGES],
            [
                ("openai", "gpt-5.6-terra", "high"),
                ("openrouter", "anthropic/claude-sonnet-5", "high"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
