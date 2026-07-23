import unittest

from icml_ai_ac.analysis.agreement import (
    average_ranks,
    build_agreement_report,
    build_axis_decomposition_report,
    kendall_tau_b,
    roc_auc,
)
from icml_ai_ac.analysis.human_comparison import ComparisonRow


def crow(pid, tier, ai_score, *, reviewer=None, axes=None, is_award=False, strong_rank=None):
    tier_rank = {"oral": 2, "spotlight": 1, "poster": 0}[tier]
    return ComparisonRow(
        paper_id=pid,
        title=pid,
        contribution_class="theory",
        tier=tier,
        tier_rank=tier_rank,
        is_award=is_award,
        award_labels=[],
        reviewer_overall=reviewer,
        reviewer_soundness=None,
        reviewer_confidence=None,
        review_count=4 if reviewer is not None else 0,
        ai_score=ai_score,
        ai_reported_percentile=None,
        ai_percentile=ai_score * 10,
        strong_rank=strong_rank,
        ai_axes=axes or {},
    )


class MetricPrimitiveTests(unittest.TestCase):
    def test_kendall_tau_b(self) -> None:
        self.assertEqual(kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertEqual(kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(kendall_tau_b([1], [1]))

    def test_average_ranks(self) -> None:
        self.assertEqual(average_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_roc_auc(self) -> None:
        self.assertEqual(roc_auc([1, 2, 3, 4], [False, False, True, True]), 1.0)
        self.assertEqual(roc_auc([1, 2, 3, 4], [True, False, True, False]), 0.25)
        self.assertIsNone(roc_auc([1, 2, 3], [True, True, True]))


class AgreementReportTests(unittest.TestCase):
    def rows(self):
        return [
            crow("o1", "oral", 9.0, reviewer=8.0, strong_rank=1),
            crow("o2", "oral", 8.5, reviewer=7.5, strong_rank=2),
            crow("s1", "spotlight", 7.0, reviewer=7.0, strong_rank=3),
            crow("p1", "poster", 4.0, reviewer=5.0, strong_rank=4),
            crow("p2", "poster", 3.0, reviewer=4.0),
            crow("p3", "poster", 2.0, reviewer=3.0),
        ]

    def test_agreement_when_ai_tracks_tier(self) -> None:
        report = build_agreement_report(self.rows(), k_values=[2], include_award=False)
        full = report["full_coverage"]
        self.assertEqual(full["roc_auc"]["honored_vs_poster"], 1.0)
        self.assertGreater(full["kendall_tau_b"]["vs_tier"], 0.7)
        self.assertEqual(full["recall_at_k"]["2"]["recall_oral"], 1.0)
        self.assertAlmostEqual(full["recall_at_k"]["2"]["recall_honored"], 2 / 3, places=3)
        self.assertIn("per_tier_ai_percentile", full)

    def test_strong_subset_layered(self) -> None:
        report = build_agreement_report(self.rows(), k_values=[2])
        strong = report["strong_subset"]
        self.assertIsNotNone(strong)
        self.assertEqual(strong["n"], 4)
        self.assertEqual(strong["roc_auc"]["honored_vs_poster"], 1.0)
        self.assertNotIn("per_tier_ai_percentile", strong)  # only on full coverage

    def test_no_strong_ranks_gives_null_subset(self) -> None:
        rows = [crow("p1", "poster", 4.0), crow("o1", "oral", 9.0)]
        report = build_agreement_report(rows, k_values=[1])
        self.assertIsNone(report["strong_subset"])


class AxisDecompositionTests(unittest.TestCase):
    def rows(self):
        # conventional_acceptance_strength tracks tier perfectly; executive is noisier;
        # benchmark axis is anti-correlated with tier.
        return [
            crow("o1", "oral", 9.0, axes={"executive_ac_priority": 9, "conventional_acceptance_strength": 9, "benchmark_or_dataset_value": 1}),
            crow("s1", "spotlight", 5.0, axes={"executive_ac_priority": 5, "conventional_acceptance_strength": 7, "benchmark_or_dataset_value": 9}),
            crow("p1", "poster", 6.0, axes={"executive_ac_priority": 6, "conventional_acceptance_strength": 3, "benchmark_or_dataset_value": 5}),
            crow("p2", "poster", 2.0, axes={"executive_ac_priority": 2, "conventional_acceptance_strength": 2, "benchmark_or_dataset_value": 8}),
        ]

    def test_axes_ranked_and_impact_vs_polish(self) -> None:
        report = build_axis_decomposition_report(self.rows())
        axes = report["axes"]
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0]["axis"], "conventional_acceptance_strength")
        self.assertEqual(axes[-1]["axis"], "benchmark_or_dataset_value")

        ivp = report["impact_vs_polish"]
        self.assertEqual(ivp["conventional_acceptance_strength_tau_b_vs_tier"], 0.9129)
        self.assertEqual(ivp["executive_ac_priority_tau_b_vs_tier"], 0.5477)
        self.assertEqual(ivp["delta_conventional_minus_executive"], 0.3652)


if __name__ == "__main__":
    unittest.main()
