from pathlib import Path

import numpy as np

from sfdao.evaluator.privacy import PrivacyEvaluator
from sfdao.evaluator.scoring import CompositeScorer, ScoreConstraint
from sfdao.evaluator.statistical import StatisticalEvaluator
from sfdao.reporter.base import EvaluationReport, PlainTextReporter


def test_scoring_and_reporter_pipeline(tmp_path: Path):
    real = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    synthetic = np.array([1.0, 2.1, 2.9, 4.2, 5.1], dtype=float)

    stats_eval = StatisticalEvaluator()
    ks_result = stats_eval.ks_test(real, synthetic)
    js_div = stats_eval.js_divergence(real, synthetic, bins=5)

    quality_score = max(0.0, min(1.0, 1.0 - ks_result.statistic))
    utility_score = max(0.0, min(1.0, 1.0 - js_div))

    privacy_eval = PrivacyEvaluator()
    real_matrix = np.column_stack((real, real * 0.5))
    synthetic_matrix = np.column_stack((synthetic, synthetic * 0.5 + 0.1))
    privacy_risk = privacy_eval.reidentification_risk(real_matrix, synthetic_matrix)
    privacy_score = max(0.0, min(1.0, 1.0 - privacy_risk))

    metrics = {
        "quality": quality_score,
        "utility": utility_score,
        "privacy": privacy_score,
    }

    scorer = CompositeScorer(
        {"quality": 0.4, "utility": 0.3, "privacy": 0.3},
        constraints=[ScoreConstraint(metric="privacy", minimum=0.5, penalty=0.1)],
    )
    composite = scorer.calculate(metrics)

    reporter = PlainTextReporter()
    evaluation_report = EvaluationReport(
        metrics=metrics, composite_score=composite, metadata={"source": "smoke"}
    )
    output_path = reporter.render_to_file(evaluation_report, tmp_path / "score_report.txt")

    content = output_path.read_text()
    assert composite.total > 0
    assert "Overall Score" in content
    assert "quality" in content
    assert "privacy" in content
