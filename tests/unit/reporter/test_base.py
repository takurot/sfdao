from pathlib import Path

import pytest

from sfdao.evaluator.scoring import CompositeScorer, ScoreConstraint
from sfdao.reporter.base import BaseReporter, EvaluationReport


class DummyReporter(BaseReporter):
    def generate(self, evaluation_report: EvaluationReport) -> str:  # pragma: no cover - thin wrapper
        context = self.build_context(evaluation_report)
        return f"Overall Score: {context['composite_score']['total']:.3f}"


def test_base_reporter_builds_context_and_saves(tmp_path: Path):
    metrics = {"quality": 0.78, "utility": 0.66, "privacy": 0.82}
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
    constraints = [ScoreConstraint(metric="privacy", minimum=0.7, penalty=0.05)]

    scorer = CompositeScorer(weights, constraints=constraints)
    composite = scorer.calculate(metrics)
    report = EvaluationReport(metrics=metrics, composite_score=composite, metadata={"title": "demo"})

    reporter = DummyReporter()
    context = reporter.build_context(report)

    assert context["metrics"] == metrics
    assert "components" in context["composite_score"]
    assert context["composite_score"]["total"] == pytest.approx(composite.total)
    assert context["metadata"]["title"] == "demo"

    output_path = reporter.render_to_file(report, tmp_path / "score.txt")
    assert output_path.exists()
    assert "Overall Score" in output_path.read_text()
