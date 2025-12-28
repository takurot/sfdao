from pathlib import Path

from sfdao.evaluator.scoring import CompositeScorer
from sfdao.reporter.base import EvaluationReport
from sfdao.reporter.html import HTMLReporter


def _build_report() -> EvaluationReport:
    metrics = {"quality": 0.82, "utility": 0.74, "privacy": 0.91}
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
    scorer = CompositeScorer(weights)
    composite = scorer.calculate(metrics)
    metadata = {"real_rows": 120, "synthetic_rows": 100, "privacy_risk": 0.12}
    return EvaluationReport(metrics=metrics, composite_score=composite, metadata=metadata)


def test_generate_html_report(tmp_path: Path) -> None:
    report = _build_report()
    reporter = HTMLReporter()

    html = reporter.generate(report)

    assert "<html" in html.lower()
    assert "Overall Score" in html
    assert "quality" in html

    output_path = reporter.render_to_file(report, tmp_path / "report.html")
    assert output_path.exists()
    assert "Overall Score" in output_path.read_text()
