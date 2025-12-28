import pytest

from sfdao.evaluator.scoring import CompositeScorer
from sfdao.reporter.base import EvaluationReport
from sfdao.reporter.pdf import PDFReporter


def _build_report() -> EvaluationReport:
    metrics = {"quality": 0.63, "utility": 0.7, "privacy": 0.88}
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
    scorer = CompositeScorer(weights)
    composite = scorer.calculate(metrics)
    metadata = {"real_rows": 90, "synthetic_rows": 90, "privacy_risk": 0.2}
    return EvaluationReport(metrics=metrics, composite_score=composite, metadata=metadata)


def test_pdf_reporter_generates_or_skips() -> None:
    report = _build_report()
    reporter = PDFReporter()

    try:
        pdf_bytes = reporter.generate(report)
    except RuntimeError:
        pytest.skip("WeasyPrint is not available in this environment.")

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
