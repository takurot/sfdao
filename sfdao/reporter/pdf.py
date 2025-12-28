from __future__ import annotations

from typing import cast

from sfdao.reporter.base import BaseReporter, EvaluationReport
from sfdao.reporter.html import HTMLReporter

try:  # pragma: no cover - exercised via tests that skip if unavailable
    from weasyprint import HTML  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - fall back when optional dependency is missing
    HTML = None

__all__ = ["PDFReporter"]


class PDFReporter(BaseReporter):
    """Generate a PDF audit report using WeasyPrint."""

    def __init__(self, html_reporter: HTMLReporter | None = None) -> None:
        self._html_reporter = html_reporter or HTMLReporter()

    def generate(self, evaluation_report: EvaluationReport) -> bytes:
        html = self._html_reporter.generate(evaluation_report)
        return self._render_pdf(html)

    def _render_pdf(self, html: str) -> bytes:
        if HTML is None:
            raise RuntimeError(
                "WeasyPrint is not available. Install weasyprint to enable PDF output."
            )

        try:
            return cast(bytes, HTML(string=html).write_pdf())
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PDF generation failed. Ensure WeasyPrint system dependencies are installed."
            ) from exc
