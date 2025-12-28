from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from sfdao.reporter.base import BaseReporter, EvaluationReport

__all__ = ["HTMLReporter"]


class HTMLReporter(BaseReporter):
    """Generate an HTML audit report using Jinja2 templates."""

    def __init__(
        self, *, template_name: str = "report.html", template_dir: Path | None = None
    ) -> None:
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self._template_name = template_name
        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate(self, evaluation_report: EvaluationReport) -> str:
        context = self.build_context(evaluation_report)
        template = self._env.get_template(self._template_name)
        return template.render(self._build_template_context(context))

    def _build_template_context(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "metrics": context["metrics"],
            "composite_score": context["composite_score"],
            "metadata": context["metadata"],
        }
