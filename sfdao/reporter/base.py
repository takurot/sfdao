from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sfdao.evaluator.scoring import CompositeScore

__all__ = [
    "BaseReporter",
    "EvaluationReport",
    "PlainTextReporter",
]


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, float]
    composite_score: CompositeScore
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseReporter(ABC):
    """Base class for reporters that generate evaluation artifacts."""

    def build_context(self, evaluation_report: EvaluationReport) -> dict[str, Any]:
        composite = evaluation_report.composite_score
        component_map = {
            name: {
                "value": component.raw_value,
                "weight": component.weight,
                "weighted": component.weighted_value,
            }
            for name, component in composite.components.items()
        }
        penalties = [
            {"metric": penalty.metric, "amount": penalty.amount, "reason": penalty.reason}
            for penalty in composite.penalties
        ]

        return {
            "metrics": dict(evaluation_report.metrics),
            "composite_score": {
                "total": composite.total,
                "components": component_map,
                "penalties": penalties,
            },
            "metadata": dict(evaluation_report.metadata),
        }

    def render_to_file(self, evaluation_report: EvaluationReport, path: str | Path) -> Path:
        content = self.generate(evaluation_report)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @abstractmethod
    def generate(self, evaluation_report: EvaluationReport) -> str:
        raise NotImplementedError


class PlainTextReporter(BaseReporter):
    """Minimal reporter that emits a plain-text summary for smoke testing."""

    def generate(
        self, evaluation_report: EvaluationReport
    ) -> str:  # pragma: no cover - exercised via tests
        context = self.build_context(evaluation_report)
        composite = context["composite_score"]

        lines = [f"Overall Score: {composite['total']:.3f}", "", "Metrics:"]

        for name, values in composite["components"].items():
            lines.append(
                f"- {name}: value={values['value']:.3f}, "
                f"weight={values['weight']:.3f}, "
                f"weighted={values['weighted']:.3f}"
            )

        if composite["penalties"]:
            lines.append("")
            lines.append("Penalties:")
            for penalty in composite["penalties"]:
                reason = f" ({penalty['reason']})" if penalty["reason"] else ""
                lines.append(f"- {penalty['metric']}: -{penalty['amount']:.3f}{reason}")

        if context["metadata"]:
            lines.append("")
            lines.append("Metadata:")
            for key, value in sorted(context["metadata"].items()):
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) + "\n"
