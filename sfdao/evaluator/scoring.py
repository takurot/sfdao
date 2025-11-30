from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

__all__ = [
    "ScoreComponent",
    "ScorePenalty",
    "ScoreConstraint",
    "CompositeScore",
    "CompositeScorer",
]


@dataclass(frozen=True)
class ScoreComponent:
    metric: str
    raw_value: float
    weight: float
    weighted_value: float


@dataclass(frozen=True)
class ScorePenalty:
    metric: str
    amount: float
    reason: str | None = None


@dataclass(frozen=True)
class ScoreConstraint:
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    penalty: float = 0.0
    description: str | None = None


@dataclass(frozen=True)
class CompositeScore:
    total: float
    components: Mapping[str, ScoreComponent] = field(default_factory=dict)
    penalties: list[ScorePenalty] = field(default_factory=list)


class CompositeScorer:
    """Combine individual metric scores into a composite score with constraints."""

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        constraints: Sequence[ScoreConstraint] | None = None,
    ) -> None:
        if not weights:
            raise ValueError("At least one weight must be provided.")

        self._weights = dict(weights)
        self._constraints = list(constraints or [])
        self._normalization_factor = self._compute_normalization_factor()

    def calculate(
        self,
        metrics: Mapping[str, float],
        *,
        constraints: Sequence[ScoreConstraint] | None = None,
    ) -> CompositeScore:
        self._validate_metrics(metrics)
        normalized_weights = self._normalized_weights()

        components: Dict[str, ScoreComponent] = {}
        weighted_total = 0.0

        for metric_name, weight in normalized_weights.items():
            raw_value = float(metrics[metric_name])
            clamped_value = self._clamp(raw_value, 0.0, 1.0)
            weighted_value = clamped_value * weight
            components[metric_name] = ScoreComponent(
                metric=metric_name,
                raw_value=clamped_value,
                weight=weight,
                weighted_value=weighted_value,
            )
            weighted_total += weighted_value

        applied_penalties = self._evaluate_constraints(metrics, constraints)
        penalty_total = sum(p.amount for p in applied_penalties)
        total_score = self._clamp(weighted_total - penalty_total, 0.0, 1.0)

        return CompositeScore(total=total_score, components=components, penalties=applied_penalties)

    def _normalized_weights(self) -> Dict[str, float]:
        return {metric: weight / self._normalization_factor for metric, weight in self._weights.items()}

    def _compute_normalization_factor(self) -> float:
        factor = sum(self._weights.values())
        if factor <= 0:
            raise ValueError("Weights must sum to a positive value.")
        return factor

    def _validate_metrics(self, metrics: Mapping[str, float]) -> None:
        missing = [metric for metric in self._weights if metric not in metrics]
        if missing:
            raise ValueError(f"Missing metrics for weights: {', '.join(missing)}")

    def _evaluate_constraints(
        self,
        metrics: Mapping[str, float],
        extra_constraints: Sequence[ScoreConstraint] | None,
    ) -> list[ScorePenalty]:
        penalties: list[ScorePenalty] = []
        all_constraints = [*self._constraints, *(extra_constraints or [])]

        for constraint in all_constraints:
            if constraint.metric not in metrics:
                continue

            value = float(metrics[constraint.metric])
            violations: list[str] = []

            if constraint.minimum is not None and value < constraint.minimum:
                violations.append(f"value {value:.3f} below minimum {constraint.minimum:.3f}")
            if constraint.maximum is not None and value > constraint.maximum:
                violations.append(f"value {value:.3f} above maximum {constraint.maximum:.3f}")

            if violations and constraint.penalty > 0:
                reason = constraint.description or "; ".join(violations)
                penalties.append(
                    ScorePenalty(
                        metric=constraint.metric,
                        amount=float(constraint.penalty),
                        reason=reason,
                    )
                )
        return penalties

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
