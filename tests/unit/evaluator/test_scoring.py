import pytest

from sfdao.evaluator.scoring import CompositeScorer, ScoreConstraint


def test_composite_score_calculation():
    metrics = {
        "quality": 0.8,
        "utility": 0.7,
        "privacy": 0.9,
    }
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}

    scorer = CompositeScorer(weights)
    score = scorer.calculate(metrics)

    expected = 0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3
    assert pytest.approx(expected, rel=1e-6) == score.total
    assert pytest.approx(0.8 * 0.4, rel=1e-6) == score.components["quality"].weighted_value
    assert score.penalties == []


def test_weight_changes_adjust_total_score():
    metrics = {"quality": 0.9, "utility": 0.4, "privacy": 0.7}

    scorer_quality_heavy = CompositeScorer({"quality": 0.6, "utility": 0.2, "privacy": 0.2})
    scorer_privacy_heavy = CompositeScorer({"quality": 0.2, "utility": 0.2, "privacy": 0.6})

    quality_score = scorer_quality_heavy.calculate(metrics)
    privacy_score = scorer_privacy_heavy.calculate(metrics)

    assert quality_score.total != privacy_score.total
    assert quality_score.total > privacy_score.total  # quality weight is larger


def test_applies_constraint_penalty_when_below_threshold():
    metrics = {"quality": 0.9, "utility": 0.8, "privacy": 0.4}
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
    constraints = [
        ScoreConstraint(metric="privacy", minimum=0.7, penalty=0.2, description="privacy floor")
    ]

    scorer = CompositeScorer(weights, constraints=constraints)
    score = scorer.calculate(metrics)

    base_score = 0.9 * 0.4 + 0.8 * 0.3 + 0.4 * 0.3
    assert pytest.approx(max(base_score - 0.2, 0.0), rel=1e-6) == score.total
    assert len(score.penalties) == 1
    assert score.penalties[0].metric == "privacy"
    assert "privacy floor" in score.penalties[0].reason


def test_zero_and_perfect_scores_are_bounded():
    weights = {"quality": 0.5, "utility": 0.25, "privacy": 0.25}
    scorer = CompositeScorer(weights)

    zero = scorer.calculate({"quality": 0.0, "utility": 0.0, "privacy": 0.0})
    perfect = scorer.calculate({"quality": 1.0, "utility": 1.0, "privacy": 1.0})

    assert zero.total == 0.0
    assert perfect.total == 1.0
