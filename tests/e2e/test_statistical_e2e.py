import numpy as np

from sfdao.evaluator.statistical import StatisticalEvaluator


def test_statistical_e2e_smoke():
    """Smoke test: StatisticalEvaluator on similar vs distant distributions."""
    evaluator = StatisticalEvaluator()
    rng = np.random.default_rng(2024)

    real = rng.normal(0, 1, 500)
    similar = rng.normal(0, 1, 500)
    distant = rng.normal(4, 1, 500)

    similar_ks = evaluator.ks_test(real, similar)
    distant_ks = evaluator.ks_test(real, distant)

    assert similar_ks.statistic < 0.1
    assert similar_ks.p_value > 0.05
    assert distant_ks.statistic > 0.2
    assert distant_ks.p_value < 1e-4

    similar_js = evaluator.js_divergence(real, similar, bins=40)
    distant_js = evaluator.js_divergence(real, distant, bins=40)

    assert 0 <= similar_js <= 1
    assert 0 <= distant_js <= 1
    assert similar_js < 0.05
    assert distant_js > 0.2
