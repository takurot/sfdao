import numpy as np
import pytest

from sfdao.evaluator.financial_facts import FinancialFactsChecker


def simulate_garch_process(
    n: int,
    *,
    omega: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = np.zeros(n)
    variances = np.zeros(n)

    variances[0] = omega / (1 - alpha - beta)
    shocks[0] = rng.normal(0, np.sqrt(variances[0]))

    for t in range(1, n):
        variances[t] = omega + alpha * shocks[t - 1] ** 2 + beta * variances[t - 1]
        shocks[t] = rng.normal(0, np.sqrt(max(variances[t], 0)))

    return shocks


def test_fat_tail_check():
    checker = FinancialFactsChecker()
    rng = np.random.default_rng(0)

    normal = rng.normal(0, 1, 10_000)
    t_dist = rng.standard_t(df=3, size=10_000)

    normal_result = checker.check_fat_tail(normal)
    t_result = checker.check_fat_tail(t_dist)

    assert normal_result.sample_size == 10_000
    assert pytest.approx(0, abs=0.5) == normal_result.excess_kurtosis
    assert t_result.excess_kurtosis > normal_result.excess_kurtosis
    assert t_result.excess_kurtosis > 3.0


def test_volatility_clustering_detects_garch_process():
    checker = FinancialFactsChecker()
    returns = simulate_garch_process(1500)

    result = checker.check_volatility_clustering(returns, lags=10)

    assert result.ljung_box_p_value < 0.05
    assert result.arch_test_p_value < 0.05


def test_volatility_clustering_not_flagged_for_iid_returns():
    checker = FinancialFactsChecker()
    rng = np.random.default_rng(24)
    returns = rng.normal(0, 1, 1500)

    result = checker.check_volatility_clustering(returns, lags=10)

    assert result.ljung_box_p_value > 0.05
    assert result.arch_test_p_value > 0.05
