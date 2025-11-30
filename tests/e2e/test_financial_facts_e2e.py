import json
from dataclasses import asdict

import numpy as np

from sfdao.evaluator.financial_facts import FinancialFactsChecker


def simulate_garch_process(
    n: int,
    *,
    omega: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.8,
    seed: int = 2024,
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


def test_financial_facts_e2e_smoke(tmp_path):
    """End-to-end style check: detect clustering vs iid and write results."""

    checker = FinancialFactsChecker()

    garch_returns = simulate_garch_process(1500)
    iid_returns = np.random.default_rng(7).normal(0, 1, 1500)

    garch_result = checker.check_volatility_clustering(garch_returns, lags=10)
    iid_result = checker.check_volatility_clustering(iid_returns, lags=10)

    summary = {
        "garch": asdict(garch_result),
        "iid": asdict(iid_result),
    }

    output_file = tmp_path / "financial_facts_result.json"
    output_file.write_text(json.dumps(summary, indent=2))

    assert output_file.exists()
    assert json.loads(output_file.read_text()) == summary

    assert garch_result.ljung_box_p_value < 0.05
    assert garch_result.arch_test_p_value < 0.05
    assert iid_result.ljung_box_p_value > 0.05
    assert iid_result.arch_test_p_value > 0.05
