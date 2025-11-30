import json
from dataclasses import asdict

import numpy as np

from sfdao.evaluator.financial_facts import FinancialFactsChecker
from sfdao.evaluator.privacy import PrivacyEvaluator
from sfdao.evaluator.statistical import StatisticalEvaluator


def test_evaluator_integration_smoke(tmp_path):
    """Smoke: run all evaluators and persist a combined summary."""

    rng = np.random.default_rng(21)
    real = rng.normal(0, 1, size=500)
    similar = rng.normal(0, 1, size=500)
    returns = rng.normal(0, 1, size=1500)
    real_matrix = rng.normal(0, 1, size=(100, 4))
    synthetic_matrix = real_matrix[:50] + rng.normal(0, 0.05, size=(50, 4))

    statistical = StatisticalEvaluator()
    financial = FinancialFactsChecker()
    privacy = PrivacyEvaluator()

    ks = statistical.ks_test(real, similar)
    js = statistical.js_divergence(real, similar, bins=30)
    fat_tail = financial.check_fat_tail(returns)
    clustering = financial.check_volatility_clustering(returns, lags=8)
    dcr = privacy.distance_to_closest_record(real_matrix, synthetic_matrix)
    risk = privacy.reidentification_risk(real_matrix, synthetic_matrix)

    summary = {
        "statistical": {"ks": asdict(ks), "js_divergence": js},
        "financial": {
            "fat_tail": asdict(fat_tail),
            "volatility_clustering": asdict(clustering),
        },
        "privacy": {"median_dcr": float(np.median(dcr)), "risk": risk},
    }

    out_file = tmp_path / "evaluator_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))

    assert out_file.exists()
    loaded = json.loads(out_file.read_text())
    assert loaded["statistical"]["ks"]["p_value"] > 0
    assert 0 <= loaded["statistical"]["js_divergence"] <= 1
    assert loaded["financial"]["fat_tail"]["sample_size"] == 1500
    assert loaded["privacy"]["median_dcr"] > 0
    assert 0 <= loaded["privacy"]["risk"] <= 1
