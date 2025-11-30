import json

import numpy as np

from sfdao.evaluator.privacy import PrivacyEvaluator


def test_privacy_e2e_smoke(tmp_path):
    """Smoke test: PrivacyEvaluator distinguishes near vs distant synthetic records."""

    evaluator = PrivacyEvaluator()
    rng = np.random.default_rng(2025)

    real = rng.normal(0, 1, size=(120, 3))
    synthetic_close = real[:60] + rng.normal(0, 0.05, size=(60, 3))
    synthetic_far = rng.normal(4, 1, size=(60, 3))

    close_dcr = evaluator.distance_to_closest_record(real, synthetic_close)
    far_dcr = evaluator.distance_to_closest_record(real, synthetic_far)

    close_risk = evaluator.reidentification_risk(real, synthetic_close)
    far_risk = evaluator.reidentification_risk(real, synthetic_far)

    summary = {
        "close": {"median_dcr": float(np.median(close_dcr)), "risk": close_risk},
        "far": {"median_dcr": float(np.median(far_dcr)), "risk": far_risk},
    }

    output_file = tmp_path / "privacy_results.json"
    output_file.write_text(json.dumps(summary, indent=2))

    assert output_file.exists()
    assert json.loads(output_file.read_text()) == summary

    assert summary["close"]["median_dcr"] < summary["far"]["median_dcr"]
    assert close_risk > far_risk
    assert close_risk > 0.4
    assert far_risk < 0.3
