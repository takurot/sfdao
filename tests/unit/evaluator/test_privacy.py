import numpy as np
import pytest

from sfdao.evaluator.privacy import PrivacyEvaluator


def test_distance_to_closest_record_orders_by_proximity():
    real = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    synthetic = np.array([[1.1, 2.1], [10, 11]], dtype=float)

    evaluator = PrivacyEvaluator()
    distances = evaluator.distance_to_closest_record(real, synthetic)

    assert distances.shape == (2,)
    assert pytest.approx(0.1414, rel=1e-3) == distances[0]
    assert distances[0] < distances[1]
    assert pytest.approx(np.sqrt(50), rel=1e-3) == distances[1]


def test_reidentification_risk_drops_for_distant_records():
    real = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    near = np.array([[0.05, 0.05], [0.95, 1.05]])
    far = np.array([[3.0, 3.0], [4.0, 4.0]])

    evaluator = PrivacyEvaluator()

    near_risk = evaluator.reidentification_risk(real, near)
    far_risk = evaluator.reidentification_risk(real, far)

    assert 0.0 <= near_risk <= 1.0
    assert 0.0 <= far_risk <= 1.0
    assert near_risk > far_risk
    assert near_risk > 0.5
    assert far_risk < 0.2
