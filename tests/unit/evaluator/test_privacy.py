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


class TestPrivacySampling:
    """Tests for PrivacyEvaluator sample_size feature (PR#18)."""

    def test_evaluator_with_sample_size_limits_data(self):
        """Verify that sample_size limits the data used for calculation."""
        np.random.seed(42)
        # Create large dataset
        real = np.random.randn(1000, 5)
        synthetic = np.random.randn(500, 5)

        # Without sampling
        evaluator_full = PrivacyEvaluator(sample_size=None)
        # With sampling
        evaluator_sampled = PrivacyEvaluator(sample_size=100)

        # Both should return valid results
        risk_full = evaluator_full.reidentification_risk(real, synthetic)
        risk_sampled = evaluator_sampled.reidentification_risk(real, synthetic)

        assert 0.0 <= risk_full <= 1.0
        assert 0.0 <= risk_sampled <= 1.0
        # Results may differ due to sampling, but both should be reasonable
        assert abs(risk_full - risk_sampled) < 0.5  # Allow some variance

    def test_sample_size_smaller_than_data_uses_subset(self):
        """Verify sampling is applied when sample_size < data size."""
        np.random.seed(123)
        real = np.random.randn(500, 3)
        synthetic = np.random.randn(500, 3)

        evaluator = PrivacyEvaluator(sample_size=50)

        # Should not raise and should complete (performance test implicitly)
        dcr = evaluator.distance_to_closest_record(real, synthetic)

        # DCR should have length of sampled synthetic data (50)
        assert len(dcr) == 50

    def test_sample_size_larger_than_data_uses_all(self):
        """Verify no error when sample_size > data size."""
        real = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        synthetic = np.array([[1.1, 2.1], [10, 11]], dtype=float)

        # sample_size larger than data
        evaluator = PrivacyEvaluator(sample_size=1000)
        distances = evaluator.distance_to_closest_record(real, synthetic)

        # Should use all data (2 synthetic rows)
        assert distances.shape == (2,)

    def test_sample_size_none_uses_all_data(self):
        """Verify None sample_size uses full dataset."""
        real = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        synthetic = np.array([[1.1, 2.1]], dtype=float)

        evaluator = PrivacyEvaluator(sample_size=None)
        distances = evaluator.distance_to_closest_record(real, synthetic)

        assert distances.shape == (1,)

    def test_sample_size_reproducibility_with_seed(self):
        """Verify sampling is reproducible when numpy seed is set."""
        real = np.random.randn(100, 3)
        synthetic = np.random.randn(100, 3)

        evaluator = PrivacyEvaluator(sample_size=20)

        # Set seed before first call
        np.random.seed(999)
        dcr1 = evaluator.distance_to_closest_record(real.copy(), synthetic.copy())

        # Set same seed before second call
        np.random.seed(999)
        dcr2 = evaluator.distance_to_closest_record(real.copy(), synthetic.copy())

        np.testing.assert_array_equal(dcr1, dcr2)
