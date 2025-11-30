import numpy as np

from sfdao.evaluator.privacy import PrivacyEvaluator
from sfdao.ingestion.loader import CSVLoader


def test_privacy_ingestion_flow(tmp_path):
    """Smoke: load CSV via ingestion, compute privacy metrics."""

    csv_path = tmp_path / "numeric_data.csv"
    csv_path.write_text("amount,balance\n100,1000\n105,999\n110,998\n95,1002\n")

    loader = CSVLoader()
    df = loader.load(csv_path)

    real = df[["amount", "balance"]].to_numpy(dtype=float)
    synthetic = real + np.array([[-1, 2], [2, -1], [0.5, -0.5], [-3, 3]])

    evaluator = PrivacyEvaluator()
    dcr = evaluator.distance_to_closest_record(real, synthetic)
    risk = evaluator.reidentification_risk(real, synthetic)

    assert dcr.shape == (4,)
    assert dcr.min() < dcr.max()
    assert 0 <= risk <= 1
    assert risk > 0.2
