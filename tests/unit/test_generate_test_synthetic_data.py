import pandas as pd

from sfdao.scripts.generate_test_synthetic_data import generate_simple_synthetic


def test_generate_simple_synthetic_creates_file_and_columns(tmp_path):
    real_csv = "tests/fixtures/creditcard_real_sample.csv"
    output_path = tmp_path / "synthetic.csv"

    generate_simple_synthetic(real_csv, output_path, n_samples=50, random_state=42)

    assert output_path.exists()
    synthetic_df = pd.read_csv(output_path)

    real_df = pd.read_csv(real_csv)
    assert list(synthetic_df.columns) == list(real_df.columns)
    assert len(synthetic_df) == 50
    assert set(synthetic_df["Class"].unique()).issubset({0, 1})


def test_generated_data_preserves_basic_stats(tmp_path):
    real_csv = "tests/fixtures/creditcard_real_sample.csv"
    output_path = tmp_path / "synthetic.csv"

    generate_simple_synthetic(real_csv, output_path, n_samples=200, random_state=123)

    synthetic_df = pd.read_csv(output_path)
    real_df = pd.read_csv(real_csv)

    numeric_cols = [col for col in real_df.columns if col != "Class"]
    for col in numeric_cols:
        real_mean = real_df[col].mean()
        synth_mean = synthetic_df[col].mean()
        assert abs(synth_mean - real_mean) < abs(real_mean) * 0.2 + 0.1

        real_std = real_df[col].std()
        synth_std = synthetic_df[col].std()
        assert abs(synth_std - real_std) < abs(real_std) * 0.2 + 0.1

    real_class_counts = real_df["Class"].value_counts(normalize=True)
    synth_class_counts = synthetic_df["Class"].value_counts(normalize=True)
    assert set(synthetic_df["Class"].unique()) == set(real_df["Class"].unique())
    for label, real_ratio in real_class_counts.items():
        synth_ratio = synth_class_counts[label]
        assert abs(synth_ratio - real_ratio) < 0.1
