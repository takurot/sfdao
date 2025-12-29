import numpy as np
import pandas as pd

from sfdao.scenario.transformations import TransformationRegistry


def test_scale_transformation():
    s = pd.Series([10.0, 20.0, 30.0])
    # Assuming registry returns a callable that takes (series, params, rng)
    func = TransformationRegistry.get("scale")
    result = func(s, {"factor": 1.5})
    assert result.iloc[0] == 15.0
    assert result.iloc[1] == 30.0


def test_shift_transformation():
    s = pd.Series([10.0, 20.0])
    func = TransformationRegistry.get("shift")
    result = func(s, {"value": 5.0})
    assert result.iloc[0] == 15.0


def test_clip_transformation():
    s = pd.Series([10, 50, 90])
    func = TransformationRegistry.get("clip")
    result = func(s, {"min": 20, "max": 80})
    assert result.iloc[0] == 20
    assert result.iloc[1] == 50
    assert result.iloc[2] == 80


def test_replace_transformation():
    s = pd.Series(["A", "B", "A"])
    func = TransformationRegistry.get("replace")
    result = func(s, {"old": "A", "new": "C"})
    assert result.iloc[0] == "C"
    assert result.iloc[1] == "B"


def test_outlier_transformation():
    s = pd.Series([10.0] * 100)
    rng = np.random.default_rng(42)
    func = TransformationRegistry.get("outlier")
    # Replace 2 values with 999
    result = func(s, {"n": 2, "value": 999}, rng=rng)
    assert (result == 999).sum() == 2
    assert (result == 10.0).sum() == 98
