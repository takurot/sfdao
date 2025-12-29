import pandas as pd

from sfdao.scenario.engine import ScenarioEngine
from sfdao.scenario.models import ScenarioConfig, TransformationConfig


def test_scenario_engine_apply():
    df = pd.DataFrame({"amount": [100.0, 200.0, 300.0], "category": ["A", "B", "C"]})

    config = ScenarioConfig(
        name="Test Scenario",
        transformations=[
            TransformationConfig(column="amount", type="scale", params={"factor": 2.0}),
            TransformationConfig(
                column="category", type="replace", params={"old": "A", "new": "Z"}
            ),
        ],
    )

    engine = ScenarioEngine(config)
    result_df, metadata = engine.apply(df)

    # Check result
    assert result_df["amount"].tolist() == [200.0, 400.0, 600.0]
    assert result_df["category"].tolist() == ["Z", "B", "C"]

    # Check metadata
    assert metadata["scenario"]["name"] == "Test Scenario"
    assert len(metadata["scenario"]["applied"]) == 2
    assert metadata["scenario"]["applied"][0]["type"] == "scale"
    assert metadata["scenario"]["applied"][0]["column"] == "amount"
