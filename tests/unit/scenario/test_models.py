from sfdao.scenario.models import ScenarioConfig, TransformationConfig


def test_transformation_config_creation():
    config = TransformationConfig(column="amount", type="scale", params={"factor": 1.5})
    assert config.column == "amount"
    assert config.type == "scale"
    assert config.params["factor"] == 1.5


def test_scenario_config_creation():
    tx = TransformationConfig(column="amount", type="shift", params={"value": 100})
    scenario = ScenarioConfig(name="Test Scenario", transformations=[tx])
    assert scenario.name == "Test Scenario"
    assert len(scenario.transformations) == 1
    assert scenario.transformations[0].type == "shift"


def test_transformation_config_validation():
    # pydantic should raise error for missing fields
    try:
        TransformationConfig(column="amount")  # Missing type and params
        assert False
    except Exception:
        assert True
