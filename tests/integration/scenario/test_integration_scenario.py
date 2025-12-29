import pandas as pd

from sfdao.config.models import GeneratorSettings, ScenarioSettings
from sfdao.generator.factory import build_generator
from sfdao.scenario.loader import load_scenario_engine


def test_integration_scenario_injection():
    # 1. Setup Data
    real = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
        }
    )

    # 2. Setup Settings
    gen_settings = GeneratorSettings(type="baseline", n_samples=10, params={})

    # Inject a scenario that shifts amount by +1000
    scenario_settings = ScenarioSettings(
        enabled=True,
        params={
            "name": "Integration Test",
            "transformations": [{"column": "amount", "type": "shift", "params": {"value": 1000.0}}],
        },
    )

    # 3. Load Components
    scenario_engine = load_scenario_engine(scenario_settings)
    generator = build_generator(gen_settings, seed=42, scenario=scenario_engine)

    # 4. Fit & Generate
    generator.fit(real)
    synthetic = generator.sample(10)

    # 5. Verify
    # Baseline generator should produce values around mean(amount) which is 200.
    # With shift +1000, values should be around 1200.
    assert synthetic["amount"].mean() > 1000.0
