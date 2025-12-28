# API Reference (Python)

SFDAOはCLIだけでなく、Pythonから各モジュールを直接呼び出せます。

## 監査（高レベルAPI）

```python
from pathlib import Path
from rich.console import Console

from sfdao.cli.audit import run_audit

run_audit(
    real_path=Path("real.csv"),
    synthetic_path=Path("synthetic.csv"),
    output_path=Path("report.html"),
    quiet=True,
    console=Console(),
)
```

## 個別モジュール

### Ingestion

```python
from sfdao.ingestion.loader import CSVLoader

df = CSVLoader().load("real.csv")
```

### Evaluators

```python
from sfdao.evaluator.statistical import StatisticalEvaluator

evaluator = StatisticalEvaluator()
ks = evaluator.ks_test([1, 2, 3], [1.1, 2.0, 2.9])
js = evaluator.js_divergence([1, 2, 3], [1.1, 2.0, 2.9])
```

### Reporting

```python
from sfdao.evaluator.scoring import CompositeScorer
from sfdao.reporter.base import EvaluationReport, PlainTextReporter

reporter = PlainTextReporter()
metrics = {"quality": 0.8, "utility": 0.7, "privacy": 0.9}
composite = CompositeScorer({"quality": 0.4, "utility": 0.3, "privacy": 0.3}).calculate(metrics)

text = reporter.generate(EvaluationReport(metrics=metrics, composite_score=composite, metadata={}))
```
