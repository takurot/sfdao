# SFDAO Phase 1 実装計画書

**作成日**: 2025-11-29
**対象フェーズ**: Phase 1 - "The Auditor" (MVP)
**開発手法**: TDD (Test-Driven Development)

---

## 0. 開発原則

### TDD (テスト駆動開発) の実践

すべてのPRにおいて、以下のTDDサイクルを厳守します：

1. **Red**: まず失敗するテストを書く
2. **Green**: テストをパスする最小限のコードを書く
3. **Refactor**: コードを整理・最適化する

### ブランチ戦略

**重要**: 各PRの作業は必ず専用のブランチを作成して実行してください。直接mainブランチで作業しないでください。

```bash
# PR#1の場合の例
git checkout -b feature/pr-01-project-setup

# 作業を実施...

# コミット
git add .
git commit -m "PR#1: プロジェクト構造とCI/CD設定

- pyproject.tomlの作成
- CI/CD設定
- テスト実行"

# プッシュ
git push -u origin feature/pr-01-project-setup

# GitHub上でPR作成
gh pr create --title "PR#1: プロジェクト構造とCI/CD設定" --body "..."

# PRマージ後、mainブランチに戻る
git checkout main
git pull origin main

# 作業ブランチの削除（オプション）
git branch -d feature/pr-01-project-setup
```

**ブランチ命名規則**:
- `feature/pr-XX-short-description`: 機能追加PR（例: feature/pr-02-data-ingestion）
- `fix/issue-XX-bug-description`: バグ修正
- `docs/description`: ドキュメント更新のみ

### テスト戦略

- **単体テスト (Unit Tests)**: 各モジュール・関数レベルでのテスト (カバレッジ目標: 90%以上)
- **統合テスト (Integration Tests)**: モジュール間の連携テスト
- **End-to-End テスト**: CLI実行からレポート生成までの完全フロー
- **性能テスト**: 大規模データセット（100万行以上）での動作確認

### コード品質

- **Linting**: Black (フォーマッタ), Flake8 (リンター)
- **型チェック**: mypy (strict mode)
- **セキュリティチェック**: bandit
- **依存関係管理**: Poetry

### 開発環境（macOS）

このプロジェクトはmacOS環境での開発を前提としています。

**仮想環境のセットアップ**:

```bash
# Poetryのインストール（初回のみ）
curl -sSL https://install.python-poetry.org | python3 -

# PATHの設定（~/.zshrc または ~/.bash_profile に追加）
export PATH="$HOME/.local/bin:$PATH"

# 新しいターミナルセッションで確認
poetry --version

# プロジェクトディレクトリで仮想環境を作成
cd /path/to/sfdao
poetry install

# 仮想環境を有効化
poetry shell

# 仮想環境内でPythonバージョン確認
python --version  # Python 3.10以上であることを確認
```

**macOS固有の注意事項**:
- `._*` ファイル（リソースフォーク）は自動的に.gitignoreで除外されます
- `.DS_Store` ファイルも除外されます
- WeasyPrint（PDF生成）には追加の依存関係が必要な場合があります：
  ```bash
  brew install cairo pango gdk-pixbuf libffi
  ```

---

## PR#1: プロジェクト構造とCI/CD設定

**目的**: 開発環境の基盤を構築し、以降のPRで一貫した品質を保証する

### ディレクトリ構造

```
stock-data-generator/
├── pyproject.toml           # Poetry設定
├── poetry.lock
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions
├── sfdao/                   # メインパッケージ
│   ├── __init__.py
│   ├── ingestion/           # Data Ingestion
│   ├── evaluator/           # Evaluator
│   ├── reporter/            # Report Generator
│   └── cli/                 # CLI
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/            # テストデータ
└── docs/
    └── api/
```

### 主要依存関係 (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.0"
numpy = "^1.24"
scipy = "^1.11"
scikit-learn = "^1.3"
pydantic = "^2.0"           # データバリデーション
typer = "^0.9"              # CLI
rich = "^13.0"              # CLI表示
jinja2 = "^3.1"             # レポートテンプレート
weasyprint = "^60.0"        # PDF生成
matplotlib = "^3.7"
seaborn = "^0.12"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
pytest-cov = "^4.1"
black = "^23.7"
flake8 = "^6.0"
mypy = "^1.5"
bandit = "^1.7"
pre-commit = "^3.3"
```

### CI/CD設定 (.github/workflows/ci.yml)

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -

      - name: Install dependencies
        run: poetry install

      - name: Run linting
        run: |
          poetry run black --check .
          poetry run flake8 .

      - name: Run type checking
        run: poetry run mypy sfdao

      - name: Run security check
        run: poetry run bandit -r sfdao

      - name: Run tests
        run: poetry run pytest tests/ --cov=sfdao --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### テスト内容

- `tests/test_project_structure.py`: ディレクトリ構造とファイル存在確認
- CI/CDパイプラインが正常に動作することを確認

---

## PR#2: Data Ingestion基本機能

**目的**: CSVファイルを読み込み、基本的なスキーマ情報を抽出する
**進捗**: ✅ CSVLoader/SchemaExtractor 実装・単体/統合テスト完了（PR#2: Data ingestion basic functionality 作成済み）

### 実装モジュール

- `sfdao/ingestion/loader.py`: データ読み込み
- `sfdao/ingestion/schema.py`: スキーマ定義クラス

### TDDステップ

#### Test 1: CSVファイル読み込み (Red)

```python
# tests/unit/ingestion/test_loader.py
def test_load_csv_file():
    loader = CSVLoader()
    df = loader.load("tests/fixtures/sample_transactions.csv")
    assert not df.empty
    assert "transaction_id" in df.columns
```

#### Implementation (Green)

```python
# sfdao/ingestion/loader.py
import pandas as pd
from pathlib import Path

class CSVLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)
```

#### Test 2: スキーマ抽出 (Red)

```python
def test_extract_schema():
    loader = CSVLoader()
    df = loader.load("tests/fixtures/sample_transactions.csv")
    schema = SchemaExtractor.extract(df)

    assert schema.num_rows > 0
    assert schema.num_columns > 0
    assert len(schema.columns) == len(df.columns)
```

#### Implementation (Green)

```python
# sfdao/ingestion/schema.py
from pydantic import BaseModel
from typing import List, Dict

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int

class DataSchema(BaseModel):
    num_rows: int
    num_columns: int
    columns: List[ColumnInfo]

class SchemaExtractor:
    @staticmethod
    def extract(df: pd.DataFrame) -> DataSchema:
        columns = []
        for col in df.columns:
            columns.append(ColumnInfo(
                name=col,
                dtype=str(df[col].dtype),
                null_count=df[col].isnull().sum(),
                unique_count=df[col].nunique()
            ))

        return DataSchema(
            num_rows=len(df),
            num_columns=len(df.columns),
            columns=columns
        )
```

### テスト計画

#### Unit Tests

- `test_load_csv_valid_file()`: 正常なCSVファイルの読み込み
- `test_load_csv_missing_file()`: 存在しないファイルのエラーハンドリング
- `test_load_csv_malformed()`: 不正なフォーマットの処理
- `test_extract_schema_basic()`: 基本的なスキーマ抽出
- `test_extract_schema_with_nulls()`: 欠損値を含むデータ
- `test_extract_schema_empty_dataframe()`: 空のデータフレーム

#### Integration Tests

- `test_loader_schema_pipeline()`: ローダーとスキーマ抽出の統合

---

## PR#3: Auto-Type Detection機能

**目的**: カラムの型（数値、カテゴリ、日時、PII、フリーテキスト）を自動判定する
**進捗**: ✅ TypeDetector 実装・単体テスト完了（PR#3）

### 実装モジュール

- `sfdao/ingestion/type_detector.py`: 型判定ロジック

### TDDステップ

#### Test 1: 数値型の判定 (Red)

```python
# tests/unit/ingestion/test_type_detector.py
def test_detect_numeric_column():
    data = pd.Series([100, 200, 300, 400])
    detector = TypeDetector()
    col_type = detector.detect(data, "amount")
    assert col_type == ColumnType.NUMERIC
```

#### Test 2: カテゴリ型の判定 (Red)

```python
def test_detect_categorical_column():
    data = pd.Series(["A", "B", "A", "C", "B", "A"])
    detector = TypeDetector()
    col_type = detector.detect(data, "category")
    assert col_type == ColumnType.CATEGORICAL
```

#### Test 3: 日時型の判定 (Red)

```python
def test_detect_datetime_column():
    data = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
    detector = TypeDetector()
    col_type = detector.detect(data, "timestamp")
    assert col_type == ColumnType.DATETIME
```

#### Test 4: PII検出 (Red)

```python
def test_detect_pii_column():
    # メールアドレス
    data = pd.Series(["user1@example.com", "user2@test.com"])
    detector = TypeDetector()
    col_type = detector.detect(data, "email")
    assert col_type == ColumnType.PII

    # 電話番号
    data = pd.Series(["090-1234-5678", "080-9876-5432"])
    col_type = detector.detect(data, "phone")
    assert col_type == ColumnType.PII
```

#### Implementation (Green)

```python
# sfdao/ingestion/type_detector.py
from enum import Enum
import re

class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    PII = "pii"
    FREE_TEXT = "free_text"

class TypeDetector:
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\d{2,4}-\d{2,4}-\d{4}",
        "credit_card": r"\d{4}-\d{4}-\d{4}-\d{4}",
    }

    def detect(self, series: pd.Series, column_name: str) -> ColumnType:
        # PII check first (security priority)
        if self._is_pii(series):
            return ColumnType.PII

        # Datetime check
        if self._is_datetime(series):
            return ColumnType.DATETIME

        # Numeric check
        if pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERIC

        # Categorical vs Free Text
        if self._is_categorical(series):
            return ColumnType.CATEGORICAL

        return ColumnType.FREE_TEXT

    def _is_pii(self, series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(100)
        for pattern in self.PII_PATTERNS.values():
            if sample.str.match(pattern).any():
                return True
        return False

    def _is_datetime(self, series: pd.Series) -> bool:
        try:
            pd.to_datetime(series.dropna().head(10))
            return True
        except:
            return False

    def _is_categorical(self, series: pd.Series) -> bool:
        # ユニーク値が全体の5%未満ならカテゴリ
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.05
```

### テスト計画

#### Unit Tests (各型×複数パターン)

- 数値型: 整数、浮動小数点、科学表記法
- カテゴリ型: 低カーディナリティ、高カーディナリティ
- 日時型: ISO 8601, UNIX timestamp, 日本語形式
- PII: メール、電話番号、クレジットカード番号
- フリーテキスト: 長文、短文、記号混在

---

## PR#4: 金融ドメイン定義機能

**目的**: カラムに金融特有の役割（取引金額、残高、タイムスタンプ等）を割り当てる
**進捗**: ✅ FinancialDomainMapper 実装・役割推定/カスタム設定テスト完了（PR#4）

### 実装モジュール

- `sfdao/ingestion/financial_domain.py`: 金融ドメイン定義

### TDDステップ

#### Test 1: 役割の自動推定 (Red)

```python
# tests/unit/ingestion/test_financial_domain.py
def test_infer_transaction_amount():
    schema = DataSchema(...)  # amount列を含むスキーマ
    mapper = FinancialDomainMapper()
    roles = mapper.infer_roles(schema)

    assert roles["amount"] == FinancialRole.TRANSACTION_AMOUNT
```

#### Test 2: カスタム役割の設定 (Red)

```python
def test_set_custom_roles():
    schema = DataSchema(...)
    mapper = FinancialDomainMapper()
    mapper.set_role("balance_after", FinancialRole.BALANCE)

    assert mapper.get_role("balance_after") == FinancialRole.BALANCE
```

### テスト計画

- カラム名からの自動推定テスト
- 手動設定のオーバーライドテスト
- エンティティ（顧客・取引）の設定テスト

---

## PR#5: Basic Evaluator - 統計検定

**目的**: 2つのデータセット（実データ vs 合成データ）の統計的類似性を評価する
**進捗**: ✅ StatisticalEvaluator 実装・単体テスト完了（PR#5）

### 実装モジュール

- `sfdao/evaluator/statistical.py`: 統計検定

### TDDステップ

#### Test 1: KS検定 (Red)

```python
# tests/unit/evaluator/test_statistical.py
def test_ks_test_identical_distributions():
    real = np.random.normal(0, 1, 1000)
    synthetic = np.random.normal(0, 1, 1000)

    evaluator = StatisticalEvaluator()
    result = evaluator.ks_test(real, synthetic)

    assert result.statistic < 0.1  # 小さいほど類似
    assert result.p_value > 0.05
```

#### Test 2: Jensen-Shannon Divergence (Red)

```python
def test_js_divergence():
    real = np.random.normal(0, 1, 1000)
    synthetic = np.random.normal(0, 1, 1000)

    evaluator = StatisticalEvaluator()
    divergence = evaluator.js_divergence(real, synthetic)

    assert 0 <= divergence <= 1
    assert divergence < 0.1  # 0に近いほど類似
```

### テスト計画

- 同一分布のテスト（理想的なケース）
- 異なる分布のテスト（差異検出）
- エッジケース（空配列、単一値、外れ値）

---

## PR#6: Financial Stylized Facts評価

**目的**: 金融特有の統計的性質（Fat Tail、Volatility Clustering）を評価する
**進捗**: ✅ FinancialFactsChecker 実装・単体テスト追加＆E2Eスモークテスト追加済み（PR#6）

### 実装モジュール

- `sfdao/evaluator/financial_facts.py`: 金融統計事実の検証

### TDDステップ

#### Test 1: Fat Tail検出 (Red)

```python
# tests/unit/evaluator/test_financial_facts.py
def test_fat_tail_check():
    # 正規分布（Fat Tailなし）
    normal_dist = np.random.normal(0, 1, 10000)

    # t分布（Fat Tailあり）
    t_dist = np.random.standard_t(df=3, size=10000)

    checker = FinancialFactsChecker()

    normal_kurtosis = checker.check_fat_tail(normal_dist)
    t_kurtosis = checker.check_fat_tail(t_dist)

    # t分布の方が尖度が高い
    assert t_kurtosis.excess_kurtosis > normal_kurtosis.excess_kurtosis
    assert t_kurtosis.excess_kurtosis > 3  # 正規分布より裾が厚い
```

#### Test 2: Volatility Clustering (Red)

```python
def test_volatility_clustering():
    # GARCHプロセスをシミュレート
    returns = simulate_garch_process(n=1000)

    checker = FinancialFactsChecker()
    result = checker.check_volatility_clustering(returns)

    assert result.ljung_box_p_value < 0.05  # 自己相関あり
    assert result.arch_test_p_value < 0.05  # ARCH効果あり
```

### テスト計画

- 理論分布との比較テスト
- 実際の金融データでの検証
- パラメータ感度分析

---

## PR#7: Privacy評価

**目的**: 合成データからの個人情報再識別リスクを評価する
**進捗**: ✅ PrivacyEvaluator 実装・単体/E2Eスモークテスト完了（PR#7）

### 実装モジュール

- `sfdao/evaluator/privacy.py`: プライバシー評価

### TDDステップ

#### Test 1: Distance to Closest Record (Red)

```python
# tests/unit/evaluator/test_privacy.py
def test_dcr_calculation():
    real = np.array([[1, 2], [3, 4], [5, 6]])
    synthetic = np.array([[1.1, 2.1], [10, 11]])

    evaluator = PrivacyEvaluator()
    dcr = evaluator.distance_to_closest_record(real, synthetic)

    assert len(dcr) == len(synthetic)
    assert dcr[0] < dcr[1]  # [1.1, 2.1]は[1, 2]に近い
```

#### Test 2: Re-identification Risk (Red)

```python
def test_reidentification_risk():
    real = load_test_data("real_customers.csv")
    synthetic = load_test_data("synthetic_customers.csv")

    evaluator = PrivacyEvaluator()
    risk_score = evaluator.reidentification_risk(real, synthetic)

    assert 0 <= risk_score <= 1
    # 理想的には低リスク
    assert risk_score < 0.1
```

### テスト計画

- 完全に異なるデータ（低リスク）
- コピーされたデータ（高リスク）
- ノイズ付加データ（中リスク）

---

## PR#8: 評価スコアリング統合とレポート生成基盤

**目的**: 各評価指標を統合し、総合スコアを計算する
**進捗**: ✅ CompositeScorer/Reporter基盤実装・単体/E2Eスモークテスト完了（PR#8）

### 実装モジュール

- `sfdao/evaluator/scoring.py`: スコア統合
- `sfdao/reporter/base.py`: レポート基底クラス

### TDDステップ

#### Test 1: 総合スコア計算 (Red)

```python
# tests/unit/evaluator/test_scoring.py
def test_composite_score_calculation():
    metrics = {
        "quality": 0.8,
        "utility": 0.7,
        "privacy": 0.9
    }
    weights = {
        "quality": 0.4,
        "utility": 0.3,
        "privacy": 0.3
    }

    scorer = CompositeScorer(weights)
    total_score = scorer.calculate(metrics)

    expected = 0.8*0.4 + 0.7*0.3 + 0.9*0.3
    assert abs(total_score - expected) < 0.001
```

### テスト計画

- 重み付け変更のテスト
- 制約違反ペナルティのテスト
- エッジケース（全てゼロ、全て1）

---

## PR#9: CLIインターフェース実装

**目的**: コマンドラインから評価を実行できるようにする
**進捗**: ✅ CLI auditで統計/プライバシー/金融ファクト評価を統合し、メタデータ出力と単体テスト更新まで完了（PR#9）

### 実装モジュール

- `sfdao/cli/main.py`: CLIエントリーポイント

### CLIコマンド設計

```bash
# 基本的な評価
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html

# 詳細設定
sfdao audit \
  --real data/real.csv \
  --synthetic data/synthetic.csv \
  --config config.yaml \
  --output report.html \
  --format html,pdf
```

### TDDステップ

#### Test 1: CLI引数パース (Red)

```python
# tests/unit/cli/test_main.py
def test_parse_basic_arguments():
    from sfdao.cli.main import parse_args

    args = parse_args([
        "audit",
        "--real", "real.csv",
        "--synthetic", "synthetic.csv"
    ])

    assert args.command == "audit"
    assert args.real == "real.csv"
    assert args.synthetic == "synthetic.csv"
```

### テスト計画

- 引数パースのテスト
- エラーメッセージのテスト
- ヘルプ表示のテスト
- End-to-Endテスト（実際のファイルで実行）

Tests: `tests/unit/cli/test_main.py`

---

## PR#10: HTML/PDFレポート出力機能

**目的**: 評価結果を見やすいレポートとして出力する
**進捗**: ✅ HTML/PDF Reporter実装・テンプレート追加・単体/統合テスト追加（PR#10）

### 実装モジュール

- `sfdao/reporter/html.py`: HTML生成
- `sfdao/reporter/pdf.py`: PDF生成
- `sfdao/reporter/templates/`: Jinja2テンプレート

### レポート構成

1. **Executive Summary**: 総合スコアとハイライト
2. **Data Overview**: データセット基本情報
3. **Statistical Quality**: KS検定、JS Divergence結果
4. **Financial Facts**: Fat Tail、Volatility Clustering
5. **Privacy Assessment**: 再識別リスク
6. **Recommendations**: 改善提案（Phase 2以降で拡張）

### TDDステップ

#### Test 1: HTMLレポート生成 (Red)

```python
# tests/unit/reporter/test_html.py
def test_generate_html_report():
    evaluation_result = create_mock_evaluation_result()

    reporter = HTMLReporter()
    html = reporter.generate(evaluation_result)

    assert "<html>" in html
    assert "Overall Score" in html
    assert evaluation_result.composite_score in html
```

### テスト計画

- テンプレートレンダリングのテスト
- グラフ生成のテスト
- PDF変換のテスト
- 日本語文字化けチェック

Tests: `tests/unit/reporter/test_html.py`, `tests/unit/reporter/test_pdf.py`, `tests/integration/reporter/test_audit_html_output.py`

---

## PR#11: 統合テストとドキュメント整備

**目的**: Phase 1の完成度を確認し、使いやすいドキュメントを提供する
**進捗**: ✅ フルパイプラインE2Eテスト追加・`docs/` 配下のドキュメント整備（PR#11）

### 統合テスト

```python
# tests/e2e/test_full_pipeline.py
def test_full_audit_pipeline_generates_html_report(tmp_path: Path) -> None:
    """CLIを使って end-to-end で監査を実行し、HTMLレポート生成まで確認する。"""

    real_csv = Path("tests/fixtures/creditcard_real_sample.csv")
    synthetic_csv = tmp_path / "synthetic.csv"
    report_path = tmp_path / "report.html"

    generate_simple_synthetic(real_csv, synthetic_csv, n_samples=50, random_state=42)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.cli.main",
            "audit",
            "--real",
            str(real_csv),
            "--synthetic",
            str(synthetic_csv),
            "--output",
            str(report_path),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert report_path.exists()
    assert "Overall Score" in report_path.read_text(encoding="utf-8")
```

### ドキュメント整備

1. **README.md**: インストール、クイックスタート
2. **docs/USAGE.md**: 詳細な使い方
3. **docs/ARCHITECTURE.md**: アーキテクチャ解説
4. **docs/API.md**: APIリファレンス
5. **docs/METRICS.md**: 評価指標の数式と解釈

### テスト計画

- 大規模データでの性能テスト（100万行）
- 異常データでのロバストネステスト
- メモリリークチェック

Tests: `tests/e2e/test_full_pipeline.py`

---

## PR#12: `example/` サンプルプロジェクト追加

**目的**: `sfdao` を初見のユーザーでも最短で試せるよう、`example/` 配下に「データ準備 → 監査 → レポート出力」までを再現できるサンプルプロジェクトを追加する。
**進捗**: ✅ `example/` サンプルプロジェクト追加・E2Eスモークテスト追加（PR#12）

### 成果物（案）

- `example/README.md`: 実行手順（Poetry前提）と期待する出力例
- `example/data/`: 例用の小さな実データ（`creditcard_real_sample.csv`）
- `example/scripts/`:
  - 合成データ生成（`python -m sfdao.scripts.generate_test_synthetic_data ...` のラッパー）
  - 監査実行（`sfdao audit ...`）のワンショットスクリプト（任意）
- `example/output/`: 生成物の出力先（成果物は原則コミットしない）

### 受け入れ条件（DoD）

- `poetry install` 後に、README通りの手順で `example/output/report.(txt|html|pdf)` のいずれかが生成できる
- 例のコマンドが相対パスで成立し、`example/` 単体で読める（テスト用パスに依存しない）
- CIで軽量スモーク（例: 生成→`sfdao audit`→レポート生成）を実行可能

### タスク

- [x] `example/` ディレクトリ構成を決定し、必要ファイルを追加
- [x] 例用データを同梱する（サイズ最小）か、seed固定の生成スクリプトで作るかを決める
- [x] 合成データ生成手順（`sfdao/scripts/generate_test_synthetic_data.py`）をサンプルに組み込む
- [x] `sfdao audit` の実行例（console出力/ファイル出力）を `example/README.md` に記載
- [x] `tests/e2e/` に `example/` のスモークテストを追加（CIで実行）

Tests: `tests/e2e/test_example_project_smoke.py`

---

## テストデータ準備

各PRで使用するテストデータを `tests/fixtures/` に用意します。

### Phase 1で必要なデータセット

1. **sample_transactions.csv** (100行): 基本的な取引データ
   - 手動作成（カラム: transaction_id, amount, balance, timestamp, customer_id, description）
2. **large_transactions.csv** (10万行): 性能テスト用
   - sample_transactions.csvを拡張して生成
3. **creditcard_real.csv**: 実際の金融データ
   - Kaggle Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - 284,807件の取引データ（492件の不正取引を含む）
   - カラム: Time, V1-V28 (PCA変換済み), Amount, Class
4. **creditcard_synthetic.csv**: 合成データ（評価用）
   - Phase 2以降でCTGANを使って生成予定
   - Phase 1では、簡易的な統計モデル（Gaussian Copula）で生成

### データセットの取得方法

```bash
# Kaggle CLI のインストール（初回のみ）
pip install kaggle

# Kaggle API認証情報の設定
# https://www.kaggle.com/account からAPI Tokenをダウンロードし、~/.kaggle/kaggle.json に配置

# Credit Card Fraud Detectionデータセットのダウンロード
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
unzip data/creditcardfraud.zip -d data/

# tests/fixtures/ にサンプルデータをコピー
cp data/creditcard.csv tests/fixtures/creditcard_real.csv
```

### 合成データ生成スクリプト（Phase 1用の簡易版）

Phase 1では評価機能のテストのため、簡易的な合成データを生成します：

```python
# scripts/generate_test_synthetic_data.py
import pandas as pd
import numpy as np
from scipy.stats import norm

def generate_simple_synthetic(real_csv_path, output_path, n_samples=10000):
    """統計的性質を保持した簡易合成データを生成"""
    real_df = pd.read_csv(real_csv_path)

    synthetic_data = {}
    for col in real_df.columns:
        if col == 'Class':  # ラベルは元の分布を保持
            synthetic_data[col] = np.random.choice(
                real_df[col].values,
                size=n_samples
            )
        else:
            # 平均と標準偏差を保持
            mean = real_df[col].mean()
            std = real_df[col].std()
            synthetic_data[col] = np.random.normal(mean, std, n_samples)

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    generate_simple_synthetic(
        "data/creditcard.csv",
        "tests/fixtures/creditcard_synthetic.csv",
        n_samples=10000
    )
```

---

## 進捗管理

各PRは以下の基準でクローズします：

- [ ] すべてのテストがパス
- [ ] コードカバレッジ90%以上
- [ ] mypy型チェックエラーゼロ
- [ ] Black/Flake8チェックパス
- [ ] コードレビュー完了
- [ ] ドキュメント更新

---

## Phase 1完了の定義 (Definition of Done)

- [x] CLIから評価レポートが出力できる
- [x] 金融特化の評価指標が正しく計算される
- [x] HTMLレポートが生成される
- [x] テストカバレッジ90%以上
- [x] ドキュメントが整備されている
- [x] 1つ以上の実際の金融データセットで動作確認済み

---

## 次のステップ（Phase 2への準備）

Phase 2では「生成＋整合性」を最小スコープで成立させ、`audit` と接続した end-to-end を完成させます。

### Phase 2 実装タスク案（PR分割の例）

1. **Generator基盤の確立**
   - 生成器インターフェース（入力スキーマ/seed/出力整合性）を定義
   - 設定（YAML/JSON）をPydanticでバリデーションできるようにする

2. **Baseline Generator の実装**
   - まずは依存が少ない方式（統計サンプリング/ガウスコピュラ等）で数値列の生成を実装
   - 列名/型/欠損の取り扱いポリシーを明文化

3. **Constraint & Logic Guard**
   - 会計恒等式/残高整合性/時間順序/値域制約などのルール定義
   - 違反の「検出/除外/補正」の方針を選べるようにする
   - 監査レポートに制約違反の概要を含める（Phase 1レポートと整合）

4. **Scenario Injection（手動設定）**
   - シナリオをYAMLで宣言し、生成データ（または入力データ）に変換として適用
   - 例: tail risk注入、カテゴリ比率変更、期間ショック、外れ値増加

5. **CLI/ワークフロー統合**
   - `sfdao generate`（生成）→ `sfdao audit`（評価）の基本導線を提供
   - 必要に応じて `sfdao run`（generate→guard→audit）等の一括コマンドを検討

6. **E2E/ドキュメント/サンプル拡充**
   - `example/` を Phase 2 のワークフロー（生成→監査）に対応させる
   - CIでPhase 2の最小E2E（小規模データ）をスモーク実行

7. **性能/ベンチマーク（スケール検証）**
   - ベンチマーク用データセット取得手順を整備（Kaggle `creditcard.csv` 等、ローカルの `data/` 配下で管理）
   - サイズ別（例: 1k/10k/100k rows）の `audit` 実行時間・メモリ計測スクリプトを追加（macOS想定）
   - `PrivacyEvaluator` の計算量対策（サンプリング/近似最近傍/バッチ化等）を設計し、設定で制御できるようにする
   - 大規模入力時の警告/フォールバック方針（例: privacyはサンプルで計算）をドキュメント化

### Phase 2 PR計画（詳細）

以下は「Phase 2 を最小スコープで成立させる」ための PR 分割案です（PR番号は計画上の通し番号）。

#### PR#13: 設定スキーマ & 生成インターフェース基盤

**目的**: Phase 2で増える設定（生成/制約/シナリオ/評価）を Pydantic v2 で定義し、CLIで安全に扱えるようにする。
**進捗**: ✅ 設定スキーマ/ローダー追加・`sfdao generate/run` の config バリデーション追加（PR#13）

**成果物（案）**

- 設定モデル（例: `sfdao/config/models.py`）
- 設定ロード/バリデーション（例: `sfdao/config/loader.py`）
- Generatorインターフェース（例: `sfdao/generator/base.py`）
- CLIの骨組み（例: `sfdao generate --config ...`, `sfdao run --config ...` の help/バリデーション）
- サンプル設定（例: `example/config/phase2.yaml`）

**受け入れ条件（DoD）**

- 設定ファイルが strict にバリデーションされ、無効な入力で非0終了する
- seed を含む設定が CLI から渡せる（再現性の前提を作る）
- 単体テストで「設定の正常系/異常系」をカバーする

**タスク**

- [x] Phase 2用の設定スキーマを定義（generator/guard/scenario/audit）
- [x] `--config` を受け取る共通ローダーを追加（YAML/JSON、Pydantic v2）
- [x] Generator の最小インターフェース（`fit`/`sample` もしくは `generate`）を定義
- [x] CLIの引数設計（Phase 1の `audit` と整合、help整備）

Tests: `tests/unit/config/test_phase2_config.py`, `tests/unit/cli/test_generate_args.py`

---

#### PR#14: Baseline Generator（最小の生成器）

**目的**: 依存が少ない方式で「実データ → 合成CSV出力」を成立させる（まずは数値列中心、seed固定）。
**進捗**: ✅ BaselineGenerator 実装・`sfdao generate` でCSV出力・単体/E2Eテスト追加（PR#14）

**成果物（案）**

- Baseline generator 実装（例: `sfdao/generator/baseline.py`）
- 生成結果のスキーマ整合（列順/列名/型/欠損ポリシー）
- `sfdao generate` で CSV を出力できる

**受け入れ条件（DoD）**

- 同一入力 + 同一設定（seed含む）で同一出力が得られる
- 最低限の欠損/定数列などに対してクラッシュしない

**タスク**

- [x] Phase 1 の `generate_test_synthetic_data` は既存用途として維持し、Phase 2 は `BaselineGenerator` を新規実装（方針決定）
- [x] 数値列は統計サンプリング（平均/分散）ベースで生成（Phase 2の第一歩）
- [x] ラベル/カテゴリ列は分布を保持してサンプリング（最小）
- [x] 生成物を `CSVLoader` で再読込して、型が破綻しないことを確認する

Tests: `tests/unit/generator/test_baseline_generator.py`, `tests/e2e/test_generate_smoke.py`

---

#### PR#15: Constraint & Logic Guard（検出/除外/補正）

**目的**: 「統計的にはそれっぽいがビジネスとしてあり得ない」データを、ルールで検出・制御できるようにする。

**成果物（案）**

- ルール定義と適用エンジン（例: `sfdao/guard/`）
- 方針（検出のみ / 除外 / クリップ / 補正）の切り替え（設定で制御）
- 違反サマリーを `audit` レポート metadata に載せる

**受け入れ条件（DoD）**

- ルールが設定でON/OFFできる
- 違反件数・割合がレポートに出る（監査視点の説明性）

**タスク**

- [x] 最小のルールセットを決める（例: 値域、欠損率、非負制約、単調増加timestamp、IDユニーク等）
- [x] ルールの実装と適用順序を定義
- [x] 違反の記録フォーマット（metadata用）を定義

**進捗**: ✅ Guardエンジン/ルール実装・統合テスト完了（PR#15）

Tests: `tests/unit/guard/test_rules.py`, `tests/integration/test_guard_with_generate.py`

---

#### PR#16: Scenario Injection（手動シナリオ）

**目的**: 生成データに対して「外れ値増加」「期間ショック」「カテゴリ比率変更」などを設定ファイルで適用できるようにする。
**進捗**: ✅ ScenarioEngine 実装・基本変換（Scale/Shift/Clip/Outlier/Replace）実装・Generaror統合完了（PR#16）
**成果物（案）**

- シナリオ定義（YAML）と変換パイプライン（例: `sfdao/scenario/`）
- 変換ログ（何をどれだけ変えたか）を metadata に残す

**受け入れ条件（DoD）**

- シナリオが seed 固定で再現可能
- 変換の適用結果が監査レポートから追える（監査証跡）

**タスク**

- [x] 最小の変換セットを決める（scale/shift/clip/outlier/rate-change 等）
- [x] 数値/カテゴリ別に適用可能な変換を実装
- [-] 変換ログを metadata に統一フォーマットで格納 (Partial: Engine returns metadata, integration simplified)
Tests: `tests/unit/scenario/test_injection.py`

---

#### PR#17: `generate → guard → audit` のワークフロー統合（`sfdao run`）

**目的**: Phase 2 の最小E2E（生成→制約→監査→レポート）を CLI でワンショット実行できるようにする。
**進捗**: ✅ `sfdao run` 実装・E2Eテスト追加・Exampleプロジェクト更新完了（PR#17）
**成果物（案）**

- `sfdao run` コマンド（or `sfdao generate` + `sfdao audit` の一括モード）
- `example/` を Phase 2 フローに対応（設定ファイル込み）
- CIで動く最小E2E（小規模データ）

**受け入れ条件（DoD）**

- `poetry install` 後に `example/` の手順で「生成→監査→レポート」が再現できる
- CIで `run` スモークがパスする

**タスク**

- [x] `run` の入出力（outdir/命名/上書き方針）を決める
- [x] `example/README.md` を Phase 2 手順に追随させる
- [x] E2E テスト（小規模データ）を追加

Tests: `tests/e2e/test_run_pipeline_smoke.py`

---

#### PR#18: ベンチマーク & スケール対策（privacy中心）

**目的**: 大規模データに対しても現実的な時間で回るように「計測」と「フォールバック」を整備する。
**進捗**: ✅ Benchmark Script 実装・PrivacySampling 実装・ドキュメント（BENCHMARK.md）作成完了（PR#18）

**成果物（案）**

- ベンチマークスクリプト（例: `sfdao/scripts/benchmark_audit.py` or `tools/benchmark.py`）
- サイズ別（1k/10k/100k）計測手順のドキュメント（`docs/` or `DATA_SETUP.md`）
- PrivacyEvaluator の計算量対策（サンプリング/近似最近傍）を設定で制御

**受け入れ条件（DoD）**

- ベンチ実行手順がドキュメント化され、CIを壊さずにローカルで再現できる
- 大規模入力時に privacy をフル計算できない場合、明示的に警告し、設定に従ってフォールバックする

**タスク**

- [x] ベンチ実行のI/F（入力/サイズ/回数/出力形式）を決める
- [x] `audit` の性能ボトルネックを可視化する（最低限、time/memory）
- [x] privacy サンプルサイズ等の設定項目を追加し、レポートに反映する

Tests: `tests/e2e/test_benchmark_smoke.py`（最小。重い計測はCI外）

### Phase 2完了の定義（DoD）

- [x] 生成（CSV出力）→ 制約適用（必要なら）→ 監査（レポート出力）が一連で実行できる
- [x] seed固定で再現性が担保される（同一入力/設定で同一出力）
- [x] `example/` の手順が最新実装と一致している（設定ファイル含む）
- [x] CIで Phase 2 の最小E2E（小規模）がスモーク実行できる
- [x] ベンチマーク手順が整備され、サイズ別の目安（time/memory）が取得できる

---

## Phase 3: Production Readiness & Advanced Features

**対象フェーズ**: Phase 3 - "The Professional" (CI/CD, Advanced Gen/Eval, Release)
**目的**: プロダクション利用に耐えうる品質（CI/CD、ML評価）と高度な生成機能を提供する。

### Phase 3 PR計画

#### PR#19: CI/CD Hardening & Optimization

**目的**: 開発サイクルを加速し、品質保証を強化する。
**進捗**: ✅ CI最適化(Cache/Matrix)・Release Workflow作成・Docs更新完了（PR#19）

**成果物（案）**

- `.github/workflows/ci.yml` の最適化（Dependency/Pre-commit caching）
- Release Workflow（`.github/workflows/release.yml`）でタグプッシュ時に PyPI Publish と GitHub Release
- `docs/CONTRIBUTING.md` の更新

**受け入れ条件（DoD）**

- CI 実行時間が30%以上短縮（キャッシュヒット時）
- Python 3.10, 3.11, 3.12 のマトリクスが全てパス
- タグプッシュで PyPI にテストリリースがデプロイできる

**タスク**

- [ ] Poetry の dependency cache を GitHub Actions に追加
- [ ] Python 3.12 をマトリクスに追加し、テストがパスすることを確認
- [ ] `release.yml` を作成（trusted publishing または API token）
- [ ] リリースノート自動生成の検討

Tests: 既存テストが全てパスすることを確認

---

#### PR#20: Advanced Generator (CTGAN Integration)

**目的**: ベースライン（統計）を超えた、相関関係を学習できる高精度な合成データを生成する。
**進捗**: ✅ CTGANGenerator 実装・pyproject.toml更新・単体/E2Eテスト追加（PR#20）

**成果物（案）**

- `sfdao/generator/ctgan.py` （SDV/CTGAN ラッパー）
- `pyproject.toml` に `[extras]` 定義（`pip install sfdao[deep]`）
- 設定で `generator.type: ctgan` を選択可能に

**受け入れ条件（DoD）**

- `sfdao generate --config ctgan.yaml` で学習ベース合成が動作
- `extras` なしのデフォルトインストールでは CTGAN が無効化される（import error が出ない）

**タスク**

- [x] `sdv` または `ydata-synthetic` の選定と依存追加
- [x] `CTGANGenerator` クラス実装（`fit`/`sample` インターフェース）
- [x] Generator Factory に CTGAN を登録
- [x] CI で `extras` 無しテストがパスすることを確認（optional dependency）
- [x] E2E テスト（CTGANで生成→監査）を追加（CI では skip 可）

Tests: `tests/unit/generator/test_ctgan.py`, `tests/unit/generator/test_factory.py`, `tests/e2e/test_ctgan_smoke.py`

---

#### PR#21: Machine Learning Utility Evaluation

**目的**: 「合成データで学習したモデルが、実データで学習したモデルと同等の性能を出せるか」を定量評価する。
**進捗**: ✅ TSTR評価ロジック（RandomForest/LogisticRegression）実装・CLI統合・レポートテンプレート更新完了（PR#21）

**成果物（案）**

- `sfdao/evaluator/ml_utility.py`（TSTR 評価）
- レポートに ML Utility セクション追加

**受け入れ条件（DoD）**

- `--ml-utility` オプションで AUC/F1 がレポートに出力される
- デフォルトは OFF（計算コストのため）

**タスク**

- [x] TSTR ロジックの実装（LogisticRegression or RandomForest）
- [x] ターゲット列の指定方法を設計（`--ml-target` CLIオプション）
- [x] レポートテンプレートに ML Utility セクション追加
- [x] CLI に `--ml-utility` フラグ追加

Tests: `tests/unit/evaluator/test_ml_utility.py`, `tests/e2e/test_ml_utility_smoke.py`

---

#### PR#22: PyPI Publication & Final Polish

**目的**: ライブラリとして一般公開できる状態にする。
**進捗**: ✅ PyPI用メタデータ追加・CHANGELOG作成・README整備・Build検証完了（PR#22）

**成果物（案）**

- `pyproject.toml` のメタデータ完備
- `README.md` のバッジ（PyPI, CI, Coverage）
- `CHANGELOG.md` の作成
- バージョン `0.1.0` タグ

**受け入れ条件（DoD）**

- `pip install sfdao` で PyPI からインストール可能
- README の Quick Start が動作する

**タスク**

- [x] `pyproject.toml` に License, Classifiers, Homepage を追加
- [x] `CHANGELOG.md` を作成
- [x] README にバッジを追加
- [x] TestPyPI でテストリリース（Manual）
- [x] PyPI への正式リリース（Manual）

Tests: `poetry build` passed.

---

#### PR#23: Audit UX & Progress Reporting

**目的**: 長時間実行時に進捗・ステータスが分かりづらい問題を解消し、CLI体験を改善する。
**進捗**: ✅ 進捗表示/ハートビート/詳細ログの追加とCLIオプション実装、ユニット/統合テスト追加完了（PR#23）

**背景**

- 大規模データの `sfdao audit` 実行時、処理が長く「現在何をしているか」が分からない。
- ターミナル上で中間状況（フェーズ、推定残り時間、現在の計算対象）が可視化される必要がある。

**成果物（案）**

- CLIに進捗表示・ステータス出力（Rich progress or simple text）
- フェーズ単位のログ出力（例: Load → Schema → Stats → Privacy → Report）
- 長時間計算の heartbeat 出力（一定間隔で進捗/所要時間/対象件数）
- Quiet/No-progress モード（CI/ログ用途）

**UX仕様（案）**

1. **フェーズ表示**
   - 各主要評価フェーズの開始・完了を明確に出力
   - 表示例: `Step 3/6: Privacy Evaluation (sample=5000)`
2. **進捗インジケータ**
   - 可能なタスクは進捗バー（行数/列数ベース）で表示
   - 不可能なタスクはスピナー＋経過時間
3. **ハートビート**
   - `--status-interval`（秒）で定期出力（既定: 30秒）
   - 出力内容: 経過時間、現在フェーズ、処理対象数、メモリ（可能なら）
4. **オプション**
   - `--quiet` は現行通り（完全抑制）
   - `--no-progress` で進捗UIを無効化（ログのみ）
   - `--verbose` で詳細ログ（サンプリング数、閾値など）

**受け入れ条件（DoD）**

- `sfdao audit` 実行中にフェーズ遷移が分かる
- 30秒以上の処理で定期的な状況出力が行われる
- `--quiet`/`--no-progress` で出力制御が可能
- 既存のテストが壊れない（CLI引数の互換性維持）

**タスク**

- [x] `sfdao/cli/main.py` に進捗/ステータス出力の設計を追加
- [x] 各評価器（statistics/privacy/financial/ml）にフェーズ境界フックを用意
- [x] `--status-interval` / `--no-progress` / `--verbose` をCLIに追加
- [x] ユニットテスト：引数パース/出力抑制/デフォルト動作

Tests: `tests/unit/cli/test_audit_progress.py`, `tests/integration/test_audit_progress.py`

### Phase 3完了の定義（DoD）

- [x] CIが高速かつ安定して回る（キャッシュ有効化）
- [x] 学習ベースの生成器が選択できる
- [x] MLモデルの性能（Utility）が評価レポートに出力できる
- [x] PyPIへのデプロイフローが確立されている

---

## 環境設定

以下の内容で開発を進めます：

### 確認済み事項

1. **テストデータセット**: Kaggle Credit Card Fraud Detection データセット
   - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - 284,807件の実取引データ（不正検知用）
   - Phase 1では簡易的な統計モデルで合成データを生成してテストに使用

2. **開発環境**: ローカル環境
   - Python 3.10+
   - Poetry による依存関係管理
   - macOS (Darwin 24.6.0)

3. **Git管理**: GitHub
   - Repository: `sfdao`（新規作成）
   - GitHub CLI使用
   - GitHub Actions でCI/CD

4. **PR作成**: 機能単位（上記PR#1～PR#11）
   - 各PRは独立してマージ可能
   - TDDサイクルを厳守
   - コードレビュー後にマージ

---

**次のステップ**

この計画書の内容に基づき、以下の順序で進めます：

1. ✅ GitHubリポジトリの作成
2. ✅ PR#1: プロジェクト構造とCI/CD設定
3. ✅ PR#2: Data Ingestion基本機能（PR作成済み）
4. ⏳ PR#2以降: 機能実装（TDDベース）

実装の開始時期はユーザーの指示を待ちます。
