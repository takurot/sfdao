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

---

## PR#10: HTML/PDFレポート出力機能

**目的**: 評価結果を見やすいレポートとして出力する

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

---

## PR#11: 統合テストとドキュメント整備

**目的**: Phase 1の完成度を確認し、使いやすいドキュメントを提供する

### 統合テスト

```python
# tests/e2e/test_full_pipeline.py
def test_full_audit_pipeline():
    """実データと合成データの完全な評価フロー"""

    # 1. データ読み込み
    real_data = "tests/fixtures/kaggle_credit_fraud_real.csv"
    synthetic_data = "tests/fixtures/kaggle_credit_fraud_synthetic.csv"

    # 2. CLIコマンド実行
    result = subprocess.run([
        "sfdao", "audit",
        "--real", real_data,
        "--synthetic", synthetic_data,
        "--output", "test_report.html"
    ], capture_output=True)

    # 3. 正常終了確認
    assert result.returncode == 0

    # 4. レポートファイル生成確認
    assert Path("test_report.html").exists()

    # 5. レポート内容確認
    with open("test_report.html") as f:
        html = f.read()
        assert "Overall Score" in html
        assert "KS Test" in html
        assert "Privacy Score" in html
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

- [ ] CLIから評価レポートが出力できる
- [ ] 金融特化の評価指標が正しく計算される
- [ ] HTMLレポートが生成される
- [ ] テストカバレッジ90%以上
- [ ] ドキュメントが整備されている
- [ ] 1つ以上の実際の金融データセットで動作確認済み

---

## 次のステップ（Phase 2への準備）

Phase 1完了後、以下の機能拡張を検討します：

- Hybrid Generator実装
- Constraint & Logic Guard
- Scenario Injection
- Rule Learning Engine（Phase 3）

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
2. ⏳ PR#1: プロジェクト構造とCI/CD設定
3. ⏳ PR#2以降: 機能実装（TDDベース）

実装の開始時期はユーザーの指示を待ちます。
