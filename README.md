# SFDAO - Synthetic Finance Data Auditor & Optimizer

**金融コンプライアンス準拠・合成データ品質保証プラットフォーム**

[![PyPI version](https://badge.fury.io/py/sfdao.svg)](https://badge.fury.io/py/sfdao)
[![Python Version](https://img.shields.io/pypi/pyversions/sfdao.svg)](https://pypi.org/project/sfdao/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/takurot/sfdao/actions/workflows/ci.yml/badge.svg)](https://github.com/takurot/sfdao/actions)
[![Codecov](https://codecov.io/gh/takurot/sfdao/branch/main/graph/badge.svg)](https://codecov.io/gh/takurot/sfdao)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

SFDAOは、金融業界向けの合成データ（Synthetic Data）の生成・制約適用・監査を一体化したツールです。Phase 1〜3で、監査（Audit）だけでなく生成（Generate）、制約チェック（Guard）、シナリオ注入（Scenario）、ML Utility評価まで対応しています。

### 主な機能

- **統計的品質評価**: KS検定、Jensen-Shannon Divergenceによる分布比較
- **金融特有の評価**: Fat Tail検出、Volatility Clusteringの確認
- **プライバシー評価**: 再識別リスク、Distance to Closest Record
- **自動型検出**: 数値、カテゴリ、日時、PII（個人特定情報）の自動判定
- **生成ワークフロー**: `generate`/`run` による合成データ生成と監査の一括実行
- **制約・シナリオ**: Guardルール適用、シナリオ注入（scale/shift/clip/outlier等）
- **ML Utility評価**: TSTR（AUC/F1）によるモデル性能評価（任意）
- **レポート生成**: HTML/PDF形式での詳細レポート出力

## Installation

### Prerequisites

- Python 3.10以上
- Poetry（推奨）

### Setup

```bash
# リポジトリのクローン
git clone https://github.com/takurot/sfdao.git
cd sfdao

# Poetryのインストール（まだの場合）
curl -sSL https://install.python-poetry.org | python3 -

# PATHの設定（~/.zshrc または ~/.bash_profile に追加）
export PATH="$HOME/.local/bin:$PATH"

# 依存関係のインストール（pyproject.toml/poetry.lockに基づく）
poetry install

# 仮想環境の有効化（必要に応じて）
poetry shell

# macOS固有: WeasyPrint用の依存関係（PDF生成機能用）
brew install cairo pango gdk-pixbuf libffi
```

**注意**: macOS環境では `._*` ファイルが自動生成されますが、.gitignoreで除外されます。

## Quick Start

```bash
# 基本的な評価の実行
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html

# 出力形式は拡張子で自動判定（.txt/.html/.pdf）
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.txt
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.pdf

# テスト用の簡易合成データ生成
poetry run python -m sfdao.scripts.generate_test_synthetic_data \
  example/data/creditcard_real_sample.csv \
  example/output/creditcard_synthetic.csv \
  --n-samples 500 \
  --random-state 42

# 生成した合成データを監査
poetry run sfdao audit \
  --real example/data/creditcard_real_sample.csv \
  --synthetic example/output/creditcard_synthetic.csv \
  --output example/output/report.html

# Phase 2: 生成→制約→監査を一括実行
poetry run sfdao run --config example/config/phase2.yaml --outdir example/output
```

## Development

### TDD（テスト駆動開発）

このプロジェクトはTDDで開発されています。新機能の追加時は以下のサイクルに従ってください：

1. **Red**: 失敗するテストを書く
2. **Green**: テストをパスする最小限のコードを書く
3. **Refactor**: コードを整理・最適化する

### Testing

```bash
# すべてのテストを実行
pytest

# カバレッジレポート付きで実行
pytest --cov=sfdao --cov-report=html

# 特定のテストファイルのみ実行
pytest tests/unit/ingestion/test_loader.py
```

### Code Quality

```bash
# フォーマットチェック
black --check .

# フォーマット適用
black .

# Lintチェック
flake8 .

# 型チェック
mypy sfdao

# セキュリティチェック
bandit -r sfdao
```

## Project Structure

```
sfdao/
├── sfdao/                  # メインパッケージ
│   ├── ingestion/          # データ取り込みと型検出
│   ├── config/             # 設定スキーマ/ローダー
│   ├── generator/          # 合成データ生成
│   ├── guard/              # ルールベース制約チェック
│   ├── scenario/           # シナリオ注入
│   ├── evaluator/          # 評価指標の計算
│   ├── reporter/           # レポート生成
│   └── cli/                # CLIインターフェース
├── tests/                  # テストコード
│   ├── unit/               # 単体テスト
│   ├── integration/        # 統合テスト
│   └── e2e/                # End-to-Endテスト
├── docs/                   # ドキュメント
└── prompt/                 # 仕様書
```

## Documentation

- [実装計画書](prompt/PLAN.md)
- [製品仕様書](prompt/SPEC.md)
- [Example](example/README.md)
- [使い方](docs/USAGE.md)
- [アーキテクチャ](docs/ARCHITECTURE.md)
- [Python API](docs/API.md)
- [評価指標](docs/METRICS.md)

## Roadmap

### Phase 1: "The Auditor" (MVP)

- [x] プロジェクト構造とCI/CD設定
- [x] Data Ingestion基本機能
- [x] Auto-Type Detection
- [x] 金融ドメイン定義
- [x] Basic Evaluator（統計検定）
- [x] Financial Stylized Facts評価
- [x] Privacy評価
- [x] 評価スコアリング統合
- [x] CLIインターフェース
- [x] レポート生成機能
- [x] 統合テストとドキュメント

### Phase 2: "The Generator & Logic"

- [x] 設定スキーマ/ローダーとCLI統合（`generate`/`run`）
- [x] Baseline Generator（統計サンプリング）
- [x] Constraint & Logic Guard（ルール検出/除外/補正）
- [x] Scenario Injection（scale/shift/clip/outlier等）
- [x] E2Eワークフロー（generate→guard→audit）
- [x] ベンチマークとPrivacyサンプリング

### Phase 3: "The Professional"

- [x] CI/CD最適化とReleaseワークフロー
- [x] Advanced Generator（CTGAN, optional）
- [x] ML Utility評価（TSTR: AUC/F1）
- [x] PyPIメタデータ/CHANGELOG/README整備

### Future Ideas

- Rule Learning Engine（強化学習ベース）
- Auto-Tuning Mode（自律的品質改善）

## Contributing

貢献を歓迎します！以下の手順に従ってください：

1. このリポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. テストを書いてから実装（TDD）
4. コミット（`git commit -m 'Add amazing feature'`）
5. ブランチをプッシュ（`git push origin feature/amazing-feature`）
6. プルリクエストを作成

## License

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## Contact

プロジェクトに関する質問や提案がある場合は、Issueを作成してください。

## Acknowledgments

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- Kaggle Credit Card Fraud Detection Dataset
