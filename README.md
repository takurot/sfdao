# SFDAO - Synthetic Finance Data Auditor & Optimizer

**金融コンプライアンス準拠・合成データ品質保証プラットフォーム**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

SFDAOは、金融業界向けの合成データ（Synthetic Data）の品質を監査・評価するためのツールです。Phase 1では、既存データの品質評価に特化した「The Auditor」機能を提供します。

### 主な機能

- **統計的品質評価**: KS検定、Jensen-Shannon Divergenceによる分布比較
- **金融特有の評価**: Fat Tail検出、Volatility Clusteringの確認
- **プライバシー評価**: 再識別リスク、Distance to Closest Record
- **自動型検出**: 数値、カテゴリ、日時、PII（個人特定情報）の自動判定
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

# 依存関係のインストール
poetry install

# 仮想環境の有効化
poetry shell
```

## Quick Start

```bash
# 基本的な評価の実行
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html

# 詳細設定を使用した評価
sfdao audit \
  --real data/real.csv \
  --synthetic data/synthetic.csv \
  --config config.yaml \
  --output report.html \
  --format html,pdf
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

- [実装計画書](IMPLEMENTATION_PLAN.md)
- [製品仕様書](prompt/SPEC.md)
- [API リファレンス](docs/api/)（準備中）
- [評価指標の詳細](docs/METRICS.md)（準備中）

## Roadmap

### Phase 1: "The Auditor" (MVP) - 現在のフェーズ

- [x] プロジェクト構造とCI/CD設定
- [ ] Data Ingestion基本機能
- [ ] Auto-Type Detection
- [ ] 金融ドメイン定義
- [ ] Basic Evaluator（統計検定）
- [ ] Financial Stylized Facts評価
- [ ] Privacy評価
- [ ] 評価スコアリング統合
- [ ] CLIインターフェース
- [ ] レポート生成機能
- [ ] 統合テストとドキュメント

### Phase 2: "The Generator & Logic"

- Hybrid Generator実装（CTGAN, Copula, LLM）
- Constraint & Logic Guard（会計恒等式チェック）
- Scenario Injection（ストレステスト）

### Phase 3: "The Optimizer"

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
