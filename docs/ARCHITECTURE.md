# Architecture

SFDAO Phase 1 ("The Auditor") は、CSV入力を読み込み、統計・金融ファクト・プライバシーの観点で評価し、レポートを出力します。

## モジュール構成

- `sfdao/ingestion/`: CSV読み込み・スキーマ/型検出（入力の整形）
- `sfdao/evaluator/`: 評価指標（統計・金融ファクト・プライバシー）
- `sfdao/evaluator/scoring.py`: メトリクスの重み付け合算（Composite Score）
- `sfdao/reporter/`: プレーンテキスト/HTML/PDF レポート生成
- `sfdao/cli/`: `sfdao audit` コマンドのエントリポイント

## データフロー（audit）

1. `sfdao.cli.main:audit` が入力パスを検証
2. `sfdao.cli.audit:run_audit` が `CSVLoader` でCSVを読み込み
3. `StatisticalEvaluator` が数値列の分布類似度を計算
4. `PrivacyEvaluator` がDCR（最近傍距離）ベースのリスクを計算
5. `FinancialFactsChecker` が金融時系列の簡易特性を要約
6. `CompositeScorer` が `quality/utility/privacy` を合算し総合スコアを算出
7. 出力拡張子に応じて `PlainTextReporter/HTMLReporter/PDFReporter` を選択してレポートを出力

