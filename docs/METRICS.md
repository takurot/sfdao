# Metrics

SFDAO Phase 1では、以下の3系統の指標を計算し、`quality` / `utility` / `privacy` の3つのスコア（0〜1）に集約します。

## Statistical Quality

- **KS検定（Kolmogorov–Smirnov）**: 実データと合成データの分布差を測る統計量（小さいほど近い）
- **Jensen–Shannon Divergence**: 2つの分布の距離（小さいほど近い）

CLIでは数値列ごとに計算し、平均してスコア化します。

- `quality = clamp(1 - avg_ks, 0, 1)`
- `utility = clamp(1 - avg_js, 0, 1)`

## Privacy

`PrivacyEvaluator` は、合成レコードごとに実データの最近傍距離（DCR: Distance to Closest Record）を計算し、スケールで正規化してリスクを推定します。

- `risk = mean(exp(-dcr / scale))`（0〜1）
- `privacy = clamp(1 - risk, 0, 1)`

## Composite Score

`CompositeScorer` が各メトリクスを重み付けして合算します（必要に応じて制約ペナルティも適用）。

デフォルト重み（CLI）:

- `quality`: 0.4
- `utility`: 0.3
- `privacy`: 0.3

