# SFDAO Example

`sfdao` を初見のユーザーでも最短で試せるように、`example/` 配下に「データ準備 → 監査 → レポート出力」までを再現できるサンプルを用意しています。

## 前提

- リポジトリルートで `poetry install` 済み
- コマンドはすべてリポジトリルートから実行する

## 実行手順（HTMLレポート）

### 1) 合成データ生成

```bash
poetry run python -m sfdao.scripts.generate_test_synthetic_data \
  example/data/creditcard_real_sample.csv \
  example/output/creditcard_synthetic.csv \
  --n-samples 200 \
  --random-state 42
```

### 2) 監査実行（レポート出力）

```bash
poetry run sfdao audit \
  --real example/data/creditcard_real_sample.csv \
  --synthetic example/output/creditcard_synthetic.csv \
  --output example/output/report.html
```

## 期待する生成物

- `example/output/creditcard_synthetic.csv`
- `example/output/report.html`

macOS の場合:

```bash
open example/output/report.html
```


## Phase 2 Workflow (Generate & Audit)

`sfdao run` コマンドを使用すると、設定ファイルに基づいて「生成→監査→レポート」を一括で実行できます。

### 実行手順

```bash
poetry run sfdao run \
  --real example/data/creditcard_real_sample.csv \
  --config example/config/phase2.yaml \
  --out-dir example/output_phase2
```

### 期待する生成物

- `example/output_phase2/synthetic.csv`
- `example/output_phase2/report.html`

