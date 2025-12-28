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

## 便利スクリプト（任意）

```bash
# まとめて実行（生成→監査）
bash example/scripts/run_example.sh
```
