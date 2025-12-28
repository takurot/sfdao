# Usage

## CLI

監査（Audit）を実行して、実データと合成データを比較します。

```bash
sfdao audit --real <real.csv> --synthetic <synthetic.csv> --output report.html
```

### オプション

- `--real/-r`: 実データCSV
- `--synthetic/-s`: 合成データCSV
- `--output/-o`: 出力先（未指定の場合は標準出力）
- `--quiet/-q`: コンソール出力を抑制（`--output` と併用推奨）

### 出力形式

`--output` の拡張子で自動判定します。

- `.html` / `.htm`: HTMLレポート
- `.pdf`: PDFレポート（WeasyPrintのシステム依存あり）
- それ以外: プレーンテキスト

## テスト用の合成データ生成

リポジトリ同梱のサンプルCSVから、簡易合成データを生成できます。

```bash
python -m sfdao.scripts.generate_test_synthetic_data \
  tests/fixtures/creditcard_real_sample.csv \
  ./synthetic.csv \
  --n-samples 500 \
  --random-state 42
```

生成後に監査を実行します。

```bash
sfdao audit --real tests/fixtures/creditcard_real_sample.csv --synthetic ./synthetic.csv --output report.html
```

Poetry環境の場合は `poetry run` を付けて実行してください。

