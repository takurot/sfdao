# テストデータセットのセットアップガイド

このドキュメントでは、SFDAOのPhase 1開発で使用するテストデータセットの準備方法を説明します。

## 概要

Phase 1では以下のデータセットを使用します：

1. **sample_transactions.csv** - 小規模テストデータ（手動作成）
2. **creditcard.csv** - Kaggle Credit Card Fraud Detection データセット
3. **creditcard_synthetic.csv** - 簡易合成データ（評価機能のテスト用）

---

## 1. Kaggle APIのセットアップ

### Kaggle CLIのインストール

```bash
pip install kaggle
```

### API認証情報の設定

1. [Kaggle Account Settings](https://www.kaggle.com/account) にアクセス
2. "Create New API Token" をクリック
3. `kaggle.json` ファイルがダウンロードされます
4. ファイルを適切な場所に配置：

```bash
# macOS / Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

---

## 2. Credit Card Fraud Detectionデータセットのダウンロード

### データセット情報

- **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **サイズ**: 約150MB（圧縮後）
- **レコード数**: 284,807件
- **不正取引**: 492件（0.172%）
- **特徴量**:
  - `Time`: 最初の取引からの経過秒数
  - `V1～V28`: PCA変換済みの特徴量（プライバシー保護のため）
  - `Amount`: 取引金額
  - `Class`: ラベル（0: 正常、1: 不正）

### ダウンロード手順

```bash
# プロジェクトルートディレクトリで実行
cd /path/to/sfdao

# dataディレクトリを作成
mkdir -p data

# データセットのダウンロード
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/

# 解凍
unzip data/creditcardfraud.zip -d data/

# 確認
ls -lh data/creditcard.csv
```

**出力例**:
```
-rw-r--r--  1 user  staff   144M  Jan  1 12:00 data/creditcard.csv
```

---

## 3. テスト用フィクスチャの準備

### tests/fixtures/ ディレクトリの作成

```bash
mkdir -p tests/fixtures
```

### 小規模サンプルデータの作成

Kaggleデータセットから小規模なサンプルを抽出します：

```bash
# 最初の1000行をサンプルとして抽出
head -n 1001 data/creditcard.csv > tests/fixtures/creditcard_sample.csv

# ファイルサイズの確認
ls -lh tests/fixtures/creditcard_sample.csv
```

---

## 4. 合成データの生成（Phase 1用）

Phase 1では評価機能のテストのため、簡易的な統計ベースの合成データを生成します。

### 生成スクリプトの作成

`scripts/generate_test_synthetic_data.py` はPR#1で作成されます。
以下は手動で実行する場合の手順です：

```bash
# scriptsディレクトリを作成
mkdir -p scripts

# Pythonスクリプトを作成（次のセクション参照）
```

### スクリプトの内容

```python
# scripts/generate_test_synthetic_data.py
"""
Phase 1用の簡易合成データ生成スクリプト

実データの統計的性質（平均、標準偏差）を保持した合成データを生成します。
注: Phase 2以降では、より高度なCTGAN/Copulaモデルを使用します。
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_simple_synthetic(
    real_csv_path: str,
    output_path: str,
    n_samples: int = 10000,
    random_seed: int = 42
):
    """
    統計的性質を保持した簡易合成データを生成

    Args:
        real_csv_path: 実データのCSVパス
        output_path: 出力先パス
        n_samples: 生成するサンプル数
        random_seed: 乱数シード
    """
    np.random.seed(random_seed)

    print(f"Loading real data from {real_csv_path}...")
    real_df = pd.read_csv(real_csv_path)

    print(f"Real data shape: {real_df.shape}")
    print(f"Generating {n_samples} synthetic samples...")

    synthetic_data = {}

    for col in real_df.columns:
        if col == 'Class':
            # クラスラベルは元の分布を保持
            class_dist = real_df[col].value_counts(normalize=True)
            synthetic_data[col] = np.random.choice(
                class_dist.index,
                size=n_samples,
                p=class_dist.values
            )
        else:
            # 数値カラムは正規分布で近似
            mean = real_df[col].mean()
            std = real_df[col].std()

            # 標準偏差が0の場合は定数値
            if std == 0:
                synthetic_data[col] = np.full(n_samples, mean)
            else:
                synthetic_data[col] = np.random.normal(mean, std, n_samples)

            # Amountカラムは負の値を避ける
            if col == 'Amount':
                synthetic_data[col] = np.abs(synthetic_data[col])

    synthetic_df = pd.DataFrame(synthetic_data)

    # 出力ディレクトリが存在しない場合は作成
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
    print(f"Synthetic data shape: {synthetic_df.shape}")

    # 基本統計の比較
    print("\n=== Basic Statistics Comparison ===")
    print("\nReal Data (Amount):")
    print(real_df['Amount'].describe())
    print("\nSynthetic Data (Amount):")
    print(synthetic_df['Amount'].describe())

if __name__ == "__main__":
    # サンプルデータから合成データを生成
    generate_simple_synthetic(
        real_csv_path="tests/fixtures/creditcard_sample.csv",
        output_path="tests/fixtures/creditcard_synthetic.csv",
        n_samples=1000
    )

    print("\n✅ Test synthetic data generation completed!")
```

### スクリプトの実行

```bash
# 依存関係のインストール（まだの場合）
pip install pandas numpy

# 合成データの生成
python scripts/generate_test_synthetic_data.py
```

**出力例**:
```
Loading real data from tests/fixtures/creditcard_sample.csv...
Real data shape: (1000, 31)
Generating 1000 synthetic samples...
Synthetic data saved to tests/fixtures/creditcard_synthetic.csv
Synthetic data shape: (1000, 31)

=== Basic Statistics Comparison ===

Real Data (Amount):
count    1000.000000
mean       88.291022
std       250.105092
...

Synthetic Data (Amount):
count    1000.000000
mean       89.123456
std       248.987654
...

✅ Test synthetic data generation completed!
```

---

## 5. データセットの検証

すべてのデータが正しく準備されたことを確認します：

```bash
# データファイルの存在確認
ls -lh data/creditcard.csv
ls -lh tests/fixtures/creditcard_sample.csv
ls -lh tests/fixtures/creditcard_synthetic.csv

# CSVファイルの行数確認
wc -l data/creditcard.csv
wc -l tests/fixtures/creditcard_sample.csv
wc -l tests/fixtures/creditcard_synthetic.csv

# ヘッダー確認
head -n 2 data/creditcard.csv
```

**期待される出力**:
```
284808 data/creditcard.csv
1001 tests/fixtures/creditcard_sample.csv
1001 tests/fixtures/creditcard_synthetic.csv  # ヘッダー含む
```

---

## 6. .gitignoreの確認

大容量のデータファイルがGitにコミットされないよう、`.gitignore` に以下が含まれていることを確認してください：

```gitignore
# Data files
data/
*.csv
!tests/fixtures/*.csv  # テスト用フィクスチャは除外
```

---

## トラブルシューティング

### Kaggle API認証エラー

```
OSError: Could not find kaggle.json
```

**対処法**:
- `~/.kaggle/kaggle.json` が存在し、パーミッションが `600` であることを確認
- `cat ~/.kaggle/kaggle.json` でファイル内容を確認

### データセットダウンロードの失敗

```
403 - Forbidden
```

**対処法**:
- Kaggleアカウントでログインしていることを確認
- データセットのルールに同意していることを確認
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud にアクセスして "Download" ボタンをクリック

### メモリ不足エラー

大規模データセット（全データ）を読み込む際にメモリ不足が発生する場合：

```python
# チャンク読み込みを使用
import pandas as pd

chunksize = 10000
for chunk in pd.read_csv('data/creditcard.csv', chunksize=chunksize):
    # 処理
    pass
```

---

## 次のステップ

データセットの準備が完了したら、以下に進みます：

1. **PR#1**: プロジェクト構造とCI/CD設定
2. **PR#2**: Data Ingestion機能の実装（これらのデータセットを使用）
3. **PR#5以降**: 評価機能のテスト（実データ vs 合成データ）

---

## 参考資料

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [研究論文](https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_using_Machine_Learning_as_Data_Mining_Technique)
