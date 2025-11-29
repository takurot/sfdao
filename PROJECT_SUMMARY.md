# SFDAO プロジェクトサマリー

**作成日**: 2025-11-29
**ステータス**: 準備完了（実装開始待ち）

---

## プロジェクト概要

**SFDAO (Synthetic Finance Data Auditor & Optimizer)** は、金融業界向けの合成データ品質保証プラットフォームです。

### Phase 1の目標

既存データに対する「評価レポート」を生成する診断ツールとしてのMVPを構築します。

**主要機能**:
- データ取り込みと自動型検出
- 統計的品質評価（KS検定、JS Divergence）
- 金融特有の評価（Fat Tail、Volatility Clustering）
- プライバシー評価（再識別リスク）
- HTML/PDFレポート生成

---

## リポジトリ情報

- **GitHub**: https://github.com/takurot/sfdao
- **ライセンス**: MIT
- **言語**: Python 3.10+
- **依存関係管理**: Poetry

---

## ドキュメント構成

| ファイル | 内容 |
|---------|------|
| `README.md` | プロジェクト概要、インストール、クイックスタート |
| `IMPLEMENTATION_PLAN.md` | 詳細な実装計画（PR#1～PR#11） |
| `DATA_SETUP.md` | テストデータセットのセットアップガイド |
| `prompt/SPEC.md` | 製品仕様書（全体設計） |
| `PROJECT_SUMMARY.md` | このファイル |

---

## 実装計画

### PR構成（全11個）

| PR | 内容 | 見積工数 |
|----|------|---------|
| PR#1 | プロジェクト構造とCI/CD設定 | 1日 |
| PR#2 | Data Ingestion基本機能 | 2日 |
| PR#3 | Auto-Type Detection | 2日 |
| PR#4 | 金融ドメイン定義 | 1日 |
| PR#5 | Basic Evaluator（統計検定） | 2日 |
| PR#6 | Financial Stylized Facts | 3日 |
| PR#7 | Privacy評価 | 3日 |
| PR#8 | 評価スコアリング統合 | 2日 |
| PR#9 | CLIインターフェース | 2日 |
| PR#10 | レポート生成 | 3日 |
| PR#11 | 統合テストとドキュメント | 2日 |

**合計見積**: 約23日

### 開発原則

1. **TDD（テスト駆動開発）**: すべての機能はテストファーストで実装
2. **カバレッジ90%以上**: 単体テスト、統合テスト、E2Eテスト
3. **型安全性**: mypy strict modeでの型チェック
4. **コード品質**: black, flake8, bandit による品質管理
5. **CI/CD**: GitHub Actionsで自動テスト

---

## 技術スタック

### コア

- **Python**: 3.10+
- **データ処理**: pandas, numpy, scipy
- **統計/ML**: scikit-learn, statsmodels

### 開発ツール

- **テスト**: pytest, pytest-cov
- **品質**: black, flake8, mypy, bandit
- **依存関係**: Poetry
- **CI/CD**: GitHub Actions

### CLI/レポート

- **CLI**: typer, rich
- **レポート**: jinja2, matplotlib, seaborn
- **PDF生成**: weasyprint

---

## テストデータセット

### 使用データ

1. **Kaggle Credit Card Fraud Detection**
   - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - レコード数: 284,807件
   - 不正取引: 492件（0.172%）

2. **合成データ（評価用）**
   - Phase 1: 統計ベースの簡易生成
   - Phase 2以降: CTGAN/Copula

### 準備状況

- [x] データセットの選定
- [x] セットアップガイドの作成（`DATA_SETUP.md`）
- [ ] Kaggle APIのセットアップ（実装開始時）
- [ ] データのダウンロード（実装開始時）
- [ ] 合成データ生成スクリプトの実装（PR#1）

---

## 開発環境

### 確認済み事項

- **OS**: macOS (Darwin 24.6.0)
- **開発環境**: ローカル
- **Git**: GitHub（リポジトリ作成済み）
- **GitHub CLI**: 利用可能（v2.82.1）

### セットアップ手順

```bash
# リポジトリのクローン
git clone https://github.com/takurot/sfdao.git
cd sfdao

# Poetryのインストール（まだの場合）
curl -sSL https://install.python-poetry.org | python3 -

# PATHの設定（~/.zshrc または ~/.bash_profile に追加）
export PATH="$HOME/.local/bin:$PATH"

# 新しいターミナルセッションでPoetry確認
poetry --version

# 依存関係のインストール（PR#1で pyproject.toml 作成後）
poetry install

# 仮想環境の有効化
poetry shell

# macOS固有: WeasyPrint用の依存関係（PDF生成機能用）
brew install cairo pango gdk-pixbuf libffi

# テストの実行
pytest
```

### ブランチ戦略

**重要**: 各PRの作業は必ず専用のブランチを作成して実行してください。

```bash
# PR作業開始
git checkout -b feature/pr-01-project-setup

# 作業完了後
git add .
git commit -m "PR#1: プロジェクト構造とCI/CD設定"
git push -u origin feature/pr-01-project-setup

# PR作成
gh pr create --title "PR#1: プロジェクト構造とCI/CD設定" --body "..."

# PRマージ後、mainに戻る
git checkout main
git pull origin main
```

---

## Phase 1完了の定義

以下の条件をすべて満たした時点でPhase 1完了とします：

- [ ] CLIから評価レポートが出力できる
- [ ] 金融特化の評価指標が正しく計算される
- [ ] HTMLレポートが生成される
- [ ] テストカバレッジ90%以上
- [ ] ドキュメントが整備されている
- [ ] 1つ以上の実際の金融データセットで動作確認済み
- [ ] CI/CDパイプラインが正常に動作
- [ ] README、API Documentation完備

---

## 次のステップ（Phase 2以降）

Phase 1完了後、以下の機能拡張を実施予定：

### Phase 2: "The Generator & Logic"

- Hybrid Generator実装（CTGAN, TVAE, Copula, GARCH）
- Constraint & Logic Guard（会計恒等式チェック）
- Scenario Injection（ストレステスト）
- LLMベースのテキスト生成（取引摘要）

### Phase 3: "The Optimizer"

- Rule Learning Engine（強化学習ベース）
- Auto-Tuning Mode（自律的品質改善）
- オーケストレーター（ワークフロー自動化）
- Web UI（React/Next.js）

---

## 実装開始チェックリスト

実装を開始する前に、以下を確認してください：

### 環境準備

- [x] GitHubリポジトリの作成
- [x] README、LICENSE、.gitignoreの作成
- [x] 実装計画書の作成
- [x] データセットセットアップガイドの作成
- [ ] Poetryのインストール確認
- [ ] Python 3.10+のインストール確認
- [ ] Kaggle APIのセットアップ

### データ準備

- [ ] Kaggle Credit Card Fraud Detectionのダウンロード
- [ ] `tests/fixtures/` へのサンプルデータ配置
- [ ] 合成データ生成スクリプトの作成と実行

### PR#1開始前の準備

- [ ] ブランチ戦略の確認（main, develop, feature/PR-XXX）
- [ ] コミットメッセージ規約の確認
- [ ] コードレビュー体制の確認

---

## 連絡先・サポート

### リソース

- **GitHub Issues**: バグ報告、機能要望
- **GitHub Discussions**: 質問、アイデア共有
- **Pull Requests**: コントリビューション

### 参考資料

- [Poetry Documentation](https://python-poetry.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
- [CTGAN Paper](https://arxiv.org/abs/1907.00503)
- [SDV Documentation](https://sdv.dev/)

---

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2025-11-29 | プロジェクト初期セットアップ完了 |
| 2025-11-29 | GitHubリポジトリ作成 (https://github.com/takurot/sfdao) |
| 2025-11-29 | 実装計画書、データセットガイド作成 |

---

**ステータス**: ✅ 準備完了（実装開始可能）

実装開始の指示をお待ちしています。
