# タスク実行プロンプト

@docs/PLAN.md を参照して、指定されたPR（例: PR2）の実装を進めてください。

## 実装フロー

### 1. ブランチ作成
- `feature/<pr番号>-<簡潔な説明>` の形式でブランチを作成
  - 例: `feature/pr-02-data-ingestion`
- `main` ブランチから分岐すること

### 2. TDD（テスト駆動開発）で実装
- **Red**: まず失敗するテストを書く
- **Green**: テストが通る最小限のコードを実装
- **Refactor**: コードを整理・改善
- ユニットテストと統合テストの両方を作成
- E2Eテストも必要に応じて追加し、`tests/e2e/` に配置
- テストは `tests/unit/` と `tests/integration/` に配置

### 3. テストの実行
```bash
# 全テストを実行（カバレッジ付き）
pytest

# 特定のテストファイルのみ実行
pytest tests/unit/ingestion/test_loader.py

# カバレッジレポート（HTML形式）
pytest --cov=sfdao --cov-report=html

# 詳細表示付き
pytest -v
```
- 既存のテストが壊れていないことを確認
- 新機能のテストを追加
- カバレッジ90%以上を目標に

### 4. コード品質の確認
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

### 5. PLAN.md の更新
- 実装したPRのステータスを `✅` に更新
- 実装内容の要約を **進捗**: セクションに追記
- テストの詳細を `Tests:` セクションに追記

### 6. コミット & プッシュ
- コミットメッセージ形式: `<type>(<scope>): <description>`
  - type: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
  - 例: `feat(ingestion): add CSV loader with schema extraction`
- 適切な粒度でコミットを分割

### 7. Pull Request 作成
```bash
gh pr create --title "PR#<番号>: <タイトル>" --body "<説明>"
```
- PRテンプレートに従って記述
- 関連するIssueやPRをリンク

### 8. CI結果の確認と対応
```bash
gh pr checks   # CIの状態確認
gh run list    # ワークフロー一覧
gh run view    # 詳細確認
```
- CIが失敗した場合は原因を特定して修正
- 全てのチェックがパスするまで繰り返す

## チェックリスト

- [ ] ブランチを `main` から作成した
- [ ] テストを先に書いた（TDD）
- [ ] 全てのテストがパスする（`pytest`）
- [ ] フォーマットチェックがパス（`black --check .`）
- [ ] Lintチェックがパス（`flake8 .`）
- [ ] 型チェックがパス（`mypy sfdao`）
- [ ] セキュリティチェックがパス（`bandit -r sfdao`）
- [ ] カバレッジ目標達成（90%以上）
- [ ] PLAN.md を更新した
- [ ] コミットメッセージが適切
- [ ] PRを作成した
- [ ] CIが全てパス

## 注意事項

- 既存のテストを壊さないこと
- 依存関係のあるPRがマージされていることを確認
- Pydantic v2を使用すること（データバリデーション）
- 型ヒントを必ず記述すること（mypy strict mode対応）
- macOS環境での開発を想定（`._*`、`.DS_Store`は.gitignoreで除外済み）

## プロジェクト構成

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
│   ├── e2e/                # End-to-Endテスト
│   └── fixtures/           # テストデータ
└── prompt/                 # 仕様書・計画書
```

## 開発コマンドまとめ

```bash
# 依存関係インストール
poetry install

# 仮想環境有効化
poetry shell

# テスト実行
pytest

# 品質チェック（一括）
black --check . && flake8 . && mypy sfdao && bandit -r sfdao

# フォーマット適用してからテスト
black . && pytest
```
