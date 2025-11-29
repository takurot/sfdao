# Product Specification: SFDAO (Synthetic Finance Data Auditor & Optimizer)

**Version:** 1.0.0
**Status:** Draft (Finalized for Development)
**Target:** Enterprise (B2B SaaS / On-premise)

---

## 1. プロダクト概要

### 1.1 プロダクト名
**SFDAO (Synthetic Finance Data Auditor & Optimizer)**
*(キャッチコピー: 金融コンプライアンス準拠・合成データ品質保証プラットフォーム)*

### 1.2 コアバリュー（差別化要素）
1.  **Auditor First**: 生成機能のおまけではなく、「金融監査・規制（Basel III, GDPR, AML等）への適合性」を判定する**監査機能**を主軸とする。
2.  **Rule-based Feedback Loop**: 評価結果に基づき、生成パラメータの調整を**強化学習的アプローチ**で自動化・半自動化する。
3.  **Hybrid Generation Strategy**: 統計モデル（CTGAN等）、数理ファイナンスモデル（Copula/GARCH）、LLMを組み合わせ、数値・時系列・テキスト（摘要）の複合データを生成する。
4.  **Scenario Injection**: 単なる分布再現だけでなく、特定の経済ショックや不正シナリオを**意図的に注入**する機能を持つ。

### 1.3 ターゲットユーザー
* **金融機関（銀行・証券・暗号資産交換業）**: リスク管理部、データ分析部、システム監査室
* **FinTechベンダー**: 決済代行、AMLソリューションプロバイダー
* **監査法人・コンサルティングファーム**: モデルリスク管理（MRM）の監査ツールとして

---

## 2. システムアーキテクチャ

システムは以下の6つのモジュールで構成されるマイクロサービスアーキテクチャを想定する。

1.  **Data Ingestion & Schema Manager**: データ取り込みと構造定義
2.  **Hybrid Generator Engine (Model Zoo)**: 多様なモデルによる生成
3.  **Constraint & Logic Guard**: 金融・会計ロジックによるフィルタリング
4.  **Advanced Evaluator (The Auditor)**: 多角的なスコアリング
5.  **Rule Learning & Optimization Core**: 改善ルールの学習と推論
6.  **Orchestrator & Dashboard**: ワークフロー制御と可視化

---

## 3. 機能要件詳細

### 3.1 Data Ingestion & Schema Manager

#### 3.1.1 データインポート
* 形式: CSV, Parquet, JSON (Web3 transaction logs)
* 機能:
    * **Auto-Type Detection**: 数値、カテゴリ、日時、PII（個人特定情報）、フリーテキスト（摘要）の自動判定。
    * **Relationship Mapping**: テーブル間の外部キー結合（ER図）の定義。

#### 3.1.2 ドメイン定義（金融特化）
* カラムの役割定義: `ID`, `Transaction_Amount`, `Balance`, `Timestamp`, `Counterparty`, `Description`
* エンティティ設定: 顧客（Node）と取引（Edge）のグラフ構造定義（AML用）。

### 3.2 Hybrid Generator Engine (Model Zoo)

単一モデルではなく、データ特性に応じた最適モデルを選択・組み合わせる。

* **Deep Learning Models**:
    * CTGAN / TVAE: 一般的なTabularデータ用。
    * TimeGAN / DoppleGANger: 時系列データ用。
* **Statistical/Financial Models**:
    * Gaussian Copula: 変数間の相関構造を厳密に維持したい場合。
    * GARCH / VAR: ボラティリティクラスタリング（金融時系列特有の変動）の再現。
* **LLM-based Generator**:
    * Transformer Models: 取引摘要（Description）、コールセンターログ、本人確認（KYC）備考欄の生成。
* **Differential Privacy Mode**:
    * DP-SGD等を適用し、数学的なプライバシー予算（$\epsilon$）を指定した生成。

### 3.3 Constraint & Logic Guard (金融整合性チェック)

統計的には正しくても「ビジネスとしてあり得ない」データを排除・補正する。

* **Accounting Identities (会計恒等式)**:
    * 資産 = 負債 + 純資産
    * 取引後の残高 = 取引前の残高 + 入金額 - 出金額
* **Temporal Logic (時間論理)**:
    * 口座開設日 $\le$ 初回取引日
    * 顧客の年齢 $\ge$ 18 (成人向け商品の場合)
* **Business Rules**:
    * 送金限度額チェック、通貨コードの整合性。

### 3.4 Advanced Evaluator (The Auditor)

#### 3.4.1 Quality (統計的忠実度)
* **Basic Stats**: KS検定, Jensen-Shannon Divergence。
* **Financial Stylized Facts**:
    * **Fat Tail Check**: 収益率分布の裾の厚さが実データと一致するか。
    * **Volatility Clustering**: 「変動が大きい期間」の持続性。

#### 3.4.2 Privacy (安全性)
* **Re-identification Risk**: 最近傍距離（Distance to Closest Record）分布。
* **Attribute Inference**: 攻撃モデルによる属性推定成功率。
* **Membership Inference**: 「学習データに使われたか」の判定耐性。

#### 3.4.3 AML / Utility (実用性・パターン)
* **Typology Matching**: 既知のマネロン手口（Structuring, Layering）が検知可能な状態で含まれているか。
* **Graph Metrics**: 取引ネットワークの次数分布、クラスタ係数。
* **Downstream Task**: XGBoost/LightGBM等での分類精度（Real vs Synthetic）。

### 3.5 Rule Learning & Optimization Core (差別化の中核)

評価結果に基づき、Generatorのパラメータ $\theta$ を最適化する。

#### 3.5.1 学習データ構造
* **State ($s_t$)**: 現在の評価スコアベクトル、データのメタ特徴量（歪度、欠損率）。
* **Action ($a_t$)**: パラメータ変更操作。
    * 例: `increase_batch_size`, `apply_log_transform`, `switch_model(CTGAN->Copula)`, `inject_noise(tail)`.
* **Reward ($r_t$)**: 総合スコア（Quality + Utility - Privacy Penalty）の増減分。

#### 3.5.2 推論と説明性 (Explainability)
* **Recommendation**: 「現在の状態 $s_t$ に対し、アクション $a_t$ を推奨（信頼度 85%）」。
* **Audit Trail (監査証跡)**: 「過去の類似ケース（N=50）において、このアクションはQualityを平均5%改善しました」という根拠テキストを生成。

### 3.6 Orchestrator & Dashboard

#### 3.6.1 動作モード
1.  **Audit Mode**: 既存データの評価レポート出力のみ（Entry Tier）。
2.  **Scenario Injection Mode**:
    * 「金利が2%急騰」「特定セグメントの倒産率5倍」等の外部ショック変数を強制入力して生成。ストレステスト用。
3.  **Auto-Tuning Mode**: ルールエンジンを用いた自律的な品質改善ループ。

#### 3.6.2 レポーティング
* PDF/HTML形式での監査レポート出力。
* 「改善の履歴（Before/After）」と「なぜそのパラメータを採用したか」の技術的根拠。

---

## 4. スコアリングと最適化ロジック

目的関数 $J(\theta)$ を最大化するパラメータ $\theta$ を探索する。

$$
J(\theta) = w_Q \cdot S_{Quality} + w_U \cdot S_{Utility} + w_{AML} \cdot S_{AML} - \lambda \cdot \max(0, P_{target} - S_{Privacy})
$$

* $w$: 各スコアの重み（ユーザー設定可能）。
* $\lambda$: 制約違反に対するペナルティ係数。
* $P_{target}$: 最低限必要なプライバシースコア。

---

## 5. ロードマップ (実装フェーズ)

### Phase 1: "The Auditor" (MVP - 診断ツールとしての確立)
* **目標**: 既存データに対する「評価レポート」だけで価値を出す。
* **機能**: Ingestion, Basic Evaluator, Financial Stylized Facts Check, レポート出力。
* **対象**: 既に何らかのデータを持つ金融機関・分析会社。

### Phase 2: "The Generator & Logic" (生成と整合性)
* **目標**: 高品質かつ矛盾のないデータを生成する。
* **機能**: Hybrid Generator (CTGAN + Rules), Constraint & Logic Guard, Scenario Injection (手動設定)。
* **対象**: POC用のデータが必要な開発部門。

### Phase 3: "The Optimizer" (自動化とAI化)
* **目標**: 専門家不在でも高品質データを自律生成する。
* **機能**: Rule Learning Engine, Feedback Orchestrator, Auto-Tuning Mode, LLMによるテキスト生成統合。
* **対象**: 全社的なデータ基盤としての導入。

---

## 6. 技術スタック案

* **Backend**: Python (FastAPI), Rust (計算負荷の高い評価指標計算用)
* **ML Core**: PyTorch, SDV (Synthetic Data Vault), Hugging Face (LLM)
* **Database**: PostgreSQL (メタデータ), Vector DB (類似性検索・ルール検索)
* **Frontend**: React / Next.js, Recharts / Plotly (可視化)
* **Infrastructure**: Docker, Kubernetes (学習ジョブの並列実行)

---

## 7. ユーザーへのアクションアイテム (Next Step)

この仕様書を基に、以下のステップを提案します。

1.  **Phase 1 (Auditor) のプロトタイプ作成**:
    * まずは「CSVを入れたら、金融特化の品質レポートが出る」CLIツールを作成しませんか？
    * 必要な評価指標（Financial Stylized Facts）の具体的な数式リストを用意できます。
2.  **ドメインデータの準備**:
    * テスト用のオープンな金融データセット（KaggleのCredit Card Fraud Detection等）を選定し、評価機能をテストする準備をします。