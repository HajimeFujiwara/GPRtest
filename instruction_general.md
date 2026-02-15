# 一般的な開発指示書 (General Instructions)

このドキュメントは、GitHub Copilot Agent Mode 等の「エージェントによる自動実装」を前提とした、一般的な開発ルールと構成を定義します。  
特に、科学技術計算・データ分析プロジェクトにおける **再現性・保守性・検証可能性** を重視します。

---

## 0. 指示の優先順位 (Precedence)

指示が矛盾する場合、以下の優先順位に従います。

1. `instruction.md`（最終設計書）
2. `instruction_org.md`（プロジェクト固有要件）
3. `instruction_general.md`（本ファイル：一般ルール）

矛盾を発見した場合、エージェントは **`instruction.md` に「採用した判断」と「理由」** を明記してから実装に進みます。

---

## 1. 開発フロー (Development Workflow)

本プロジェクトでは、以下の2段階プロセスで開発を進めます。

### 1.1 指示書の作成 (Instruction Generation)
- `instruction_org.md`（自然言語による要件メモ）と本ファイル（`instruction_general.md`）を元に、エージェントが詳細設計書 `instruction.md` を作成します。
- 曖昧な要件は **具体的なAPI設計**（モジュール/関数/型/入出力仕様/例外/テスト戦略）に落とし込みます。
- `instruction.md` は「実装手順書」ではなく、**設計仕様（契約）** として記述します。

### 1.2 実装 (Implementation)
- 完成した `instruction.md` に基づき、エージェントがコード実装を行います。
- 実装中に設計上の不足や矛盾が判明した場合、先に `instruction.md` を修正し、仕様を確定してから実装を進めます。

### 1.3 完了条件 (Definition of Done)
以下を満たすことを「実装完了」とします。
- テストが通る（`pytest`）
- 型チェックが通る（`mypy` または `pyright`）
- フォーマット/静的解析が通る（`ruff` 推奨）
- 再現性（seed/rng方針）がドキュメント化され、実行結果が再現できる

---

## 2. ディレクトリ構成 (Project Structure)

プロジェクトのルートディレクトリを基準とし、以下の構成を遵守してください。

- `root/` (プロジェクトルート)
    - `data/`: データ格納（入力/中間/出力）
        - `raw/`: 入力（不変）。原則として加工しない。
        - `interim/`: 中間生成物（再生成可能）。
        - `processed/`: モデル入力等の整形済みデータ（再生成可能）。
        - `output/`: 図表・推定結果・レポート等の成果物（再生成可能だが保存することもある）。
    - `src/`: ソースコード（Pythonパッケージ）
        - `YOUR_PACKAGE_NAME/`:
            - `__init__.py`
            - 機能別モジュール（例: `io.py`, `preprocess.py`, `model.py`, `metrics.py` 等）
    - `notebook/`: 実行用Jupyter Notebook（実験・分析・結果の記録）
    - `tests/`: 単体テスト（`pytest`）
    - `requirements.txt`: 依存ライブラリ定義（ルートに配置）
    - `instruction_general.md`: 本ファイル
    - `instruction_org.md`: プロジェクト固有要件（要件メモ）
    - `instruction.md`: エージェントへの最終指示書（詳細設計書）
    - （推奨）`pyproject.toml`: ruff/mypy等の設定集約
    - （推奨）`README.md`: 実行手順・データ取得手順・再現方法

---

## 2.1 `data/` の運用ルール
- `data/raw` は不変（入力の原典）。破壊的変更を禁止します。
- `data/interim`, `data/processed`, `data/output` は再生成可能であることを原則とします。
- 大容量データ・機密データは **git管理しない**（`.gitignore` により除外）。必要な場合は取得手順・生成手順を `README.md` に記載します。

---

## 2.2 パッケージ化とインポート (Import Strategy)
- 原則: **`sys.path.append` 等のパス操作は禁止**（例外が必要なら `instruction.md` に理由を明記）。
- 推奨: `src` を Pythonパッケージとして扱い、開発時は **editable install** により notebook から利用します。
    - 例: `pip install -e .`
- import は **絶対import** を基本とし、プロジェクト内の参照を一貫させます。

---

## 3. コーディングルール (Coding Standards)

### 3.1 基本ルール
- **機能実装（`src/`）**
    - 機能は「モジュール単位」で分割します（例: I/O、前処理、モデル、評価、可視化、ユーティリティ等）。
    - 状態（データ構造）は `@dataclass` を優先し、可能な限り `frozen=True` を設定して不変性を確保します。
    - 計算ロジックは **副作用の少ない関数** として実装し、状態とロジックを分離します。
    - クラス（メソッド中心）は、状態管理や戦略差し替え等が明確に必要な場合に限定します。

- **実行と分析（`notebook/`）**
    - Notebook は「実行結果の記録」と「試行錯誤」の場です。
    - 再利用可能なロジックは notebook に書かず、`src/` に移管します。
    - 最終成果物として提出する際は、上から順に実行してエラーなく動作することを確認します（Restart & Run All）。

- **テスト（`tests/`）**
    - 主要な計算ロジックに対して `pytest` による単体テストを作成し、品質を担保します。

- **型指定（Type Hinting）**
    - **必須**: すべての関数・メソッドの引数と戻り値に型ヒントを付与します。
    - 例: `def process_data(df: pd.DataFrame) -> pd.DataFrame:`

- **Docstrings**
    - クラス・関数には Docstring を記述し、目的、引数、戻り値、例外、前提条件（assumptions）を明示します。
    - 形式は NumPyスタイルまたはGoogleスタイルを推奨します。

- **ログ**
    - `print` ではなく `logging` を使用します（デバッグ/監査/再現性の観点）。
    - ログレベル（DEBUG/INFO/WARNING/ERROR）を適切に使い分けます。

- **パスの扱い**
    - ファイルパスは `pathlib` を使用し、OS差異を吸収します。

---

### 3.2 科学技術計算のベストプラクティス
- **ベクトル化と性能**
    - forループを無条件に禁止しません。
    - まず正しさと可読性を確保し、ボトルネックが確認できた箇所をベクトル化（NumPy/pandas）や最適化（必要ならNumba等）で改善します。
    - 巨大配列の一括生成によるメモリ増大に注意します。

- **再現性（Reproducibility）**
    - 乱数は `numpy.random.Generator` を使用し、`np.random.default_rng(seed)` により生成します。
    - 関数は `rng: np.random.Generator | None = None` を受け取り、`None` の場合のみ内部生成します。
    - グローバル乱数状態（例: `np.random.seed()`）への依存は避けます。
    - seed は設定クラスや設定ファイル等により外部から制御可能にします。

- **データの不変性（Immutability）**
    - 関数内で `DataFrame` 等を変更する場合は、原則として `.copy()` を作成してから操作し、副作用を回避します。
    - ただしメモリが支配的な場合は例外を認め、その場合は関数のDocstringに「破壊的変更」を明記します。

- **数値安定性**
    - dtype（float32/float64等）方針、スケーリング、正則化、安定化手法（例: clipping, log-sum-exp 等）を必要に応じて明記します。
    - 例外・警告（オーバーフロー、ゼロ除算、NaN/Inf）に対して方針（検出/補正/停止）を定義します。

---

### 3.3 入力検証 (Validation)
- I/O境界（データ読み込み直後、外部API入力、CLI入力）で以下を検証します。
    - スキーマ（列名、dtype）
    - 欠損・外れ値の扱い
    - shape/次元、単位、許容範囲
- 計算コアは「検証済み入力」を前提として単純化し、前提条件は Docstring に明記します。
- 不正入力は必要に応じて独自例外（例: `DataSchemaError`）で区別します。

---

### 3.4 テスト (Testing)
- **浮動小数の比較**
    - `numpy.testing.assert_allclose` を用い、`rtol/atol` をテストごとに明記します。
    - 単純な `==` 比較は原則禁止です（整数・厳密一致が仕様である場合を除く）。

- **乱数を用いるテスト**
    - seed/rng を固定します。
    - 統計的性質のテストは十分なサンプル数を確保し、許容誤差を設計します。

- **重いテスト**
    - 実行時間が長いテストは `@pytest.mark.slow` 等で分離し、通常CIでは除外/選択実行できるようにします。

---

### 3.5 依存ライブラリの管理 (Dependency Management)
- **一元管理**
    - 外部ライブラリはすべてルートの `requirements.txt` で管理します。
    - 環境構築は `pip install -r requirements.txt` を前提とします。

- **更新の徹底**
    - 新たなライブラリが必要になった場合、コード実装と同時に `requirements.txt` を更新します。

- **Python/バージョン固定方針**
    - Python バージョンを明記します（原則: Python 3.11。変更する場合は `instruction_org.md` に明記）。
    - `requirements.txt` は再現性のため、原則としてバージョン固定（`==`）または上限付き（`<`）で管理します。
    - 必要に応じて `pip freeze` 相当のロック情報を生成する手順を `instruction.md` または `README.md` に記載します。

---

### 3.6 品質ゲート (Quality Gates)
以下のツールチェーンを推奨し、エージェントはこれを満たすことを前提に実装します。
- formatter: `ruff format`（または `black` を採用する場合はプロジェクトで統一）
- linter: `ruff`
- type check: `mypy` または `pyright`
- test: `pytest`

（推奨）CI（GitHub Actions 等）で上記を自動実行し、品質を継続的に担保します。

---

## 4. 推奨事項 (Best Practices)

- **設定とパラメータの分離**
    - パラメータ（試行回数、閾値、学習率等）をハードコーディングしません。
    - 設定ファイル（YAML/JSON）または設定クラス（dataclass）で管理します。

- **エラーハンドリング**
    - 予期せぬエラーに対処するため、適切な例外処理（`try-except`）と、エラー原因が特定可能なメッセージを実装します。
    - 例外を握り潰さず、必要なら上位へ伝播させます。

- **ドキュメント**
    - 主要なアルゴリズムや数式の根拠、参照文献、仮定（assumptions）を `README.md` またはモジュールDocstringに記載します。

---

## 5. `instruction.md` に含めるべき最低限

`instruction.md` は少なくとも以下を含めます。

- 目的/スコープ（やらないこと含む）
- 入力データ仕様（列名、dtype、欠損許容、単位、shape）
- 出力仕様（ファイル、指標、図表、保存形式）
- 主要 API 一覧（モジュール、関数シグネチャ、戻り値型、例外）
- 数値上の注意（許容誤差、安定化、dtype方針）
- 再現性方針（seed/rng、バージョン記録）
- テスト計画（高速/低速、主要エッジケース）
- 実行手順（notebook、CLI、再現コマンド）
- データ運用（raw/interim/processed/output、git管理境界、生成手順）

---
