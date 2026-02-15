# Copilot Agent Mode 指示書：LightGBMでn期先予測モデル（回帰/分類）を検証するNotebook作成

## 目的
次の仕様を満たす **Jupyter Notebook（.ipynb）をPythonで作成**せよ。

- 入力：特徴量Excel、ターゲット（非説明変数）Excel（2種類：変化率/変化フラグ）、LightGBMパラメータExcel
- 任意に指定する **基準日（cutoff_date）までのデータで学習**し、**date > cutoff_date** の期間を予測・評価する
- **n期先予測**：時刻 t の特徴量から **t+n のターゲット**を予測する（特徴量とターゲットの対応をシフトして整列）
- 変化率：回帰（LGBMRegressor）
- 変化フラグ（-2,-1,1,2）：分類（LGBMClassifier）
- 学習後、**date > cutoff_date** の予測を算出し、実績と予測から **混同行列**を作成する  
  - 回帰モデル：予測リターン > 0 を 1、< 0 を -1 にフラグ化（0 は 1 扱いでよい。明示すること）
  - 分類モデル：予測フラグ（-2,-1,1,2）を出力
- Excelの空白セルは **欠損として保持**（埋めない・補完しない・勝手に削除しない）。pandas上では通常 `NaN` になるので、そのまま保持する。

---

## 前提条件（実行環境）
- Python 3.11 を想定
- 必須ライブラリ：`pandas, numpy, openpyxl, lightgbm, scikit-learn, matplotlib`
- 入力はExcel（`.xlsx`）を前提とし、既定で先頭シートを読む
- Notebookは `notebook/` 配下に作成し、再利用可能な処理は将来的に `src/` へ移管しやすい関数設計にする

---

## 入力ファイル仕様（厳守）
### 1) 特徴量データ（Excel）
- 1列目：日付（空白なし）
- 2列目以降：特徴量数値（空白あり）
- 1行目：ヘッダー

### 2) 非説明変数データ（Excel）
- 1列目：日付（空白なし）
- 2列目：ターゲット値（空白あり）
- 1行目：ヘッダー
- 変化率用と変化フラグ用で **別々のExcel**

### 3) LightGBMパラメータ（Excel）
- 1列目：パラメータ名
- 2列目：パラメータ値
- 1行目：ヘッダー

---

## Notebookの要求仕様（成果物）
### A. Notebookの構成（セル構造）
以下の順で、見出し（Markdownセル）とコードセルを作ること。

1. **概要 / 使い方**
   - このNotebookが何をするか
   - 入力ファイルと設定（cutoff_date, horizon_n, mode）をどこで変えるか

2. **環境・依存関係**
   - import
   - バージョン表示（pandas, numpy, lightgbm, sklearn）
   - 依存：`pandas, numpy, openpyxl, lightgbm, scikit-learn, matplotlib`

3. **設定（ユーザが触る場所を1セルに集約）**
   - `FEATURES_XLSX_PATH`
   - `TARGET_RETURN_XLSX_PATH`（変化率）
   - `TARGET_FLAG_XLSX_PATH`（変化フラグ）
   - `LGB_PARAMS_XLSX_PATH`
   - `DATE_COL_NAME`（未指定なら自動で「先頭列」を使う実装でも可）
   - `CUTOFF_DATE`（例："2024-12-31" のようなISO文字列）
   - `HORIZON_N`（int）
   - `TASK_MODE`：`"regression_return"` or `"classification_flag"`
   - `RANDOM_SEED`

4. **入出力ユーティリティ関数**
   - `load_features_excel(path) -> pd.DataFrame`
   - `load_target_excel(path) -> pd.Series`
   - `load_lgb_params_excel(path) -> dict`
   - 仕様：
     - 1列目を日付として `pd.to_datetime`、昇順ソート、**重複日付があれば `ValueError` で停止**
     - 空白セルは欠損（`NaN`）として保持し、埋めない
     - featuresは `DataFrame`（index=DatetimeIndex）
     - targetは `Series`（index=DatetimeIndex、nameは列名）
   - パラメータExcelは `{"param_name": parsed_value}` のdictへ
     - 値の型変換：`ast.literal_eval` を優先し、失敗したら文字列のまま
     - 空欄は無視

5. **データ読み込みと整合性チェック**
   - shape、先頭/末尾日付、欠損率（列ごと）を表示
   - featuresとtargetの日付範囲の交差（intersection）を確認
   - 以降のモデリングは基本的に **日付でinner join** して整列（方針を明示）

6. **n期先予測用のアラインメント（最重要）**
   - `build_aligned_dataset(X, y, horizon_n) -> (X_aligned, y_aligned)`
   - 定義：
     - 特徴量は t、ターゲットは t+horizon_n
     - 実装は例として：
       - `y_shifted = y.shift(-horizon_n)`（indexはtのまま、値がt+hの実績）
       - `df = X.join(y_shifted, how="inner")`
     - 学習に使う行は **ターゲットが欠損でない行のみ**（features欠損はLightGBMが扱えるので残す）
   - cutoff_dateによる分割：
     - 学習：`date <= cutoff_date` かつ `y_shifted` 非欠損
     - 予測・評価：`date > cutoff_date`（評価は実績が存在する行のみ）

7. **モデル学習（LightGBM）**
   - 共通：
     - パラメータExcelから読み込んだdictを使用
     - 乱数seed固定（`random_state`）
   - 回帰（変化率）：
     - `lgb.LGBMRegressor(**params)`
     - objectiveが入っていない場合は `objective="regression"` を補完
     - `multiclass`, `num_class` が含まれる場合は警告して無視
   - 分類（変化フラグ）：
     - クラスは {-2,-1,1,2} の4クラス
     - `LabelEncoder` か `mapping = {-2:0, -1:1, 1:2, 2:3}` を用意して学習
     - `lgb.LGBMClassifier(**params)`
     - objectiveが入っていない場合は `objective="multiclass"` と `num_class=4` を補完
     - 学習データに4クラスが揃わない場合は **Warningを出して学習継続** する

8. **予測・評価（混同行列）**
   - 回帰：
     - `y_pred_return = model.predict(X_test)`
     - フラグ化：`pred_flag = np.where(y_pred_return >= 0, 1, -1)`
     - 実績フラグ：`actual_flag = np.where(y_test_return >= 0, 1, -1)`（欠損行は除外）
     - 混同行列（labels=[-1,1]）
     - 追加でRMSEも出してよい（ただし主評価は混同行列）
   - 分類：
     - `pred_class = model.predict(X_test)`
     - `pred_flag = inverse_mapping[pred_class]`
     - 混同行列（labels=[-2,-1,1,2] で固定。欠落クラスがあっても順序は固定）
   - 表示：
     - `sklearn.metrics.confusion_matrix`
     - `ConfusionMatrixDisplay(...).plot()`
     - accuracy、macro F1 なども併記（任意だが推奨）

9. **結果テーブル出力**
   - DataFrame：`date, y_true, y_pred_raw, y_pred_flag`（分類はraw=class_idでも可）
   - `date > cutoff_date` の行を保存：
     - 保存先は `data/output/` 固定
     - ファイル名は `pred_{TASK_MODE}_h{HORIZON_N}_cutoff{YYYYMMDD}.csv` とする
   - 混同行列も画像保存（任意）

10. **再現性・注意点**
   - 欠損は埋めない
   - horizonとcutoffの取り方で「学習に使える最終日」が実質的に `cutoff_date - horizon` になる点を説明

---

## 実装上の注意（重要）
- **空白は欠損として保持**：勝手に `fillna` しない。`dropna` は **ターゲット欠損の行だけ**に限定。
- 日付列は必ず `datetime64[ns]` 化し、indexとして扱う。
- 予測対象は「`date > cutoff_date`」の特徴量行。ただし評価は実績がある行のみ。
- 変化フラグは4クラス分類。ラベル変換と逆変換を必ず実装。
- パラメータExcelは型が混在しうるので、`ast.literal_eval` によるパースを実装して堅牢化する。
- エラー時に原因が分かる例外メッセージを出す（ファイル未存在、列数不足、日付重複など）。

---

## 受け入れ条件（完了の定義）
- Notebookを上から順に実行して、**エラーなく**以下が出力されること：
  1) 読み込んだfeatures/target/paramsの概要
  2) horizonシフト後の学習・テスト件数
  3) 学習済みモデル
  4) cutoff_date以降の予測結果テーブル（先頭数行表示）
  5) 混同行列（図と数値）
  6) 予測結果ファイルの保存

---

## コード品質要件
- 関数化して見通しよく（上記ユーティリティ＋build_aligned_datasetは必須）
- 型ヒントを可能な範囲で付与（`-> pd.DataFrame` 等）
- 変数名は `snake_case`
- ログは `print` でもよいが、重要な中間情報（件数、欠損率、日付範囲）は必ず出す

---

## Notebookファイル名
- `lgbm_forecast_validation.ipynb`
