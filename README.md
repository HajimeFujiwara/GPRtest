# GPRtest
Gaussian Process Regressionのお試し

## セットアップ
- Python 3.11 を想定
- 依存インストール:

```bash
pip install -r requirements.txt
```

## Notebook実行
- `notebook/lgbm_forecast_validation.ipynb` を上から順に実行
- 入力Excelパス、`CUTOFF_DATE`、`HORIZON_N`、`TASK_MODE` を設定セルで変更
- 予測結果は `data/output/` 配下に保存
