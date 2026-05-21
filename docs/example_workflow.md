# CTA Research 範例工作流

這份文件把專案目前的功能，對應到「因子實戰、開源策略解析、Qlib 與鏈上數據」的研究路線。它不是實盤交易流程，而是研究、回測、驗證與資料輸出的操作指南。

## 1. 安裝

```powershell
python -m pip install -e ".[dev]"
```

## 2. 準備 OHLCV 資料

每個交易標的放一個 CSV 檔到資料目錄中：

```text
timestamp,open,high,low,close,volume
```

可以先用 `configs/example.yaml` 當最小可執行回測設定。YAML 裡的相對資料路徑會以設定檔所在位置為基準解析。

## 3. 執行研究回測

```powershell
python -m cta_research.cli configs/example.yaml --output-dir runs
```

每次執行會產生一個新的 run 目錄，內容包含：

- 回測檔案：`equity_curve.csv`、`positions.csv`、`trades.csv`、`metrics.json`
- 策略歸因：`strategy_returns.csv`
- 因子研究：`factor_ic.csv`、`factor_ic_summary.csv`、`factor_correlation.csv`、`momentum_quantile_returns.csv`
- Qlib 銜接：`qlib_ohlcv.csv`、`qlib_alpha360_like.csv`
- HTML 總覽：`report.html`

## 4. 加入市場資料下載器

使用 `cta_research.downloaders` 抓取交易所研究資料，並快取到本機：

```python
from cta_research.downloaders import BinanceUsdmClient, write_ohlcv_cache

client = BinanceUsdmClient()
frame = client.fetch_ohlcv("BTCUSDT", "1d", limit=1000)
write_ohlcv_cache(frame, "data/ohlcv", "BTCUSDT")
```

下載器模組也會正規化 funding rate 與 open interest 資料，方便後續做替代因子或市場結構因子研究。

## 5. 加入鏈上特徵

離線鏈上特徵依照 feature name 與 symbol 存放：

```text
onchain/sopr/BTCUSDT.csv
onchain/nupl/BTCUSDT.csv
onchain/mvrv/BTCUSDT.csv
```

每個檔案格式為：

```text
timestamp,value
```

接著用 `load_onchain_feature_directory(...)` 載入，並把結果傳給 `calculate_factor_set(..., onchain_data=...)`。這樣 `alternative` 策略袖就能使用鏈上綜合因子。

## 6. 先做因子准入

任何完整策略都要先通過單因子驗證。每個因子至少要檢查：

- IC Mean
- IR
- positive rate
- quantile long-short spread
- turnover
- 單因子回測

CLI run 會輸出：

- `factor_scorecard.csv`
- `factor_quantile_returns.csv`
- `factor_single_backtests.csv`

只有通過准入的因子，才應該進入多因子組合。弱因子不要用 ML 硬救。

## 7. 驗證穩健性

在把任何策略想法升級之前，先用 `cta_research.validation` 做：

- Walk-forward analysis
- Monte Carlo return paths
- Parameter stability report
- Overfitting checks

這一步的目的不是追求單次漂亮績效，而是確認策略是否能跨期間、跨參數維持合理表現。

## 8. 建立機器學習 Baseline

使用 `cta_research.ml` 可以把現有因子轉成 supervised dataset，先做簡單模型驗證：

- Linear Regression
- Ridge
- Random Forest

第一階段目標是檢查特徵是否有預測力，例如：

- directional accuracy 是否高於隨機
- rank IC 是否穩定
- model signal backtest 是否比規則策略更穩

深度模型如 LSTM、Transformer 不建議一開始就上，應該等線性與樹模型 baseline 有正面訊號後再做。

## 9. 匯出到 Qlib

每次 CLI run 都會輸出 Qlib-friendly CSV：

- `qlib_ohlcv.csv`：長表格式的 OHLCV panel
- `qlib_alpha360_like.csv`：受 Alpha360 啟發的 rolling feature 實驗表

目前這是 Qlib 銜接資料層，不要求本機一定安裝 Qlib。後續若要接正式 Qlib workflow，可以從這兩個 CSV 開始轉成 Qlib dataset。
