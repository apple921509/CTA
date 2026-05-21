# CTA Research

CTA Research 是一個研究用途的加密貨幣 CTA 回測框架，設計方向對應「因子實戰、開源策略解析、Qlib 與鏈上數據」這份進階指南。

本專案只做研究、回測、因子分析與資料輸出，不做實盤交易。

## 安裝

```powershell
python -m pip install -e ".[dev]"
```

## 執行範例

```powershell
python -m cta_research.cli configs/example.yaml --output-dir runs
```

真實 Binance spot 日線回測設定：

```powershell
python -m cta_research.cli configs/crypto_binance_spot_1d.yaml --output-dir runs
```

這份設定預期資料位於 `data/ohlcv/binance_spot_1d/`。`data/` 目錄已被 `.gitignore` 排除，適合存放本機下載的真實行情資料。

每次執行會在 `runs/` 底下建立新的 run 目錄，輸出：

- `config.yaml`
- `equity_curve.csv`
- `positions.csv`
- `trades.csv`
- `strategy_returns.csv`
- `factor_ic.csv`
- `factor_ic_summary.csv`
- `factor_correlation.csv`
- `momentum_quantile_returns.csv`
- `qlib_ohlcv.csv`
- `qlib_alpha360_like.csv`
- `metrics.json`
- `report.html`

完整操作流程請看 [docs/example_workflow.md](docs/example_workflow.md)。

## 輸入資料

每個交易標的一個 CSV。必要欄位：

```text
timestamp,open,high,low,close,volume
```

YAML 設定檔中的相對資料路徑，會以該設定檔所在位置為基準解析。

## 市場資料下載器

`cta_research.downloaders` 提供研究資料下載與正規化工具：

- Binance USD-M futures OHLCV、funding rate history、open interest history。
- OKX candles、funding rate history、open interest snapshot。

下載器輸出會正規化成 CSV 友善欄位。OHLCV 可透過 `write_ohlcv_cache(...)` 寫成本專案回測可直接讀取的資料格式。

## 因子研究

目前支援：

- 動量、均線斜率、Donchian、RSI、Bollinger z-score、ATR、歷史波動率、量能異常、VWAP deviation、價量背離。
- winsorization、z-score/rank normalization。
- IC / Rank IC、IC decay、factor correlation。
- factor orthogonalization。
- quantile forward return analysis。

## 鏈上與替代因子

`cta_research.onchain` 支援離線鏈上特徵研究：

- SOPR
- NUPL
- MVRV
- exchange net flow
- whale transaction count
- active addresses
- composite alternative factor

鏈上 feature CSV 格式：

```text
timestamp,value
```

建議目錄：

```text
onchain/sopr/BTCUSDT.csv
onchain/nupl/BTCUSDT.csv
onchain/mvrv/BTCUSDT.csv
```

載入後可傳給 `calculate_factor_set(..., onchain_data=...)`，並由現有 `alternative` 策略袖使用。

## 進階驗證與 Qlib

`cta_research.validation` 提供：

- walk-forward analysis
- Monte Carlo return-path simulation
- parameter stability report
- overfitting checks

`cta_research.qlib` 提供：

- Qlib-friendly OHLCV 長表輸出。
- Alpha360-like rolling feature table。

目前不強制安裝 Qlib；先提供穩定資料銜接層，之後可再加 native Qlib workflow。

## 機器學習 Baseline

`cta_research.ml` 提供第一版 supervised learning 研究流程：

- 將多因子面板轉成 supervised dataset。
- 支援 Linear Regression、Ridge、Random Forest baseline。
- 使用 expanding window 做 walk-forward 預測。
- 將模型預測轉成 cross-sectional long/short signal。
- 可直接跑 ML signal backtest。

這一層目前用來驗證特徵是否有穩定預測力，不建議直接視為可交易模型。深度模型如 LSTM、Transformer 會等 baseline 有明確價值後再加入。

## 目前進度

- 完成 Python package 骨架與 CLI。
- 完成 YAML 設定載入與驗證。
- 完成嚴格多商品 OHLCV CSV 載入、時間戳正規化與缺值檢查。
- 完成多組技術因子計算。
- 完成 trend、mean-reversion、swing、alternative 策略訊號。
- 完成 portfolio blending、volatility sizing、exposure cap、drawdown de-risking。
- 完成 backtest accounting：fee、slippage、positions、trades、equity curve、metrics。
- 完成 Binance/OKX 研究資料下載器。
- 完成因子研究工具、鏈上因子工具、進階驗證工具與 Qlib 匯出工具。
- 完成 Linear/Ridge/Random Forest 機器學習 baseline。
- 每次 CLI run 會輸出回測、策略歸因、因子研究、Qlib 銜接與 HTML 總覽檔案。
- 目前驗證：`pytest -v` 通過 73 個測試。

## Roadmap

- 加入 downloader CLI commands。
- 加入 4h timeframe 範例與 multi-timeframe config。
- 加入 Glassnode、CryptoQuant、Dune、Flipside 等付費/金鑰型 API adapter。
- 加入 optional native Qlib integration。
- 加入 XGBoost/LightGBM baseline。
- 視資料量與 baseline 表現，再評估 LSTM/Transformer。
- 強化 HTML 圖表：equity、drawdown、monthly returns、strategy attribution。
- 等研究回測流程穩定後，才考慮 paper trading；live trading 不在目前版本範圍。
