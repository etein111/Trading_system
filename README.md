# Personal Trading System (v0.1)

这是根据当前交易约束搭建的研究与模拟盘版本：

- 美股、中周期（数周到数月）
- 无杠杆、无做空、无期权
- 核心仓位 + 主动仓位 + 现金缓冲
- 200 日均线市场状态过滤
- 单笔风险、主动仓位上限、组合回撤熔断
- 价格止损、逻辑止损和时间止损的接口
- 回测时避免用当天收盘价预测当天收益

它不会连接真实券商，也不会自动下单。先用真实历史数据和模拟盘验证规则，确认执行纪律和回撤符合预期后，再单独增加券商适配器。

## 快速开始

```bash
cd /Users/chuan/Documents/Codex/2026-11-20/cong/outputs/trading_system
python3 -m trading_system demo --days 1500
```

运行后会在 `artifacts/` 生成：

- `demo_prices.csv`：可重复生成的示例价格数据
- `equity.csv`：每日净值和回撤
- `trades.csv`：目标权重变化记录
- `metrics.json`：回测摘要

## 严格时间过滤的研究回测

每个执行日 `t` 的信号只读取 `t` 之前的历史数据，交易收益从前一交易日收盘到 `t` 收盘计算；交易日志同时保存 `signal_asof` 和 `execution_date`，研究命令会检查两者是否发生未来数据穿越。

```bash
python3 -m trading_system research \
  --data paper/prices.csv \
  --output research
```

会生成策略净值、交易日志、年度收益、基准买入并持有结果和 `lookahead_audit`。当前价格数据来自 yfinance 的复权历史价格，行级时间过滤是严格的；如果要审计供应商的历史修订、公司行动和点时基本面，还需要接入提供 bitemporal 数据的正式数据源。

## 选股器

系统现在提供 `select` 命令。它先做可交易性和趋势过滤，再对候选进行横截面评分：

整体架构和 Qlib 接入方式见 [SELECTOR_DESIGN.md](SELECTOR_DESIGN.md)。

- 12 个月动量，跳过最近 1 个月；
- 60 日短期动量；
- 相对 200 日均线的趋势距离；
- 63 日波动率，低波动得分更高；
- 可选的点时基本面：`roe`、`roa`、`fcf_yield`、`earnings_revision`、`pe`、`pb`、`ev_ebitda`。

基本面 CSV 必须至少包含：

```text
symbol,available_at,roe,roa,fcf_yield,earnings_revision,pe,pb,ev_ebitda
MSFT,2025-01-30,0.32,0.18,0.021,0.04,35.1,12.4,25.0
```

系统只会使用 `available_at <= signal_asof` 的记录，并取当时最近一条记录。没有基本面数据时，输出会明确标记为 `price_factors_only`，不会把价格因子结果描述成完整基本面选股。

```bash
python3 -m trading_system select \
  --data research/prices.csv \
  --symbols QQQ XLK MSFT AMZN NVDA PLTR \
  --output research/selection.csv
```

使用自己的数据时，CSV 可以是长表：

```text
date,symbol,close
2020-01-02,SPY,323.54
2020-01-02,QQQ,214.20
```

也支持宽表：第一列为 `date`，其余列为股票代码和收盘价。

```bash
python3 -m trading_system backtest --data /absolute/path/prices.csv
```

如果安装了可选的 `yfinance`，也可以下载复权收盘价：

```bash
pip install yfinance
python3 -m trading_system download --symbols SPY QQQ XLK MSFT AMZN NVDA PLTR --start 2015-01-01 --output data/prices.csv
python3 -m trading_system signals --data data/prices.csv
```

`signals` 只打印目标权重，不会连接券商或提交订单。

## 公开数据训练

`train-public` 使用 SEC XBRL Company Facts 的实际申报时间和历史价格构建点时样本，训练一个透明的随机森林横截面模型。模型输出的 `model_scores.csv` 兼容 `QlibScoreBackend`。

```bash
python3 -m trading_system train-public \
  --prices research/prices.csv \
  --symbols AAPL MSFT AMZN GOOGL META NVDA AMD AVGO ORCL CRM ADBE NFLX NOW PANW PLTR SNOW MU ANET INTU QCOM TXN IBM \
  --output research/public_model
```

训练数据、SEC 缓存、模型文件和滚动样本外结果都会写入输出目录。当前模型是研究基线，仍需与规则因子和 SPY/QQQ 基准比较后再接入主动仓位。

### Qlib/LightGBM 训练

项目已经创建 `.venv`（Python 3.11.18），并声明 `pyqlib>=0.9.7` 与 `lightgbm>=4.0`。训练默认使用扩充后的技术/成长股票池、横截面特征排名、60 日标签边界 purge 和固定树数；SEC 基本面作为可选对照，不默认混入主模型。

先补充公开数据：

```bash
.venv/bin/python scripts/build_expanded_public_data.py \
  --output-dir research/expanded_data
```

然后使用 Qlib 的 `DataLoader`、`DataHandlerLP` 和 `DatasetH` 做点时 walk-forward 训练：

```bash
.venv/bin/python -m trading_system train-qlib \
  --prices research/expanded_data/prices.csv \
  --symbols $(tr '\n' ' ' < research/expanded_data/symbols.txt) \
  --cache-dir research/expanded_data/sec_cache \
  --volumes research/expanded_data/volumes.csv \
  --output research/qlib_model_enhanced
```

默认使用价格、相对 SPY 和成交量特征、原始超额收益目标、固定 80 棵树，并生成 2024 年之后的最终留出指标；需要把 SEC 基本面加入对照时追加 `--include-fundamentals`，需要排名目标时使用 `--target-mode rank`。输出的 `qlib_scores.csv` 可由 `QlibScoreBackend` 读取，也可以直接用于选股：

```bash
.venv/bin/python -m trading_system select \
  --data research/expanded_data/prices.csv \
  --symbols AAPL MSFT AMZN NVDA PLTR CRWD DDOG AVGO MU \
  --asof 2026-11-19 \
  --qlib-scores research/qlib_model_enhanced/qlib_scores.csv
```

Qlib 训练报告见 [QLIB_TRAINING_REPORT.md](QLIB_TRAINING_REPORT.md)；报告中的 LightGBM 结果仍低于随机森林基线，所以不会自动改变模拟账户仓位。历史分数还可以进入组合回放：

```bash
.venv/bin/python -m trading_system research \
  --data research/expanded_data/prices.csv \
  --output research/qlib_backtest_enhanced \
  --qlib-scores research/qlib_model_enhanced/qlib_walk_forward_scores.csv
```

模型到账户的桥接使用同一份历史分数和扩展候选池：

```bash
.venv/bin/python -m trading_system research \
  --data research/expanded_data/prices.csv \
  --output research/qlib_backtest_bridge \
  --qlib-scores research/qlib_model_enhanced/qlib_walk_forward_scores.csv \
  --candidates-file research/expanded_data/symbols.txt
```

本地行情驱动模拟账户时，可以避免实时数据源波动：

```bash
.venv/bin/python -m trading_system paper-run \
  --config config_qlib.yaml \
  --data research/expanded_data/prices.csv \
  --qlib-scores research/qlib_model_enhanced/qlib_scores.csv \
  --candidates-file research/expanded_data/symbols.txt
```

`--enforce-price-eligibility` 可选地再加趋势门槛；默认让模型分数负责排序，由组合风险规则负责仓位。组合风险层说明见 [PORTFOLIO_RISK.md](PORTFOLIO_RISK.md)。

分析现有账户的 beta、波动率、集中度和压力测试：

```bash
.venv/bin/python -m trading_system portfolio-risk \
  --data research/expanded_data/prices.csv \
  --state paper/account.json \
  --config config_qlib.yaml
```

## 真实行情模拟账户

`paper-run` 使用本地 JSON 账户保存现金和持仓，读取最新历史行情后按策略目标权重进行模拟成交。它只写入本地状态和交易日志，不访问券商，也不会提交真实订单。

```bash
pip install -r requirements.txt
python3 -m trading_system paper-run \
  --state paper/account.json \
  --log paper/trades.csv \
  --lookback-days 900
```

默认按最新日线复权收盘价模拟成交，并加入 5 bps 的滑点和配置中的交易成本。`yfinance` 的行情可能存在延迟和限流，适合当前的日频/中周期模拟，不等同于交易所实时行情。若需要实时盘中数据，应替换 `download_yfinance` 为有授权的行情供应商适配器。

## 当前默认规则

- 60% 核心仓位：宽基 ETF（默认 `SPY`）
- 20% 主动仓位：最多 4 个候选，每个默认不超过 6%
- 20% 现金缓冲
- 大盘指数在 200 日均线之上且 20 日均线高于 50 日均线时，允许完整主动仓位
- 大盘转弱时，主动仓位降为 0，核心仓位也会降低
- 账户回撤达到 10% 时冻结扩张风险，15% 时减半主动风险，20% 时暂停新增主动交易
- 单笔计划风险默认为账户的 0.5%；仓位由 `账户风险 ÷ 止损距离` 计算

这些数字是第一版起始参数，不是收益承诺。正式使用前应更换成经过复核的、无生存者偏差的数据，并加入手续费、滑点、分红和拆股处理。

## 后续扩展顺序

1. 接入点时数据和公司行动数据；
2. 加入完整订单撮合、部分成交和交易时段；
3. 用 MLflow 或类似工具记录每次实验；
4. 用 Qlib/LightGBM 增加市场状态或风险分类模型；
5. 最后才接入券商模拟账户。

