# 组合风险层

组合风险层位于模型选股和模拟成交之间。模型负责产生候选和分数，组合风险层负责回答“这些候选加入当前账户后，组合的风险是否可接受”。

## 参考实现

设计参考了以下开源项目和文献中的常用做法：

- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)：协方差矩阵、组合波动率、风险预算和组合约束。
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)：风险贡献、风险平价和压力风险度量。
- [Qlib](https://github.com/microsoft/qlib)：将模型信号、组合策略、交易成本和回测结果分层记录。
- Ledoit and Wolf (2004)：协方差估计需要注意小样本和噪声；当前版本使用历史协方差，后续可加入收缩估计。
- Maillard, Roncalli and Teiletche (2010)：风险平价思想，即组合权重不能只看预期收益，还要看每个资产对组合风险的贡献。

## 当前指标

`trading_system/portfolio_risk.py` 计算：

- 现金、总敞口、净敞口和主动敞口
- 持仓权重和集中度 HHI
- 历史年化波动率
- 组合 beta 相对 SPY
- 历史最大回撤
- 每个持仓的边际风险贡献
- SPY 下跌 5%、10%、20% 的压力情景
- 所有持仓下跌 10%、主动仓位下跌 20% 的压力情景

## 决策规则

生成目标仓位后，风险层按顺序处理：

1. 单个主动仓位不超过 `max_single_active_weight`。
2. 主动仓位总和不超过 `active_weight` 和当前回撤状态给出的 `active_cap`。
3. 如果组合波动率超过 `max_portfolio_volatility`，按比例缩小主动仓位。
4. 如果组合 beta 超过 `max_portfolio_beta`，按比例缩小主动仓位。
5. 如果 SPY 下跌 20% 的压力损失超过 `portfolio_stress_loss_limit`，按比例缩小主动仓位。
6. 风险层只缩小主动仓位，不增加核心仓位；被缩小的部分留在现金中。

默认阈值：

```yaml
risk_lookback_days: 252
max_portfolio_volatility: 0.35
max_portfolio_beta: 1.25
portfolio_stress_loss_limit: 0.20
```

## 使用方式

分析现有模拟账户：

```bash
.venv/bin/python -m trading_system portfolio-risk \
  --data research/expanded_data/prices.csv \
  --state paper/account.json \
  --config config_qlib.yaml
```

`paper-run` 会把组合风险报告放在输出 JSON 的 `portfolio_risk` 字段中，并在计算目标仓位时应用风险层：

```bash
.venv/bin/python -m trading_system paper-run \
  --config config_qlib.yaml \
  --data research/expanded_data/prices.csv \
  --state paper/account.json \
  --qlib-scores research/qlib_model_run_20261123/qlib_scores.csv \
  --candidates-file research/expanded_data/symbols.txt
```

当前账户的 SPY 40% / 现金 60% 状态对应年化波动率约 5.7%、beta 约 0.40，没有组合风险告警。SPY 核心仓位不受主动仓位上限告警影响。

当前版本使用历史协方差和单一 SPY 基准，尚未包含行业协方差、融资融券、期权 Greeks、税务和多账户合并。

## 历史回放结果

使用增强 Qlib 分数、61 个候选标的和组合风险层回放：

- CAGR：10.68%
- 最大回撤：21.71%
- Sharpe：0.959
- SPY CAGR：13.28%
- lookahead violations：0

风险层略微降低收益，但改善了风险调整收益和最大回撤。它的作用是限制组合风险，不负责把模型变成超额收益策略。
