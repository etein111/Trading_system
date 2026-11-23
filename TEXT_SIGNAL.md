# SEC 文本信号层

文本层负责从 SEC 8-K、10-Q、10-K 文件中提取点时事件信号，并把信号接入现有 Qlib 选股和组合风险层。

## 时间边界

每个文件使用 SEC 的 `acceptanceDateTime` 作为 `available_at`；如果 SEC 没有该字段，回退到 `filingDate` 当日收盘后。回测只允许使用：

```text
available_at <= signal_asof
```

文件正文、申报日期、表单、accession、来源 URL 和证据摘录都会保存在输出 CSV，便于复核。

## 默认提取方式

默认实现不依赖大型语言模型，使用可审计的事件词和上下文规则：

- 正面：上调指引、重申指引、强劲需求、新合同、订单积压、回购、提高股息
- 负面：下调指引、弱需求、重大内控缺陷、重述、持续经营、传票、调查、诉讼、裁员、减值、网络攻击、数据泄露
- 10-K/10-Q 的通用风险因素会降低权重，只有明确事件语句才会产生高风险
- `going concern`、明确财务重述和管理层确认的重大内控缺陷会标记为高风险

输出字段包括：

```text
symbol,score,event_risk,hard_risk,available_at,form,accession,source,evidence_id,evidence_excerpt
```

## 运行

```bash
.venv/bin/python -m trading_system text-signals \
  --symbols AAPL MSFT NVDA \
  --cache-dir research/sec_text_cache \
  --output research/text_signals.csv \
  --limit-per-symbol 100
```

文本分数可以单独选股，也可以和 Qlib 分数融合：

```bash
.venv/bin/python -m trading_system select \
  --data research/expanded_data/prices.csv \
  --symbols AAPL MSFT NVDA \
  --asof 2026-11-20 \
  --qlib-scores research/qlib_model_run_20261123/qlib_scores.csv \
  --text-scores research/text_signals.csv \
  --text-weight 0.35
```

`TextScoreBackend` 默认会阻止 `event_risk >= 0.65` 的新主动仓位；Qlib 分数和文本分数通过横截面排名融合。最终目标仓位仍然要通过组合风险层的波动率、beta、压力损失和集中度约束。

## 当前限制

文本信号是可审计的规则基线，不等于完整的语义理解或研报质量评级。它还没有接入分析师预期、电话会议信息、新闻供应商和 FinBERT。加入这些数据时必须保留发布时间、来源和原文证据，并重新做点时回测。
