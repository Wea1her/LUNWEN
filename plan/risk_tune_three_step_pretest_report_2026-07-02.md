# 风险感知增强三步预验报告

## Material Passport

- Report ID: `risk_tune_three_step_pretest_seed42_2026-07-02`
- Mode: code experiment validation
- Verification Status: ANALYZED
- Scope: 单 seed 预验，不作为正式论文结果。
- Protocol: `seed=42`, `episodes=800`, `eval_episodes=400`, `pool_size=300`, `val_metric=two_stage`, `operating_mode=balanced`
- Source runs:
  - `results_risk_tune_A_seed42`
  - `results_risk_tune_B_seed42`
  - `results_risk_tune_C_seed42`
- Primary evidence files:
  - `aggregated_main.json`
  - `constrained_eval_summary.json`
  - `seed_42/checkpoints/training_progress.json`
  - `seed_42/checkpoints/checkpoint_meta.json`
  - `run_summary.json`
- Baseline comparator: `dynamic_tri_objective`

## 1. 实验配置

| Config | GAMMA_R | Terminal Risk | Terminal Top10 | Risk Fee Lambda | Best Episode | Best two_stage |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.8 | 0.35 | 0.25 | 0.5 | 200 | 16.6896 |
| B | 1.0 | 0.50 | 0.40 | 0.6 | 150 | 16.6467 |
| C | 1.2 | 0.70 | 0.50 | 0.8 | 150 | 16.2936 |

说明：当前代码没有独立的 `TERMINAL_EDGE10_RISK_WEIGHT`，因此本次只覆盖现有的 `TERMINAL_TOP10_RISK_WEIGHT`。

## 2. A/B/C 结果对比

以下表格只比较 `ours` 在三组风险增强配置下的单 seed 评估均值。

| Config | Fee | Fairness | OldTx | Risk | Top10 | Edge10 | Feasible | Packing | TradeScore | RiskAwareTrade | ConstrainedTrade | RiskyIncl | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | 1595.93 | 0.9274 | 1.0000 | 0.1955 | 0.1148 | 0.1243 | 0.8075 | 0.6150 | 0.7134 | 0.6984 | 0.6848 | 0.5904 | 轻度增强，可行率提升，但风险仍偏高 |
| B | 1602.69 | 0.9278 | 1.0000 | 0.2037 | 0.1425 | 0.1367 | 0.7900 | 0.6235 | 0.7053 | 0.6890 | 0.6711 | 0.6295 | 均衡增强未带来风险下降 |
| C | 1615.70 | 0.9252 | 1.0000 | 0.1822 | 0.0975 | 0.1168 | 0.8375 | 0.6326 | 0.7257 | 0.7122 | 0.7117 | 0.6099 | 三组中风险和可行率最好 |

预验趋势：

- C 是三组中 `risk_exposure` 最低的一组，从先前约 0.21 的 operating point 降到 0.1822，但仍未进入方案目标区间 0.15-0.17。
- C 的 `feasible_rate=0.8375`，明显高于目标下限 0.70，也高于 A/B。
- C 的 `top10_risk=0.0975` 和 `edge10_risk=0.1168` 是三组中最低。
- C 没有通过少打包来换低风险：`packing_ratio=0.6326` 是三组最高，`risky_inclusion_rate=0.6099` 也没有明显下降。
- B 不符合原方案对“均衡风险增强”的预期；它的 `risk_exposure` 和 `edge10_risk` 均高于 A/C。

## 3. Ours vs Dynamic Tri-Objective

Dynamic Tri-Objective 在三组评估中相同，因为同一 seed 和测试池下 baseline 规则固定。

| Config | Method | Fee | Fairness | OldTx | Risk | Top10 | Edge10 | Feasible | Packing | TradeScore | RiskAwareTrade | ConstrainedTrade | RiskyIncl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | ours | 1595.93 | 0.9274 | 1.0000 | 0.1955 | 0.1148 | 0.1243 | 0.8075 | 0.6150 | 0.7134 | 0.6984 | 0.6848 | 0.5904 |
| A | dynamic_tri_objective | 1638.40 | 0.9162 | 1.0000 | 0.1214 | 0.0860 | 0.0643 | 0.7575 | 0.6267 | 0.7830 | 0.7788 | 0.7795 | 0.4954 |
| B | ours | 1602.69 | 0.9278 | 1.0000 | 0.2037 | 0.1425 | 0.1367 | 0.7900 | 0.6235 | 0.7053 | 0.6890 | 0.6711 | 0.6295 |
| B | dynamic_tri_objective | 1638.40 | 0.9162 | 1.0000 | 0.1214 | 0.0860 | 0.0643 | 0.7575 | 0.6267 | 0.7830 | 0.7788 | 0.7795 | 0.4954 |
| C | ours | 1615.70 | 0.9252 | 1.0000 | 0.1822 | 0.0975 | 0.1168 | 0.8375 | 0.6326 | 0.7257 | 0.7122 | 0.7117 | 0.6099 |
| C | dynamic_tri_objective | 1638.40 | 0.9162 | 1.0000 | 0.1214 | 0.0860 | 0.0643 | 0.7575 | 0.6267 | 0.7830 | 0.7788 | 0.7795 | 0.4954 |

Interpretation:

- 与 Dynamic Tri-Objective 相比，C 的 fee 低约 22.71，fairness 高约 0.0090，feasible rate 高约 0.0800，packing ratio 高约 0.0059。
- Dynamic Tri-Objective 的风险指标仍更低：risk exposure 低约 0.0608，edge10 risk 低约 0.0525。
- 因此，当前结果不能叙述为 ours 在风险上超过 Dynamic Tri-Objective；更合适的叙述是风险增强后 ours 的 operating point 向风险感知方向移动，同时保留较好的公平性、可行率和打包能力。

## 4. Checkpoint 收敛观察

三组最佳 checkpoint 都出现在很早的位置：

- A: episode 200
- B: episode 150
- C: episode 150

这与此前单 seed 预验中“150 左右出现最高点”的现象一致。可能原因是 two_stage 选择规则强依赖固定验证池上的可行率、风险和收益折中；训练后期虽然平均训练奖励和 fee 继续变化，但并没有提高验证池上的 two_stage selection score。

对缩短 episode 的启示：

- 对当前协议和 seed 42，`800 episodes` 主要用于确认后期没有再次刷新最佳点。
- 若只做快速配置筛选，可以考虑把单 seed 预验缩短到 `250-300 episodes`，因为 A/B/C 的最佳点都不晚于 200。
- 正式 5 seed 不建议直接缩到 150；更稳妥的是先用 `300 episodes` 做一轮 5 seed pilot，确认各 seed 的最佳点是否也集中在 50-250。

## 5. 推荐正式配置

推荐配置：C

理由：

- C 是三组中 `risk_exposure`、`top10_risk`、`feasible_rate`、`packing_ratio`、`trade_score`、`risk_aware_trade_score`、`constrained_trade_score` 表现最好的 ours 配置。
- B 没有达到原先“均衡增强”的预期，风险暴露反而高于 A/C。
- C 的风险仍未到 0.15-0.17 目标区间，因此它更适合作为“风险增强候选配置”进入正式 5 seed，而不是作为已经达到目标的最终配置。

## 6. 是否进入正式 5 seed

是否建议用该配置进入正式 5 seed：是，建议以 C 进入正式 5 seed 预验/正式候选，并保留 Dynamic Tri-Objective 作为关键参照。

执行建议：

- 首选：`C, episodes=800, eval_episodes=1000, seeds=5`，用于保持与既有正式协议可比。
- 若主要目标是节省时间：先跑 `C, episodes=300, eval_episodes=400 or 1000, seeds=5` 的 pilot，检查最佳 checkpoint 是否仍集中在早期，再决定正式实验是否采用 300 episodes。
- 报告中应明确这是配置级风险增强，不涉及 MDP、网络结构、baseline 或指标体系变化。

## 7. 论文叙事建议

可使用的谨慎表述：

> 单 seed 预验表明，提高风险惩罚与终止风险权重后，强化学习排序策略的 operating point 向风险感知方向移动：在维持较高公平性、旧交易覆盖和打包能力的同时，可行率进一步提高，风险暴露有所下降。

需要避免的表述：

- 显著优于
- 最终证明
- 全局最优
- 真实 MEV 防护

## 8. 结论

三步预验已完成。A/B/C 三组均生成完整结果目录。就单 seed 证据看，C 是更合适的正式 5 seed 候选，但风险指标仍没有达到预设目标区间，后续正式实验应重点验证 C 的跨 seed 稳定性。
