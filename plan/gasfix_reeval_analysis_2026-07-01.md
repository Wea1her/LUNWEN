# Gas 修复后重评估分析

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: gasfix_single_seed_3000_reeval_v1

## 结论摘要

本次重评估使用 `results_single_seed_3000_20260701` 中已有 seed 42 checkpoint，执行 `--skip-train` 重新评估 1000 episodes。旧结果已备份到 `results_single_seed_3000_20260701/pre_gasfix_backup/`。

结论：gas 上限 bug 已被修复并被重评估结果验证。修复前 6 个非动态 baseline 的 `gas_util_mean=1.164774`，修复后所有方法均约为 `0.99966`，不再超过 block gas limit。修复后本文方法成为 `block_fee_mean` 和 `packing_ratio_mean` 第一，但仍不是 fairness、risk、top10 risk、composite 或 constrained fee 的第一。因此结果更靠近“高收益 + 打包效率 + 多目标折中”的叙事，但仍不能支持“全维度最优”或“约束 operating point 第一”。

## 重评估命令

```bash
PYTHONPATH=experiments python3 experiments/run_experiments.py \
  --stages main \
  --seeds 42 \
  --skip-train \
  --eval-episodes 1000 \
  --pool-size 300 \
  --workers 1 \
  --max-gpu-workers 1 \
  --device cuda:0 \
  --operating-mode balanced \
  --val-metric two_stage \
  --fairness-first \
  --output results_single_seed_3000_20260701
```

`run_summary.json` 显示 seed 42 `status=success`，`policy_source=trained`，`resolved_device=cuda:0`，本次重评估耗时约 599s，其中 eval 约 577s。

## Gas 修复验证

| Method | Gas util 修复前 | Gas util 修复后 | Delta |
|---|---:|---:|---:|
| FIFO | 1.164774 | 0.999663 | -0.165111 |
| Gas Priority | 1.164774 | 0.999667 | -0.165107 |
| Heuristic | 1.164774 | 0.999667 | -0.165108 |
| Fee-Risk Linear | 1.164774 | 0.999652 | -0.165122 |
| Fair-Fee Greedy | 1.164774 | 0.999665 | -0.165109 |
| Center-Insertion | 1.164774 | 0.999659 | -0.165115 |
| Dynamic Tri-Objective | 0.999655 | 0.999655 | 0.000000 |
| 本文方法 | 0.999656 | 0.999656 | 0.000000 |

修复解释：`experiments/baselines.py` 中 `_apply_nonce_order` 原先在一轮内使用 stale `gas_left` 批量筛选候选交易，导致同一轮累计加入的交易 gas 可超过剩余 block gas。修复后改为逐笔交易重新检查 `tx.gas > gas_left`，并在加入后立即扣减 `gas_left`。

## 修复后核心结果

| Method | Fee ↑ | Jain ↑ | Old tx ↑ | Risk exposure ↓ | Edge10 risk ↓ | Gas util | Packing ↑ | Composite ↑ | Constrained fee ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1503.09 | **0.9340** | 1.0000 | 0.1903 | 0.1273 | 0.9997 | 0.6224 | 0.5944 | **0.3738** |
| Gas Priority | 1586.27 | 0.8268 | 0.9300 | 0.3605 | 0.2522 | 0.9997 | 0.6225 | 0.5367 | -1.0000 |
| Heuristic | 1534.49 | 0.8261 | 0.9310 | 0.0858 | 0.0556 | 0.9997 | 0.6249 | 0.5832 | -1.0000 |
| Fee-Risk Linear | 1564.83 | 0.8271 | 0.9299 | 0.0489 | 0.0358 | 0.9997 | 0.6218 | 0.5955 | -0.9984 |
| Fair-Fee Greedy | 1360.64 | 0.8711 | 0.9098 | 0.3027 | 0.1822 | 0.9997 | 0.6247 | 0.5191 | -0.8630 |
| Center-Insertion | 1425.51 | 0.8598 | 0.9037 | **0.0003** | **0.0002** | 0.9997 | 0.6219 | 0.5864 | -0.8183 |
| Dynamic Tri-Objective | 1639.46 | 0.9168 | 1.0000 | 0.1259 | 0.0662 | 0.9997 | 0.6252 | **0.6255** | 0.3074 |
| 本文方法 | **1663.01** | 0.9198 | **1.0000** | 0.2246 | 0.1589 | 0.9997 | **0.6457** | 0.6104 | 0.0870 |

## Narrative Guard

修复后的 `narrative_guard.json`：

- `can_claim_all_dimensions_best=false`
- blocking reasons:
  - `ours_not_best_on_fairness_mean`
  - `ours_not_best_on_risk_exposure_mean`
  - `ours_not_best_on_top10_risk_mean`
  - `ours_not_best_on_composite_score_mean`
  - `ours_not_best_on_constrained_fee_score_mean`

维度赢家：

| Dimension | Winner |
|---|---|
| `block_fee_mean` | ours |
| `fairness_mean` | FIFO |
| `risk_exposure_mean` | Center-Insertion |
| `top10_risk_mean` | Heuristic |
| `packing_ratio_mean` | ours |
| `composite_score_mean` | Dynamic Tri-Objective |
| `constrained_fee_score_mean` | FIFO |

## Operating Points

修复后 balanced constrained ranking：

| Rank | Method | Feasible rate ↑ | Feasible fee mean ↑ | Risk-adjusted fee ↑ | Two-stage score ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | Dynamic Tri-Objective | 0.783 | 0.6697 | **0.6285** | **16.8185** |
| 2 | FIFO | **0.848** | 0.6201 | 0.5462 | 16.5990 |
| 3 | 本文方法 | 0.648 | **0.6775** | 0.5985 | 16.1003 |

不同 operating mode：

| Mode | Top method | 本文方法 rank |
|---|---|---:|
| aggressive | Dynamic Tri-Objective | 2 |
| balanced | Dynamic Tri-Objective | 3 |
| conservative | FIFO | 3 |

约束瓶颈仍是 `fairness_floor`。本文方法 1000 个 episode 中 feasible rate 为 0.648，352 个 episode infeasible；违反分解为 fairness floor 171 次、risk ceil 154 次、top10 risk ceil 95 次、oldest coverage 0 次。

## 对论文叙事的影响

修复后可以支持的叙事：

1. 本文方法在合法 gas 约束下取得最高收益：`block_fee_mean=1663.01`，高于 Dynamic Tri-Objective 的 1639.46。
2. 本文方法打包效率最高：`packing_ratio_mean=0.6457`。
3. 本文方法保持老交易服务完整：`oldest_coverage_mean=1.0000`，`old_tx_pack_rate_mean=1.0000`。
4. 本文方法在 aggressive operating point 下排名第 2，说明若偏向收益和可行子集收益，叙事比修复前更有利。

仍不能支持的叙事：

1. 不能说 fairness 最优；FIFO 的 Jain 指数为 0.9340，高于本文方法 0.9198。
2. 不能说风险控制最优；Center-Insertion、Fee-Risk Linear、Heuristic、Dynamic Tri-Objective、FIFO 的风险暴露均低于本文方法。
3. 不能说综合指标最优；Dynamic Tri-Objective 的 composite score 为 0.6255，高于本文方法 0.6104。
4. 不能说 balanced/conservative operating point 第一；本文方法分别为第 3、第 3。
5. 不能使用正式显著性措辞；当前仍是 `dryrun_single_seed`，正式门槛为 5 independent seeds。

## 下一步代码方向

优先级最高的是提高本文方法的约束可行率，同时压低 risk/top10 risk。当前主要问题不是 fee，而是 feasible rate 低于 FIFO 和 Dynamic Tri-Objective，且风险指标偏高。

建议下一步按这个顺序改：

1. 在训练 reward 或动作选择中增加显式 top10/risk 早期惩罚，避免策略用高风险交易换收益。
2. 把 evaluation 的 two-stage selection score 代理进 validation/early-selection，使 best checkpoint 更贴近论文表中的 operating-point 目标。
3. 给 policy inference 加可选 safety filter：当候选动作会突破 risk/top10 risk 约束时降权或屏蔽，但要单独报告 filtered 与 unfiltered 两组，避免隐藏方法本体表现。
4. 跑 5 seeds 正式实验前，先用 seed 42 做 3000 episode 的小改动 A/B，对比 feasible rate、risk exposure、top10 risk、block fee 是否同时改善。

## 统计与方法学状态

- 本次只验证 seed 42，仍不能形成正式统计结论。
- Episode-level paired tests 只可作为诊断，不能当成 1000 个独立训练重复。
- 修复后旧 over-gas baseline 结论已失效，任何论文表格必须使用本次重评估后的产物。
- 当前最稳妥正文叙事是“收益与打包效率领先，但约束可行性和风险控制仍受强基线挑战；方法展示了可解释 trade-off，正式结论待 5 seeds 验证”。
