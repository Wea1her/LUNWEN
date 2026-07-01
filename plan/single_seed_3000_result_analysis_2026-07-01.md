# 单 seed 3000-episode 实验结果与论文叙事一致性检查

> Update 2026-07-01: 本报告主体记录的是 gas 上限 bug 修复前的首次分析，其中多个 baseline 的 `gas_util > 1`，对应收益、打包率、维度赢家和 operating-point 排名已不可作为最终依据。修复后重评估结论见 `plan/gasfix_reeval_analysis_2026-07-01.md`，后续论文叙事应以后者为准。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-01
- Verification Status: ANALYZED
- Version Label: single_seed_3000_analysis_v1

## 结论摘要

本次有效实验为 `results_single_seed_3000_20260701`，不是先前误跑的 `results_single_seed_20260701` 200-episode 预验目录。实验已完整完成：seed 42 训练 3000 episodes，主实验评估 1000 episodes，聚合表、seed-level 统计、narrative guard、protocol manifest 与 outputs manifest 均已生成。

总体判断：**单 seed 结果只部分符合当前论文叙事，不能支持“本文方法在收益或综合指标上全面领先”的强叙事；可以支持“本文方法在等待公平、老交易服务和部分约束可行性方面体现出折中特征，但强基线仍构成主要威胁”的谨慎叙事。**

## 实验配置

执行命令：

```bash
PYTHONPATH=experiments python3 experiments/run_experiments.py \
  --stages main \
  --seeds 42 \
  --episodes 3000 \
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

关键运行证据：

- `run_summary.json`: seed 42 成功完成，`policy_source=trained`，`resolved_device=cuda:0`。
- `timing.json`: total runtime 9217.65s，其中 train 8595.92s，eval 593.29s。
- `outputs_manifest.json`: `report_incomplete=false`，主叙事所需产物无缺失。
- `v5_protocol_manifest.json`: `evidence_level=dryrun_single_seed`，`formal_statistics_ready=false`，正式门槛为 5 seeds。
- `checkpoint_meta.json`: 虽然训练跑满 3000 episodes，但正式评估加载固定验证池 two-stage 规则选择的 `best_model.pt`，`best_episode=150`。

## 核心结果

| Method | Fee ↑ | Jain ↑ | Old tx ↑ | Risk exposure ↓ | Edge10 risk ↓ | Risky inclusion ↑ | Gas util ↑ | Packing ↑ | Composite ↑ | Constrained fee ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | **1730.43** | 0.8228 | 0.9824 | 0.1942 | 0.1290 | **0.7157** | **1.1648** | **0.7169** | 0.6008 | -1.0000 |
| Gas Priority | **1730.43** | 0.8228 | 0.9824 | 0.2692 | 0.1784 | **0.7157** | **1.1648** | **0.7169** | 0.5858 | -1.0000 |
| Heuristic | **1730.43** | 0.8228 | 0.9824 | 0.0177 | 0.0119 | **0.7157** | **1.1648** | **0.7169** | 0.6360 | -1.0000 |
| Center-Insertion | **1730.43** | 0.8228 | 0.9824 | **0.0035** | **0.0024** | **0.7157** | **1.1648** | **0.7169** | **0.6389** | -1.0000 |
| Dynamic Tri-Objective | 1639.46 | 0.9168 | 1.0000 | 0.1259 | 0.0662 | 0.4911 | 0.9997 | 0.6252 | 0.6255 | **0.3074** |
| 本文方法 | 1663.01 | **0.9198** | **1.0000** | 0.2246 | 0.1589 | 0.6870 | 0.9997 | 0.6457 | 0.6104 | 0.0870 |

维度赢家由 `narrative_guard.json` 给出：

- `block_fee_mean`: FIFO
- `fairness_mean`: ours
- `risk_exposure_mean`: Center-Insertion
- `top10_risk_mean`: Heuristic
- `packing_ratio_mean`: FIFO
- `composite_score_mean`: Center-Insertion
- `constrained_fee_score_mean`: Dynamic Tri-Objective

`narrative_guard.json` 明确给出：`can_claim_all_dimensions_best=false`。阻塞原因包括本文方法不是收益、风险暴露、Top10 风险、打包比、综合分和 constrained fee score 的最优方法。

## 约束与 operating point

在 balanced operating mode 下：

| Method | Feasible rate ↑ | Feasible fee mean ↑ | Risk-adjusted fee ↑ | Two-stage score ↑ | Rank |
|---|---:|---:|---:|---:|---:|
| Dynamic Tri-Objective | **0.783** | 0.6697 | **0.6285** | **16.8185** | 1 |
| 本文方法 | 0.648 | **0.6775** | 0.5985 | 16.1003 | 2 |

解释：

- 本文方法在可行 episode 子集里的 fee mean 高于 Dynamic Tri-Objective，但 feasible rate 明显更低。
- V5 two-stage 规则优先可行率，因此 balanced 排名中 Dynamic Tri-Objective 第一，本文方法第二。
- 对所有方法而言，主要约束瓶颈是 `fairness_floor`。本文方法 1000 个 episode 中有 352 个 infeasible，其中 fairness_floor 违反 171 次，risk_ceil 违反 154 次，top10_risk_ceil 违反 95 次。

## 与论文叙事的符合度

### 支持的部分

1. **等待公平叙事得到支持。** 本文方法 Jain 指数 0.9198，为所有方法最高；oldest coverage / old tx pack rate 均为 1.0000，starvation 指标为 0。
2. **“多目标折中”叙事得到部分支持。** 本文方法不是收益或风险单项最优，但在 fairness、old-tx service、可行子集收益之间形成了可解释折中。
3. **强基线必要性得到支持。** Dynamic Tri-Objective 和 Center-Insertion 的表现证明 V5 引入强基线是必要的，论文不能只与弱启发式对比。
4. **论文中“不能声称所有维度最优”的叙事是正确的。** `narrative_guard` 明确阻止 all-dimensions-best claim。

### 不支持或需要降级的部分

1. **不支持“收益领先”。** 本文方法 fee 为 1663.01，低于 FIFO/Gas/Heuristic/Center/Fee-Risk/Fair-Fee 的 1730.43，仅高于 Dynamic Tri-Objective 的 1639.46。
2. **不支持“综合指标领先”。** composite score 为 0.6104，低于 Center-Insertion 0.6389、Heuristic 0.6360、Dynamic Tri-Objective 0.6255。
3. **不支持“风险控制最优”。** risk exposure 为 0.2246，明显高于 Center-Insertion 0.0035、Heuristic 0.0177、Dynamic Tri-Objective 0.1259，也高于 FIFO 0.1942。
4. **不支持“balanced operating point 第一”。** Dynamic Tri-Objective 在 balanced 排名第一，本文方法第二。
5. **不能使用正式显著性措辞。** 本次只有 1 个训练 seed，`formal_statistics_ready=false`；seed-level paired tests 均为 exploratory only，p 值不具备正式统计解释力。

## 建议写法

建议保留或改成如下谨慎表达：

> 在单 seed 主场景探索性实验中，本文方法在等待公平性与老交易服务方面取得最优表现，并在 balanced operating point 下位列第二，显示出收益、等待服务与约束可行性之间的折中特征。然而，Dynamic Tri-Objective Greedy 在 two-stage operating point 排名中领先，Center-Insertion 与 Heuristic 在风险暴露指标上明显更优。因此，当前单 seed 结果不能支持本文方法在收益、风险或综合指标上的全面领先，正式结论仍需 5 个独立训练 seed 验证。

不建议使用如下表述：

- “本文方法显著优于所有基线。”
- “本文方法在综合指标上领先。”
- “本文方法实现最佳风险控制。”
- “本实验已经证明稳定最优。”

## 统计与方法学检查

统计状态：

- Seed-level 统计单位正确：independent training seed。
- 本次 `n_completed_seeds=1`，低于正式门槛 `formal_min_seeds=5`。
- Episode-level paired tests 只可作为诊断，不可当作 1000 个独立训练重复。
- 多重比较风险存在；输出已包含 Holm 口径，但单 seed 下不能形成正式显著性结论。

11 类统计/方法学谬误扫描：

| Fallacy | Status | Detail |
|---|---|---|
| Simpson's paradox | NOTE | 本次只跑 main 场景，未跑风险比例/池规模/倍率分组鲁棒性，不能检查跨场景方向反转。 |
| Ecological fallacy | NOTE | 当前结论限于仿真 episode 和 seed，不应外推到真实链上个体交易效果。 |
| Berkson's paradox | NOTE | 无真实样本筛选，但仿真分布本身是参数化生成，外推需谨慎。 |
| Collider bias | NOTE | 本次未做回归控制变量分析，不构成直接风险。 |
| Base rate neglect | CAUTION | 风险比例默认 15%，报告风险指标时应同时说明该 base rate；不可只报 risk exposure。 |
| Regression to the mean | NOTE | 非 pre-post 极端组设计，不适用。 |
| Survivorship bias | NOTE | 无 dropout，但仅单 seed，不能代表跨 seed 稳定性。 |
| Look-elsewhere effect | CAUTION | 指标很多，若只挑本文方法优势指标，会造成选择性报告。 |
| Garden of forking paths | CAUTION | 有多指标、operating mode、best checkpoint 选择规则；必须固定叙事口径并报告 narrative guard。 |
| Correlation != causation | CAUTION | 可说仿真中策略产生某些排序表现，不能说已经证明真实链上 MEV 防护效果。 |
| Reverse causality | NOTE | 非观察性因果建模，不是主要风险。 |

覆盖：11/11 已检查。

## 后续动作

1. 若论文要维持“正式结果”口径，必须补齐 5 seeds：`42 123 456 789 2025`。
2. 当前单 seed 可作为技术可运行性与叙事风险检查材料，不宜直接回填成正文正式主表。
3. 如果想强化本文方法叙事，需要重点解决两点：提高 feasible rate，降低 risk exposure / edge10 risk；否则 Dynamic Tri-Objective 与 Center-Insertion 会持续压制主叙事。
4. 论文正文建议把核心叙事从“综合领先”调整为“公平优先折中 + 强基线边界清晰 + 正式多 seed 验证后定论”。
