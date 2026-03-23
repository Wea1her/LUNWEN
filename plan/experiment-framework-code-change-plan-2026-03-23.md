# 实验框架代码改动文档（2026-03-23，基于最新 dry-run 复核版）

## 1. 文档目标

本文档用于指导下一轮实验框架代码改造。当前目标不再是笼统地“把 fairness 做高”，而是要让实验系统能够更真实地支撑以下两类论文叙事：

1. **本文方法在 fairness 已不落后关键公平基线的前提下，进一步改善收益-公平-风险的综合折中；**
2. **本文方法在预注册的多目标 / 约束式评估协议下，具备冲击综合最优的能力，而不是只在单一分项指标上占优。**

最新 dry-run 已经表明：框架当前阶段的主要瓶颈不是“fairness 拉不起来”，而是 **fairness 已恢复后，risk 与 fee recovery 仍未同步到位**。因此，本轮改造的重点应该从“继续放大公平性 shaping”转向“**把 fairness 从持续主驱动改造成阶段性约束，并在此基础上回补 risk 与 fee**”。

---

## 2. 基于最新 dry-run 的现实诊断

### 2.1 当前最新 dry-run 透露出的关键信号

最新 dry-run 的核心现象可以概括为：

- 本文方法的 **fairness 已经达到甚至略高于 FIFO**；
- `oldest_coverage` 与 `starvation_gap` 已经表现良好，说明 oldest backlog 服务问题基本得到缓解；
- 但 **risk_exposure 仍高于 FIFO、Heuristic 与 Fee-Risk Linear**；
- `composite_score` 仍未拿到第一，说明“综合最优”叙事尚不能成立；
- reward ablation 显示：一旦引入完整 fairness-aware reward，fairness 会继续提升，但 fee 往往下滑，说明当前机制容易把策略推向“公平优先但收益回补不足”的区域。

换言之，当前框架已经初步完成了“fairness recovery”，但还没有完成“composite optimality”。

### 2.2 当前最主要的缺陷不是单纯调参问题

本轮复核认为：

- **当前主要问题首先是框架机制问题，而不是单纯权重问题；**
- 单纯继续调 `beta_age / beta_terminal_fair / gamma_starvation`，大概率只会在 fairness 与 fee 之间继续来回拉扯；
- 若不引入新的训练协议、选模规则和终局 risk 对齐机制，仅靠调参很难让“综合最优”叙事自然成立。

### 2.3 当前框架的真实状态判断

当前实验框架更接近：

- 一个**公平性恢复型**的训练与评估体系；
- 而不是一个**在公平约束下做风险-收益回补**的训练与评估体系。

因此，本轮改造应将优化主轴调整为：

> **先保证 fairness floor，再压 risk，再回补 fee / packing ratio。**

---

## 3. 本轮代码改造总目标（修订版）

### G1：把 fairness 从“持续主奖励”升级为“阶段性约束”

训练阶段应先达到：

- fairness 不低于关键公平基线；
- oldest coverage 接近饱和；
- starvation gap 接近 0；

达到上述条件后，fairness 奖励应逐步退场，把优化重心转向：

- risk exposure 控制；
- fee recovery；
- packing ratio / gas utilization 保持。

### G2：让训练期 risk 目标与论文评估 risk 指标对齐

正式论文汇报使用的是：

- `risk_exposure`
- `top10_risk`
- `risky_rank`

因此训练阶段不应只依赖步级位置惩罚，而应补充与最终评估口径一致的终局 risk 惩罚或 risk-aware validation 协议。

### G3：把综合最优从“线性加权口头解释”升级为“约束式多目标协议”

实验输出中不应只保留单一 `composite_score`。必须支持：

- constrained fee（在 fairness / risk 约束下选 fee 最优）；
- hypervolume / Pareto 型协议；
- 逐 episode dominance 或可行域优势分析。

### G4：把“框架改造”和“调参”明确分阶段

本轮应明确：

1. **先改框架机制；**
2. **再在新框架内调参；**
3. **最后跑正式多 seed 实验。**

---

## 4. 需要改动的代码模块与方向

---

## 4.1 `experiments/env.py`：从 fairness recovery reward 升级到 fairness-constrained reward

### 目标

将当前 reward 从“持续推动公平性”的结构，升级为“**fairness 先达标，达标后把自由度让给 risk 与 fee**”的结构。

### 必改项

#### A. 引入 `fairness_gate` / `fairness_stage_weight`

当前 age / oldest_cover / terminal_fair / starvation 项默认持续生效。建议改为：

- 当 fairness proxy 尚未达标时，公平项保持较高权重；
- 一旦 fairness、oldest coverage、starvation gap 达到预设阈值，则逐步下调公平项权重；
- 下调可以是 hard gate，也可以是 annealing / sigmoid gate。

建议形式示意：

\[
\omega_{fair}(s_t) \in [0,1]
\]

并令：

\[
r_t = \alpha r_{fee} + \omega_{fair}(\beta_{age}r_{age}+\beta_{old}r_{old}) - \gamma_r r_{risk} + r_{packing}
\]

终局项同理：

\[
r_T = \omega_{fair}(\beta_T r_{fair}^{terminal}-\gamma_s r_{starvation}) - \lambda_r r_{risk}^{terminal}
\]

#### B. 新增终局 risk 对齐项

当前 risk 惩罚主要是步级位置惩罚。建议增加：

1. `r_terminal_risk_exposure`
2. `r_terminal_top10_risk`
3. `r_terminal_risky_rank_dev`

即直接按论文汇报口径对 episode 最终序列追加惩罚。

#### C. 新增 fee recovery / packing 相关项

最新 dry-run 表明 fairness 恢复后仍存在 fee 与 packing 不足的问题。建议新增：

1. `r_packing`
   - 对 gas utilization / packing ratio 做弱激励；
2. `r_late_fee_recovery`
   - 对“后到达但高手续费且未造成明显风险劣化”的交易给小幅奖励；
3. `r_unused_gas_penalty`
   - episode 结束时对显著未用 gas 施加惩罚。

#### D. 重构 STOP 惩罚

当前 STOP 惩罚仍偏“剩余均值 fee + fairness 风险”。建议改为更接近综合目标：

- 剩余 fee mass；
- 未用 gas 比例；
- 剩余高 fee / gas 候选强度；
- high-fee late tx 未服务比例；
- fairness 是否已达标（若已达标，则更强抑制过早 STOP）。

### 输出要求

- `info` 中新增 risk decomposition 与 packing decomposition；
- 训练日志中新增 `proxy_terminal_risk_penalty / proxy_packing_reward / proxy_unused_gas_penalty`。

---

## 4.2 `experiments/metrics.py`：从单一 composite 扩展到约束式协议指标

### 目标

让“综合最优”叙事不再只依赖一个线性加权分数，而是能在更强的协议下成立。

### 必改项

#### A. 保留现有指标链路

继续保留：

- `oldest_coverage_ratio`
- `starvation_gap`
- `tail_wait_reduction`
- `composite_score`

这些指标已经具备良好的公平性解释能力，不应删除。

#### B. 新增约束式主指标

建议新增：

1. `constrained_success_score(metrics, constraints)`
   - 若 fairness / oldest coverage / risk 满足约束，则返回 fee 或 composite；
   - 否则返回不可行标记或极低分。

2. `pareto_dominance_rate(ours, baseline)`
   - 统计逐 episode 上是否同时不劣于对手的多个核心指标；
   - 用于证明“可行域优势”而非单点优势。

3. `risk_adjusted_fee_score`
   - 在 fairness 已达标前提下，用更直接的 fee-vs-risk tradeoff 指标替代无约束线性和。

#### C. 保留 `composite_score` 但降级其叙事地位

`composite_score` 仍可作为辅助主指标，但正式结论应优先使用：

- constrained fee；
- hypervolume；
- pareto dominance；

而不是只靠单一线性加权分数得出“综合最优”。

---

## 4.3 `experiments/networks.py` 与 `experiments/env.py`：从 fairness summary 扩展到 risk-aware block summary

### 目标

当前 fairness block summary 已较完整，下一阶段应补齐 risk-aware context，帮助策略学会“**既不饿死老交易，也不把高风险交易堆到头尾**”。

### 建议新增 block state summary

在现有 fairness summary 基础上新增：

1. `selected_top10_risk_ratio`
2. `selected_edge_risk_ratio`
3. `selected_risky_rank_mean`
4. `remaining_risky_oldest_overlap_ratio`
5. `late_high_fee_unserved_ratio`

### 设计原则

- 继续保持低侵入，优先扩 block state，不急于大改 Actor-Critic 主体；
- 若状态增强后仍无明显改善，再评估 attention / set encoder。

---

## 4.4 `experiments/train.py`：必须补上 warm-start + curriculum + 多 checkpoint 协议

### 目标

避免 PPO 在 fairness 恢复后停留在“保守但不综合最优”的区域。

### 必改项

#### A. 新增 imitation warm-start

增加：

- `--pretrain-policy fifo | fair_fee | mixed`
- `--pretrain-epochs`

建议使用：

- `FIFO` 提供 oldest coverage / starvation 的基础行为；
- `Fair-Fee` 提供兼顾 fee 与 wait 的次优启发式；
- `mixed` 作为默认 warm-start 策略。

#### B. 新增三阶段 curriculum

1. **阶段一：fairness recovery**
   - 高公平项权重；
   - 中等 risk 惩罚；
   - 目标是 fairness 不输 FIFO / heuristic。

2. **阶段二：risk alignment**
   - 打开 / 强化终局 risk 惩罚；
   - fairness 保持 floor，不再继续线性放大；
   - 目标是压低 `risk_exposure` 与 `top10_risk`。

3. **阶段三：fee recovery / constrained optimization**
   - fairness gate 明显下降；
   - 保留 risk ceiling；
   - 重点回补 fee、packing ratio 与 composite / constrained fee。

#### C. 从单 checkpoint 升级到多 checkpoint 管理

建议训练过程中同时维护：

- `best_fairness_recovery.pt`
- `best_risk_aligned.pt`
- `best_constrained_fee.pt`
- `best_hypervolume.pt`

最后用固定 validation pools 统一比较。

#### D. 日志增强

每个阶段必须记录：

- `block_fee`
- `fairness`
- `risk_exposure`
- `oldest_coverage`
- `packing_ratio`
- `top10_risk`

并记录阶段切换 episode。

---

## 4.5 `experiments/evaluate.py`：把 fairness 分解扩展成“综合最优证据包”

### 目标

让正式评估能够回答：

- 是否已经恢复 fairness；
- 是否在 fairness 达标前提下回补了 fee；
- 是否把 risk 控制住了；
- 是否在可行域中形成优势。

### 必改项

#### A. 保留 fairness 分解输出

继续输出：

- `fairness_decomposition.json`
- `main_episode_metrics.json`

#### B. 新增约束式分析输出

建议新增：

1. `constrained_eval_summary.json`
   - 在 fairness / risk 可行域内比较各方法 fee；
2. `pareto_episode_analysis.json`
   - 逐 episode 分析 Ours 是否支配各 baseline；
3. `dominance_matrix.json`
   - 方法间 dominance 关系；

#### C. 新增图表输出

建议新增：

- fee–fairness scatter；
- fairness–risk scatter；
- Pareto frontier 图；
- per-episode feasible-region plot；
- oldest-decile service comparison。

---

## 4.6 `experiments/run_experiments.py`：从“公平恢复 track”扩展到“综合最优 track”

### 目标

让实验编排支持从 fairness recovery 平稳过渡到 composite optimality，而不是所有配置混在一起人工解释。

### 必改项

#### A. 引入正式 track 概念

建议至少保留以下两类正式配置：

1. `fairness_recovery_track`
   - 目标：先确认 fairness 是否稳定不落后 FIFO / heuristic；
   - checkpoint 协议偏 fairness floor。

2. `composite_optimal_track`
   - 目标：在 fairness floor 与 risk ceiling 满足的前提下，争取 fee / constrained score 最优；
   - checkpoint 协议偏 constrained fee / hypervolume。

可选保留：

3. `fifo_challenge_track`
   - 目标：在不显著恶化 risk 的条件下挑战 FIFO fairness。

#### B. 扩展 reward ablation 与 protocol ablation

除 reward ablation 外，新增：

- validation protocol ablation；
- fairness gate ablation；
- terminal risk ablation；
- warm-start / no-warm-start ablation；
- curriculum / no-curriculum ablation。

#### C. 正式输出“综合最优证据表”

新增：

- `table_constrained_main.tex`
- `table_pareto_main.tex`
- `table_protocol_ablation.tex`

---

## 4.7 `experiments/config.py`：参数体系从静态权重扩展到协议参数

### 目标

将配置体系从“固定 reward 权重”扩展到“**reward + 约束 + 阶段 + 选模协议**”四类参数。

### 建议新增配置项

#### 奖励与 gate 配置

- `FAIRNESS_GATE_TYPE`
- `FAIRNESS_GATE_THRESHOLD`
- `PACKING_REWARD_WEIGHT`
- `TERMINAL_RISK_EXPOSURE_WEIGHT`
- `TERMINAL_TOP10_RISK_WEIGHT`
- `TERMINAL_RISKY_RANK_DEV_WEIGHT`
- `UNUSED_GAS_PENALTY_WEIGHT`
- `LATE_FEE_RECOVERY_WEIGHT`

#### validation / selection 配置

- `VALIDATION_METRIC = composite_score | constrained_fee | hypervolume | pareto_score`
- `VALIDATION_FAIRNESS_FLOOR`
- `VALIDATION_OLDEST_COVERAGE_FLOOR`
- `VALIDATION_RISK_CEIL`
- `VALIDATION_TOP10_RISK_CEIL`

#### curriculum 配置

- `PRETRAIN_POLICY`
- `PRETRAIN_EPOCHS`
- `CURRICULUM_STAGE_EPISODES`
- `CURRICULUM_STAGE_METRICS`
- `CURRICULUM_FAIRNESS_GATE_SCHEDULE`
- `CURRICULUM_GAMMA_R_SCHEDULE`

---

## 5. 什么时候改框架，什么时候调参

### 5.1 本轮判断：先改框架，不先做大规模调参

当前问题首先是：

- 训练目标与评估口径不完全对齐；
- fairness 奖励缺少退场机制；
- 缺少 warm-start / curriculum；
- 缺少 constrained / Pareto 型正式协议。

这些都属于**框架缺口**，不是简单调 `beta` 与 `gamma` 能补齐的。

### 5.2 调参应在何时介入

调参应在以下条件满足后介入：

1. fairness gate 已实现；
2. terminal risk 对齐项已实现；
3. warm-start / curriculum 已接入；
4. constrained fee / hypervolume 选模已接通。

此时再重点调：

- `gamma_r`
- `fairness_floor`
- `risk_ceil`
- STOP penalty 相关权重；
- curriculum 各阶段长度与 schedule；
- PPO 学习率、entropy、validation interval。

### 5.3 推荐工作原则

本轮建议遵循：

> **先改框架，后调参；先做机制对齐，再做参数精修。**

---

## 6. 推荐实施顺序（修订版）

### P0：先修综合最优的根因

1. [ ] 在 `env.py` 中加入 fairness gate。
2. [ ] 在 `env.py` 中加入终局 risk 对齐项。
3. [ ] 在 `env.py` 中加入 packing / fee recovery / unused gas 项。
4. [ ] 在 `train.py` 中补 warm-start 与 curriculum。
5. [ ] 在 `train.py` 中加入多 checkpoint 协议。

### P1：再修正式协议

6. [ ] 在 `metrics.py` 中加入 constrained score / Pareto 指标。
7. [ ] 在 `evaluate.py` 中补 feasible-region / dominance 输出。
8. [ ] 在 `run_experiments.py` 中加入 `fairness_recovery_track` 与 `composite_optimal_track`。

### P2：再做状态增强

9. [ ] 在 `env.py` / `networks.py` 中加入 risk-aware block summaries。
10. [ ] 视效果决定是否升级网络结构。

### P3：最后做调参与正式多 seed

11. [ ] 在新框架内做小规模协议验证。
12. [ ] 再做 reward / gate / threshold 调参。
13. [ ] 最后跑正式 5-seed 主实验与鲁棒性实验。

---

## 7. 建议的首批实验矩阵（修订版）

### 7.1 机制对齐验证组

- 配置 A：当前框架 + constrained fee 选模
- 配置 B：当前框架 + hypervolume 选模
- 目的：确认仅换选模是否足以改善 composite

### 7.2 reward / protocol 联合验证组

- 配置 C：fairness gate + terminal risk
- 配置 D：配置 C + packing / fee recovery
- 配置 E：配置 D + constrained fee validation
- 目的：确认综合分下降是否主要来自 risk 还是 fee

### 7.3 训练策略验证组

- 配置 F：配置 E + warm-start
- 配置 G：配置 F + curriculum
- 目的：确认 warm-start / curriculum 是否能在 fairness 不退化前提下回补 fee 与 risk

### 7.4 正式判断标准

在进入正式 5-seed 前，应至少满足：

1. fairness 不低于 FIFO 或与 FIFO 无显著差异；
2. risk_exposure 明显低于当前 dry-run 基线版本；
3. constrained fee / hypervolume 至少达到全体第一梯队；
4. 主叙事不再依赖单一线性 composite 强行解释。

---

## 8. 一句话工作总结

本轮代码改造的真正任务不是“继续把 fairness 调高”，而是：

> **把实验框架从 fairness recovery 系统，升级为在 fairness 约束下做 risk-fee 回补的综合最优系统。**
