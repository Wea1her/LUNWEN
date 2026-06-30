# 期刊论文大纲（V5 同步版）

## 题目

**基于深度强化学习的以太坊交易序列决策排序方法研究**

---

## 一、摘要

### 摘要应回答的问题

1. 交易排序为什么是区块链系统中的关键问题；
2. 传统 FIFO / Gas 优先 / 启发式方法的局限；
3. 本文方法在状态、动作、奖励和约束上的核心设计；
4. 实验是否支持“收益-公平-风险折中改进”而非“全指标最优”。

### 摘要口径

- 结果表述采用“综合折中优势 + 适用边界”；
- 不使用“所有指标均最优”措辞；
- 若证据等级为 exploratory，明确标注探索性结论。

---

## 二、引言

### 2.1 研究背景

- 区块构建中的交易排序影响收益分配、公平体验与风险暴露；
- MEV 场景放大排序策略差异；
- 排序问题本质是多目标冲突决策。

### 2.2 问题与挑战

- 收益提升与公平约束存在张力；
- 风险控制与手续费最大化难以同时最优；
- 规则方法可解释但适应性有限。

### 2.3 研究目标

在 gas、nonce 与风险约束下，学习一种可执行的交易序列策略，使收益、公平与风险达到更优折中。

### 2.4 贡献概述

- 将排序建模为逐笔选择序列决策；
- 提出可行率优先的两阶段评估协议；
- 建立主实验、鲁棒性、消融、行为解释一体化验证链路。

---

## 三、问题建模

### 3.1 MDP 建模

- 状态：交易特征 + 区块摘要；
- 动作：选择交易或 STOP；
- 约束：公平下限、风险上限、Top10 风险上限。

### 3.2 目标函数

- 主目标：收益、公平、风险折中；
- 约束层：先判定可行，再比较可行域收益；
- 解释层：风险位置与等待尾部指标。

---

## 四、方法设计

### 4.1 策略网络与训练

- PPO 训练框架；
- 动作掩码与 STOP 联合控制；
- 可选 fairness-first curriculum（阶段化训练）。

### 4.2 选择与部署协议

- 默认 checkpoint 指标为 `two_stage`；
- 两阶段选择：`feasible_rate` 优先，`feasible_set_fee` 次优先，`risk_adjusted_fee` 辅助；
- 运行档位：`aggressive / balanced / conservative`。

---

## 五、实验设计

### 5.1 研究问题

- RQ1 主实验有效性；
- RQ2 鲁棒性；
- RQ3 消融必要性与权衡转移；
- RQ4 行为解释。

### 5.2 默认场景

- pool size=300；risk ratio=15%；risk fee multiplier=2.0；
- seeds=5；eval episodes=1000；
- train episodes 采用“两阶段收敛”：先 2200，再对晚收敛 seed 补跑到 3000。

### 5.3 基线与对照

- FIFO / Gas Priority / Heuristic Risk-Aware / Fee-Risk Linear / Fair-Fee Greedy；
- V5 强基线：Center-Insertion Heuristic / Dynamic Tri-Objective Greedy；
- 含参数基线在固定验证池上调参。

### 5.4 评价指标

- 主指标：`block_fee`, `fairness`, `old_tx_pack_rate`, `risk_exposure`, `edge10_risk`, `risky_inclusion_rate`, `gas_util`, `packing_ratio`；
- 约束指标：`feasible_rate`, `feasible_fee_mean`, `violation_count`；
- 诊断指标：非法动作率、非法截断率、连续非法动作长度、推理时间；
- 解释指标：`wait_p95/p99`, `wait_gini`, `risky_rank`, `late_promo`。

### 5.5 证据等级

- `dryrun_single_seed`、`multi_seed_exploratory`、`formal_multi_seed`；
- 非 formal 结果仅作探索性证据。

---

## 六、结果呈现结构

### 6.1 主实验表格

- `table_main_core.tex`（V5 核心结论）；
- `table_main_fullmetrics.tex`（全指标和诊断指标）；
- `table_main_constraints.tex`（约束结果）；
- `table_baseline_params.tex`（验证池调参结果）；
- `table_seed_level_significance.tex`（seed 级正式统计）。

### 6.2 新增策略与瓶颈表

- `table_operating_points.tex`（三档策略 top method 对比）；
- `table_constraint_bottleneck.tex`（违约主因）。

### 6.3 结果口径

- 主结论基于“可行率优先、可行域收益次优先”；
- 同步给出支持结论与边界条件；
- 报告不回避失败维度与权衡代价。

---

## 七、消融与行为解释

### 7.1 奖励/结构消融

- 奖励消融采用 `AgeOnly / Age+Risk / Age+TerminalFair / FullBalanced`；
- 结构消融采用 `No-SeqSummary / No-ActionMask / No-STOP / Ours-Full`；
- 若某消融在部分指标优于 Full，解释为 trade-off shift，而非方法退化矛盾。

### 7.2 行为解释

- case study + paired deltas；
- 关注高风险位置迁移与尾部等待变化。

---

## 八、结论与展望

### 8.1 结论

- 本文方法在默认协议下体现稳定的综合折中优势；
- 结果不支持“全指标统治”结论；
- 约束可行率是当前性能瓶颈与后续优化重点。

### 8.2 展望

- 更真实链上数据与部署评估；
- 面向 DEX/PBS/FSS 的扩展；
- 约束强化学习与多智能体博弈建模。

## 九、2026-06-30 同步回填

论文大纲已同步 V5 当前实现：分层奖励、7 类基线、验证池调参、seed 级统计、No-ActionMask 诊断和推理时间指标均已纳入第 4、5、6、7 部分的写作口径。
