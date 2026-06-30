# 多目标最优可行性分析（基于 `results_formal_full_20260324_r1`）

## 1) 结论先行

- **可行性判断：有条件可行（不是“不可做”，而是当前约束下可行率不足）**。在当前四项硬约束（fairness≥0.9, oldest_coverage≥0.9, risk≤0.3, top10_risk≤0.3）下，`ours` 的可行率为 **23.97%**，明显低于 `fifo` 的 **85.00%**，主瓶颈是公平性约束。  
- **是否先改框架还是先调参：先调参，再做小幅框架改动**。当前结果显示框架已经具备明显多目标 trade-off 能力（对多数基线有支配优势），但“公平性地板”触发频率高，优先通过奖励权重/门控阈值/验证指标组合进行针对性调参，预计更快提升可行率；若两轮调参后可行率仍 <40%，再进入框架级改动。  

---

## 2) 数据证据（为什么说“有条件可行”）

### 2.1 综合表现与冲突结构

- `ours` 在收益与装载方面领先：
  - `block_fee_mean = 1777.54`（高于全部基线）
  - `packing_ratio_mean = 0.6926`（高于全部基线）
  - `composite_score_mean = 0.6236`（最高）
- 但公平性低于 `fifo`：
  - `ours fairness_mean = 0.8789`
  - `fifo fairness_mean = 0.9340`

这说明模型已经学到“高收益+较高打包率”的策略，但在严公平阈值下可行域覆盖不足。  

### 2.2 约束可行率是当前最大问题

在 constrained 评估里（3000 episodes）：

- `ours`：`feasible_rate = 0.2397`，`infeasible_count = 2281`
- `fifo`：`feasible_rate = 0.85`
- `fee_risk_linear`：`feasible_rate = 0.00033`

`ours` 的违规分解显示：

- `fairness_floor` 违规 **2137** 次（占主导）
- `risk_ceil` 违规 387 次
- `top10_risk_ceil` 违规 8 次
- `oldest_coverage_floor` 违规 0 次

=> 说明不是“整体失效”，而是“主要卡在公平性门槛”。  

### 2.3 Pareto 关系支持“可优化”而不是“推倒重来”

两两支配率（ours 对 baseline）显示：

- 对 `gas`：`ours_dominates_rate = 85.17%`
- 对 `fair_fee`：`67.37%`
- 对 `heuristic`：`20.13%`
- 对 `fee_risk_linear`：`10.80%`
- 对 `fifo`：`0.033%`（基本互不支配）

关键点：`ours` 与 `fifo` 近乎互不支配，说明你面对的是典型 Pareto 前沿冲突，不是单边失败。  

### 2.4 敏感性：公平阈值轻微放松时可行率明显上升

在相同 episode 数据上做阈值扫描：

- `ours`：
  - fairness floor 0.90 -> feasible rate **0.2397**
  - fairness floor 0.88 -> **0.4283**
  - fairness floor 0.87 -> **0.5287**

这说明当前策略分布离 0.90 的公平阈值“差一点点”，更像校准问题而非能力缺失。  

---

## 3) 应该先做什么：调参优先（两轮），再决定是否改框架

## Phase A（优先，低成本）— 只动参数与训练策略

目标：把 `ours` 在 fairness floor=0.90 下 feasible rate 从 0.24 提升到 >=0.40，同时尽量保持 block_fee 领先。

建议参数方向（按优先级）：

1. **提高公平性相关奖励强度**（先小步）
   - `BETA_TERMINAL_FAIR`: 0.35 -> 0.45/0.50
   - 观察 fairness_mean 与 block_fee_mean 的斜率变化。
2. **调公平门控阈值/形状**（让训练更早感知公平风险）
   - `FAIRNESS_GATE_THRESHOLD`: 0.92 -> 0.90
   - `FAIRNESS_GATE_SHARPNESS`: 18 -> 12~15（减小过硬切换）
   - `FAIRNESS_GATE_MIN`: 0.35 -> 0.45
3. **验证指标切换实验**
   - 从纯 `hypervolume` 对照加入 `constrained_fee` 选择 checkpoint（至少做 A/B）
4. **小规模 curriculum**
   - 早期放宽公平阈值，后期收紧到 0.9，减少策略初期陷入“重收益局部最优”。

建议用 2x2 小网格（每格 1~2 seeds）先筛，再放大。  

## Phase B（触发条件式）— 若 A 两轮后仍不达标，再做框架小改

触发条件：`fairness>=0.9` 下 feasible rate 仍 <0.40，且提升趋于饱和。  

再做这两类结构改动（保持最小侵入）：

1. **Lagrangian/Primal-Dual 约束学习**
   - 将 fairness/risk 违规转为动态乘子惩罚，避免固定权重下某目标长期被压制。
2. **双头策略或条件策略（收益头 + 公平头）**
   - 推理时按约束状态做 head mixing，提高边界 case 的可行性通过率。

---

## 4) 如果你问“当前可不可继续做多目标最优？”

- **答案：可以继续，而且值得继续。**
- 你现在不是“模型没学会多目标”，而是“在严格公平阈值 0.9 下可行率不够高”。
- 这类问题通常先通过参数与验证策略对齐就能拿到一轮明显增益；只有当调参增益耗尽时，才值得改框架。  

---

## 5) 下一步执行清单（可直接开跑）

1. 固定当前代码框架，建立 `fairness-priority` 调参分支。  
2. 跑 Phase A：
   - 2x2 参数网格（`BETA_TERMINAL_FAIR` × `FAIRNESS_GATE_THRESHOLD`）
   - 每格至少 2 seeds，记录 feasible_rate / block_fee / fairness。  
3. 选出 Pareto 最优参数，再做 3~5 seeds 复验。  
4. 若 feasible_rate 仍 <0.40，则进入 Phase B 的 Lagrangian 约束版。  
5. 最终报告按“两套结果”呈现：
   - “高收益优先配置”
   - “高可行率配置”

这样论文叙事会更完整：你不是追求单点最优，而是展示可控的 Pareto 解族。  


## 6) 回答你的问题：论文叙事是不是“天然”在公平性上要差？

短答案：**不天然会差，但如果你把主叙事放在“收益最优”，在同一模型同一配置下通常会牺牲一部分公平性，这是多目标冲突的常见现象，不是你论文的问题。**

更准确的叙事方式是：

1. **承认 Pareto 冲突，不回避公平性短板**
   - 你现在的数据已经显示 `ours` 在收益/装载领先，但 fairness 均值低于 `fifo`。
   - 这应当被写成“可解释的 trade-off”，而不是“方法失败”。

2. **把“单点最优”改成“前沿可控”叙事**
   - 不要只展示一个 checkpoint；要展示两类配置：
     - 收益优先（higher fee）
     - 可行率/公平优先（higher feasible-rate）
   - 这样评审会看到你能沿 Pareto 前沿移动，而不是卡死在一个点。

3. **把公平性当作可调设计目标，而非被动结果**
   - 通过 `BETA_TERMINAL_FAIR`、fairness gate 参数和验证指标切换，证明公平性可以被“系统性抬升”。
   - 只要你能给出“公平性提升曲线 + 收益损失可控区间”，叙事就会更强。

4. **建议在论文中加一句“研究定位”**
   - 本文目标不是在所有目标上同时绝对最优，而是在给定约束与业务偏好下学习可调的 Pareto 解族。

结论：

- 你的叙事不需要也不应该被理解为“公平性天然差”。
- 更专业的表述是：**当前配置位于 Pareto 前沿的收益偏好区域；通过约束/奖励再平衡可迁移到公平偏好区域。**

## 7) 你问的“收益-公平-风险平时（实务）最优”怎么选？

如果你说的是“日常运行时的平衡最优”（不是单次冲榜），建议用**两层最优**：

1. **先保约束可行（硬约束层）**
   - fairness >= 0.90（或业务可接受阈值）
   - risk_exposure <= 0.30
   - top10_risk <= 0.30

2. **再做效用最优（软目标层）**
   - 在可行集合里最大化 `block_fee_norm - \lambda * risk_exposure`（或你现有的 constrained fee / risk-adjusted fee）

按你当前结果，给一个务实建议：

- **训练与评估分离**：训练仍可用 hypervolume 保多样性，但上线/主结论用 constrained ranking（先可行率再可行域收益）。
- **部署默认档位**：不要直接用“最高收益点”，而用“可行率优先 + 收益次优”的 operating point。
- **报告里给三档策略**：
  - Aggressive（收益优先）
  - Balanced（推荐默认）
  - Conservative（风险/公平优先）

这样“平时最优”就不是某一个固定参数，而是**在业务约束下可切换的策略族最优**。

## 8) 如果目标是“收益-公平-风险三项尽可能最优”，现在该重点优化哪个？

基于你当前结果，**第一优先级应是公平性（fairness）**，理由很直接：

- 约束不可行主要由 `fairness_floor` 触发（远高于 risk/top10 risk 违规次数）。
- `ours` 在收益（block fee）和装载（packing）已经是优势项，继续单纯提收益的边际价值低于补齐公平短板。
- 一旦公平性跨过阈值（如 0.90），整体可行率会显著提升，你的“综合最优”才有实际意义。

### 建议的优化优先顺序（按阶段）

1. **先提 fairness（主攻）**
   - 目标：fairness_mean 向 0.90 靠近，并把 fairness 违规率明显压低。
2. **再控 risk（协同）**
   - 保持 risk_exposure / top10_risk 不反弹，避免“补公平带来风险恶化”。
3. **最后稳 fee（收敛）**
   - 在可行率达标后，再做收益回补（fine-tune fee）。

### 一个实用判据（你可以直接拿来筛选实验）

在 fairness>=0.90, risk<=0.30, top10_risk<=0.30 的前提下：

- 先按 feasible_rate 排序（越高越好）；
- feasible_rate 接近时，再按 feasible_set_fee_mean / risk_adjusted_fee 排序。

一句话：

- **当前阶段重点优化 fairness，不是 fee。**
- 你的收益已经够强，短板在公平约束穿透能力；先把可行率做起来，三目标“尽可能最优”才成立。

## 9) 2026-06-30 V5 落地回填

该分析中的核心判断已经转化为 V5 实验协议和代码实现：

- 主线默认选模改为 `two_stage`，不再把 `hypervolume` 作为默认正式口径；
- `hypervolume` 保留为协议消融或对照设置，用于分析不同选模规则带来的 trade-off；
- 含参数基线已在固定验证池上调参，避免 PPO 与基线比较不公平；
- 正式统计改为以独立训练 seed 为单位，避免把测试 episode 误当作独立重复；
- 论文结果叙事从“单点最优”调整为“可行率优先下的多目标折中”。

后续若继续优化 fairness，可基于 V5 当前输出的 `constraint_bottleneck_report.json`、`operating_points_summary.json` 和 `seed_level_statistics.json` 决定是否进入约束强化学习或动态乘子版本。
