# 实验框架代码改动计划（2026-03-26）

---

## 0. 总体判断（为什么要改代码框架）

最新 dryrun 显示：

1. 主实验里本文方法收益/打包率/综合分数领先，但不是最低风险方法；
2. constrained 评估中 FIFO 排名第一，和“约束下我们最优”叙事冲突；
3. 单种子 + 多个指标贴边（1.0/0.0）导致统计与解释强度不足；
4. N=100 场景收益同质化，无法区分方法。

因此下一步不是“继续堆实验表”，而是**改实验框架协议与指标链路**，让结果天然对齐论文叙事。

---

## 1. 叙事对齐的四条改动主线（优先级从高到低）

### P0（必须先做）：统一“主结论协议”

**问题**：当前主结论在不同文件中可能由 `composite_score`、`constrained_fee`、`ranking` 各自驱动，容易出现叙事冲突。  
**改动目标**：固定一个主协议：

- 一级结论：`constrained_success_rate + feasible_set_fee`
- 二级结论：`risk_adjusted_fee_score + composite_score`
- 解释结论：`risk_exposure / top10_risk / risky_rank / fairness`

**代码改动点**：

- `experiments/metrics.py`
  - 新增 `feasible_set_fee_score()`：只在满足约束时累计 fee；
  - 新增 `feasible_rate()` 与 `violation_breakdown()`；
  - 统一 `summary_metric_bundle()` 输出字段，避免各脚本各算各的。
- `experiments/evaluate.py`
  - 主评估输出必须包含上述一级/二级/解释指标；
  - `constrained_eval_summary.json` 增加 `score_policy_version` 字段（可追溯评分版本）。

---

### P1（与 P0 并行）：修 constrained ranking 偏置

**问题**：当前 `ranking` 中 FIFO 第一，说明“可行率优先 vs 可行域内收益优先”的权重设计可能偏置。  
**改动目标**：让 constrained ranking 符合论文叙事：在满足约束前提下比较收益，而不是被“保守但低收益”策略统治。

**代码改动点**：

- `experiments/metrics.py`
  - 由“单一排序字段”改为两段式排序：
    1) 先按 `feasible_rate` 过滤至可比区间（例如 >= 0.3）；
    2) 再按 `constrained_fee_mean` 或 `risk_adjusted_fee` 排序。
  - 对 `-1.0` 这类不可行占位值改为 `null` + 独立计数，避免把失败值混入均值。
- `experiments/latex_tables.py`
  - 在约束主表中同时展示 `feasible_rate` 与 `constrained_fee_mean`，不再仅给 ranking。

---

### P2（必须做）：增强低区分场景与指标分辨率

**问题**：N=100 下收益同质化，`oldest_coverage`/`tail_wait_reduction` 贴边。  
**改动目标**：提高实验可分辨性，降低“指标饱和”风险。

**代码改动点**：

- `experiments/config.py`
  - 新增 `scenario_profile`：`light/default/heavy`；
  - 为 `N=100` 提供更高拥堵/更低 gas cap 的 profile（保证有排序压力）。
- `experiments/env.py`
  - 在 episode 输出中新增 `wait_p95`, `wait_p99`, `wait_gini`；
  - 保留旧公平指标用于兼容，但将贴边指标降级为辅助解释。
- `experiments/metrics.py`
  - 增加尾部等待汇总与跨方法差分（delta）输出。

---

### P3（建议做）：把“单种子 dryrun”与“正式统计”代码层强分离

**问题**：dryrun 结果容易被误用为正式显著性结论。  
**改动目标**：在代码层防止误报“显著优于”。

**代码改动点**：

- `experiments/stat_tests.py`
  - 新增硬门槛：`n_seeds < 3` 时默认不输出“显著性结论句”，仅输出“探索性统计”。
- `experiments/run_experiments.py`
  - 输出中写入 `evidence_level: dryrun|formal`；
  - `dryrun` 模式下自动给 markdown/latex 打水印注释（如 `Exploratory, single-seed`）。

---

## 2. 与论文叙事的逐条映射（改动后应达到）

### RQ1（主实验有效性）

- 目标叙事：高收益 + 可控风险折中（不是最低风险）。
- 框架保障：主表固定并列 `fee/fairness/risk/top10/packing + feasible_rate`，杜绝单指标胜利叙事。

### RQ2（鲁棒性）

- 目标叙事：中大规模稳定领先，小规模场景说明边界条件。
- 框架保障：profile 化场景生成 + `N=100` 强排序压力，避免同质化结果。

### RQ3（消融必要性）

- 目标叙事：组件改变 Pareto 权衡，不承诺单指标单调最优。
- 框架保障：消融报告默认输出 Pareto/feasible 维度，不仅输出 fee 单指标。

### RQ4（行为解释）

- 目标叙事：策略学习条件决策而非线性模板。
- 框架保障：保留 case study，同时增加尾部等待与风险位置分解，形成可解释证据链。

---

## 3. 建议的最小实现批次（两周）

### Batch A（1~2 天，必须）
- 改 `metrics.py`：新增 `feasible_rate / feasible_set_fee / violation_breakdown`；
- 改 `evaluate.py`：统一输出 bundle；
- 改 `latex_tables.py`：约束表并列可行率与收益。

### Batch B（2~3 天，必须）
- 改 `config.py` + `env.py`：scenario profile + 尾部等待指标（P95/P99/Gini）；
- 回归验证：N=100 不再收益全同。

### Batch C（1 天，建议）
- 改 `stat_tests.py` + `run_experiments.py`：evidence_level、dryrun 水印、低 seed 限制。

---

## 4. 验收标准（代码改完后必须检查）

1. `constrained_eval_summary.json` 中不再使用 `-1.0` 表示不可行均值；
2. 约束表至少包含：`feasible_rate / constrained_fee_mean / violation_count`；
3. N=100 场景下各方法 `block_fee_mean` 不再完全一致；
4. dryrun 模式输出自动标注 `evidence_level=dryrun`；
5. 单种子时 LaTeX 不再自动生成“显著优于”措辞。

---

## 5. 不建议本轮做的事（防止范围失控）

1. 不大改网络结构（先不引入复杂 attention/transformer）；
2. 不新增过多基线（先把协议做实）；
3. 不在旧协议上继续重调 reward 权重（先改协议再调参）。

---

## 6. 直接可执行的下一条命令建议（供研发同学）

1. 先实现 Batch A，并跑一次单 seed 回归：
   - 验证字段完整性 + ranking 逻辑正确性。
2. 再实现 Batch B，重点检查 N=100 可分辨性。
3. 最后补 Batch C，确保 dryrun/正式报告在文案层自动区分。


---

## 7. 基于 C1~C10 不一致点的修订（本次新增）

> 目的：把“问题清单”直接映射成“代码改动项”，避免文档停留在原则层。

### 7.1 C1/C3/C10（风险最优/公平全面最优/三目标统治叙事冲突）

**新增改动要求：**

1. `experiments/latex_tables.py`
   - 主表固定输出“冠军标注矩阵”：fee 最优 / fairness 最优 / risk 最优分别标注，禁止单行“本文方法全优”模板。
2. `experiments/evaluate.py`
   - 输出 `metric_winner_by_dimension` 字段（json），显式记录每个指标的最优方法。
3. `experiments/run_experiments.py`
   - 自动生成 `narrative_guard.json`：若“单方法全指标最优”为假，则阻止导出“全面领先”措辞模板。

### 7.2 C2（constrained ranking 与主叙事冲突）

**新增改动要求：**

1. `experiments/metrics.py`
   - 将 constrained 结果拆成三列：`feasible_rate`、`feasible_fee_mean`、`all_episode_fee_mean`；
   - `ranking` 从单一序切换为二维排序键：`(feasible_rate_tier, feasible_fee_mean)`。
2. `experiments/latex_tables.py`
   - 约束表增加 `infeasible_count` 与 `constraint_violation_top1`（最主要违约项）列。
3. `experiments/stat_tests.py`
   - 对 constrained 指标新增“仅在可行子集上”的 paired 检验，避免把不可行占位值带入统计。

### 7.3 C4/C8（小规模同质化 + 指标贴边）

**新增改动要求：**

1. `experiments/config.py`
   - 增加 `small_pool_stress_mode=true`：N<=120 时自动收紧 gas cap 或提高到达拥堵系数。
2. `experiments/env.py`
   - 增加 `tail_wait_p95/p99` 与 `wait_gini` 的 episode 统计输出。
3. `experiments/metrics.py`
   - 对贴边指标增加 `effective_variance_check`，若方差<阈值则在报告中标记“低区分度”。

### 7.4 C5/C6（消融并非单调）

**新增改动要求：**

1. `experiments/latex_tables.py`
   - 消融表默认附带 `pareto_tag`（如：Fee-lean / Fair-lean / Risk-lean），不再只给“是否优于 Full”。
2. `experiments/evaluate.py`
   - 为 ablation 输出 `delta_vs_full`（按全部核心指标）。
3. `experiments/run_experiments.py`
   - 若检测到“某消融在>=1项核心指标优于 Full”，自动附加文案：
     `Ablation indicates trade-off shift, not monotonic degradation.`

### 7.5 C7（单种子证据等级冲突）

**新增改动要求：**

1. `experiments/run_experiments.py`
   - 强制写入 `evidence_level`：
     - `dryrun_single_seed`
     - `multi_seed_exploratory`
     - `formal_multi_seed`
2. `experiments/stat_tests.py`
   - `n_seeds < 3` 时不输出 “significant superiority” 文案字段，仅输出 p 值与提示。
3. `experiments/latex_tables.py`
   - 表注自动追加：`This table is exploratory (single-seed).`（当 evidence_level 非 formal）。

### 7.6 C9（选择性呈现风险）

**新增改动要求：**

1. `experiments/latex_tables.py`
   - 统一生成三类表：
     - `table_main_core.tex`（主叙事核心）
     - `table_main_fullmetrics.tex`（全指标）
     - `table_main_constraints.tex`（约束）
2. `experiments/run_experiments.py`
   - 将三类表作为“完整叙事包”统一写入 `outputs_manifest.json`，缺一则标记 `report_incomplete=true`。

---

## 8. 文档级 DoD（Definition of Done）补充

在原第 4 节验收标准基础上，新增以下 DoD：

1. 结果目录中必须存在 `narrative_guard.json` 与 `outputs_manifest.json`；
2. 主实验必须同时产出 core/full/constraints 三表；
3. 单种子 dryrun 的所有 LaTeX 表均带 exploratory 表注；
4. 消融报告必须包含 `delta_vs_full` 与 `pareto_tag`；
5. constrained 排序逻辑可追溯（含 `ranking_policy_version`）。


---

## 9. 按“最新要求”更新：三目标尽可能最优（收益/公平/风险）

> 最新要求摘要：
> 1) 目标不是“单点收益最高”，而是三目标在业务约束下尽可能最优；
> 2) 当前阶段优先补齐 fairness（可行率瓶颈）；
> 3) 实验框架要支持 Aggressive / Balanced / Conservative 三档可切换策略。

### 9.1 新的主协议（替换旧“单主分数”心智）

统一采用“两层最优协议”：

1. **硬约束层（必须先过）**
   - fairness >= 0.90
   - risk_exposure <= 0.30
   - top10_risk <= 0.30
2. **软目标层（在可行集合内优化）**
   - 优先 `feasible_rate`
   - 其次 `feasible_set_fee_mean`
   - 并列时 `risk_adjusted_fee_score`

### 9.2 代码改动点（新增/强化）

#### A) `experiments/config.py`

新增“运行档位配置”（用于三档策略切换）：

- `OPERATING_MODE = "balanced"`（`aggressive|balanced|conservative`）
- 三档权重模板：
  - aggressive：fee 权重高，fairness/risk 约束保持不变；
  - balanced：默认档，强调 feasible_rate 与可行域收益；
  - conservative：fairness/risk 惩罚增强。

并新增阈值分组配置：

- `CONSTRAINT_PROFILE = "strict"|"relaxed_for_training"`
- 训练可用 relaxed（仅训练阶段），评估固定 strict。

#### B) `experiments/train.py`

1. 增加阶段化训练调度（fairness-first curriculum）：
   - Stage 1：提高公平奖励/门控影响；
   - Stage 2：恢复收益权重，做 fee 回补；
2. 增加 checkpoint 选择策略开关：
   - `--val-metric constrained_fee|hypervolume|two_stage`；
   - `two_stage`：先筛可行率，再筛可行域收益。

#### C) `experiments/metrics.py`

新增统一打分接口：

- `two_stage_selection_score(metrics, constraints, mode)`
  - 输出：`feasible_rate_tier`, `feasible_set_fee_mean`, `risk_adjusted_fee`
- `operating_point_rank(payload, mode)`
  - 输出三档模式下的排序结果（用于部署与论文表格）

并保证所有不可行值不混入均值（保持 `None/null` + 计数）。

#### D) `experiments/evaluate.py`

新增输出：

- `operating_points_summary.json`
  - `aggressive/balanced/conservative` 三档各自的 top method、关键指标、可行率；
- `constraint_bottleneck_report.json`
  - 自动给出各方法 top1 违约项（当前应为 fairness）。

#### E) `experiments/latex_tables.py`

新增表格：

- `table_operating_points.tex`（三档策略对比）
- `table_constraint_bottleneck.tex`（违约主因）

并在主表脚注固定声明：

- “Primary decision rule: feasible-rate first, feasible-set fee second.”

### 9.3 实施优先级（按你当前问题排）

**P0（本周必须）**

1. `metrics.py`：补 `two_stage_selection_score` 与 `operating_point_rank`；
2. `evaluate.py`：产出 `constraint_bottleneck_report.json`；
3. `run_experiments.py`：在 summary 写入 `operating_mode` 与 `selection_policy_version`。

**P1（紧随其后）**

1. `train.py`：加 `two_stage` 验证指标与 fairness-first curriculum 开关；
2. `config.py`：落地三档 operating mode 配置模板。

**P2（报告层）**

1. `latex_tables.py`：生成 operating points 与 bottleneck 两张新表；
2. `outputs_manifest.json`：要求新表与新 json 缺一不可。

### 9.4 新验收标准（替换“只看综合分”）

1. 任一正式结果目录必须包含：
   - `constrained_eval_summary.json`
   - `operating_points_summary.json`
   - `constraint_bottleneck_report.json`
2. 默认 `balanced` 模式下，排序遵循：
   - feasible_rate 优先，fee 次优先；
3. 结果报告必须给出三档策略（A/B/C）并明确推荐默认档（Balanced）；
4. 若 fairness 仍为 top1 违约项，训练日志必须提示“进入 fairness-first 调参轮次”。

### 9.5 与你当前决策问题的直接对应

- 你问“三项尽可能最优先优化哪个”：框架层将其固化为**fairness-first**（先提可行率，再收敛收益）。
- 你问“平时最优怎么选”：框架层将输出三档 operating points，默认给 `balanced`。
- 这样后续实验不再靠人工解释，而是由代码协议直接产出可执行结论。


## 10. 实验预期结果（你可以直接拿来做里程碑）

### 10.1 Phase A（调参优先）预期

在不改大框架前，预期出现以下变化：

1. `balanced` 模式下 `feasible_rate` 明显上升（相对当前基线提升为主要目标）；
2. `fairness_floor` 仍可能是 top1 违约项，但违约次数占比应下降；
3. `block_fee_mean` 可能有小幅回落，但 `feasible_set_fee_mean` 与 `risk_adjusted_fee` 应更稳定；
4. 三档 operating points（A/B/C）之间形成清晰梯度：
   - Aggressive：fee 最高，可行率较低；
   - Balanced：可行率与收益折中最优（推荐默认）；
   - Conservative：风险最低/公平最高，但收益最低。

### 10.2 Phase B（条件触发的小框架改动）预期

若 Phase A 两轮后仍受 fairness 瓶颈限制，进入 Phase B 后预期：

1. `constraint_bottleneck_report.json` 中 fairness 占比继续下降；
2. `balanced` 档与 `conservative` 档的 gap 缩小（说明公平提升不再依赖极端保守策略）；
3. 在不显著恶化 risk 的前提下，`feasible_rate` 进入稳定高位区间；
4. 论文主叙事从“单点最好”转为“可控 Pareto 前沿 + 可切换部署策略”。

### 10.3 最终交付层预期（报告可读性）

1. 每次正式实验都能自动产出：
   - `table_main_core.tex`
   - `table_main_fullmetrics.tex`
   - `table_main_constraints.tex`
   - `table_operating_points.tex`
   - `table_constraint_bottleneck.tex`
2. 读者可以在一页内看到：
   - 谁在收益最强；
   - 谁在公平/风险更稳；
   - 默认部署为什么选 `balanced`。

> 一句话预期：实验最终应从“指标拼盘”升级为“可行率优先、三档可部署、叙事可复现”的结果体系。