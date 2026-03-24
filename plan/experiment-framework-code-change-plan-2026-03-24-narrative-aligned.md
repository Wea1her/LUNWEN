# 实验框架代码改动计划（2026-03-24，面向“论文整体叙事一致性”）

> 输入依据：`results_dryrun_gpu_20260324_defaultgpu` 最新 dryrun。  
> 目标：将代码框架从“能跑 + 局部指标好看”升级为“可稳定支撑论文叙事：高收益下的可控风险折中，并在约束协议下具备一致优势”。

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

