# 写作备注

## 2026-03-22

- 风险位置比的论文口径以 `experiments/env.py` 的实现为准，不再使用“候选池规模”作为分母。
- 第 3 章中 $\rho_t$ 已改写为基于平均 Gas 估计的区块可容纳交易数，用于表达更接近区块内位置的风险惩罚定义。
- 论文中不再使用“达到最大选择步数”作为终止条件，终止逻辑以环境实现中的 STOP / 无合法动作 / 候选为空 / 非法动作提前结束为准。
- 正式实验、鲁棒性实验与消融实验统一使用 `best_model.pt` 作为评估 checkpoint；`final_model.pt` 仅保留为训练末轮快照。
- 统计口径当前以 `experiments/stat_tests.py` 的实现为准，正文默认表述为 Welch t 检验、Cohen's d 与 95% 置信区间，不再写 Mann-Whitney U。
- 引言与问题建模中的真实链上执行语义已弱化，当前论文定位为通用候选池排序仿真验证，不宣称已显式模拟合约状态回滚或 DEX/AMM 执行反馈。

## 2026-03-23

- 第 4 章 checkpoint 选模描述已更新为“固定验证池 + 周期评估 + 验证指标选优”，与 `experiments/train.py` 和 `checkpoint_meta.json` 一致。
- 第 5 章统计口径已更新为 episode 级配对检验（paired t-test + Wilcoxon signed-rank），并保留跨 seed 均值±标准差作为稳定性报告。
- 第 5 章环境描述已更新为 `correlated_v1` 交易池生成机制，明确 `fee/risk/arrival/hist_delay` 的相关性建模。
- 第 3 章 `r_t^{valid}` 已改为同时覆盖“提前 STOP 惩罚”和“非法动作惩罚”。

## 2026-06-30

- 当前正式口径升级为 V5：默认验证指标为 `two_stage`，正式结论以独立训练 seed 为统计单位。
- `paired_significance_tests.json` 仍可生成，但只作为共享测试池 episode 级诊断；正文显著性结论应引用 `seed_level_paired_tests.json` 和 `table_seed_level_significance.tex`。
- 奖励函数在论文中按 `step + terminal + valid` 分层描述，不再使用旧四项奖励作为实现公式。
- 奖励消融统一使用 `AgeOnly / Age+Risk / Age+TerminalFair / FullBalanced`。
- V5 强基线默认启用：`Center-Insertion Heuristic` 与 `Dynamic Tri-Objective Greedy`；`center_aware` 仅保留为 legacy alias。
- 论文不声明真实 MEV 防护效果，MEV 仅作为位置敏感排序风险的动机之一。
