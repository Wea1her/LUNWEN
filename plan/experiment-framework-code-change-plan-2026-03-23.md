# 实验框架代码改动文档（2026-03-23）

## 1. 文档目标

本文档用于指导下一轮实验框架代码改造，目标不是做局部调参，而是让实验系统能够更真实地支撑以下两类论文叙事：

1. **本文方法在收益、公平与风险控制之间实现更强的综合最优表现；**
2. **本文方法在公平性上不再明显落后于 FIFO / 启发式基线，至少能够稳定逼近甚至超过关键公平基线。**

当前 dryrun 已经说明：仅靠现有 `fee + ΔJain - risk` 的奖励形式与 `block_fee` 单指标选模，难以自然导出上述叙事。因此，本次改造应覆盖**奖励定义、状态建模、选模规则、评估口径与实验编排**五个层面，而不应继续停留在单纯权重微调。

---

## 2. 当前框架存在的核心问题

### 2.1 奖励中的公平性信号是“局部 shaping”，不是“终局公平目标”

当前环境中公平性奖励采用 `ΔJain(selected)`，它衡量的是“把当前交易加入已选集合后，局部 Jain 指数变化了多少”，而不是“是否优先服务了更老的交易”或“最终 fairness 是否更高”。

这会导致两个问题：

- 它更像是 selected 集合内部等待时间分布的局部平滑项；
- 它没有直接惩罚 oldest backlog 被长期遗留在候选池中的情况。

### 2.2 选模规则仍是 fee-first

当前 best checkpoint 按固定验证池上的 `block_fee_mean` 选择。即使训练过程中某些 checkpoint 在 fairness 与 risk 上更平衡，只要收益不是最高，就不会成为正式评估使用的模型。

### 2.3 状态表示缺少 backlog fairness 上下文

当前 block state 主要包含：

- 剩余 gas；
- 已选比例；
- 累计手续费；
- 已选序列的均值摘要。

但缺少以下与公平性直接相关的全局上下文：

- oldest transaction 的等待时间；
- backlog 等待时间分布；
- oldest decile / quintile 的未服务比例；
- sender 级 starvation 统计。

这意味着 Actor 虽然能看到单笔交易 arrival feature，但不容易学到“当前是否已经开始饿死老交易”这种全局判断。

### 2.4 综合最优缺少正式判据

当前表格主要报告分项指标，没有一个预注册的“综合目标”作为主排序标准。因此即便方法在 fee 上很强、在 risk 上优于 Gas Priority，也很难直接写成“收益-公平-风险全面综合最优”。

### 2.5 实验编排没有针对公平叙事提供专门产物

当前编排可以输出主实验、鲁棒性、消融与行为案例，但没有专门产出：

- 终局 fairness 相关分层分析；
- oldest backlog 覆盖率；
- starvation 指标；
- 综合 score / hypervolume 表格。

---

## 3. 本轮代码改造总目标

本轮实验框架代码改造的总目标如下：

### G1：把公平性从“弱 shaping”升级为“显式优化对象”

训练阶段应同时感知：

- 老交易是否被服务；
- oldest backlog 是否被覆盖；
- 最终 fairness 是否改善；
- 是否存在 starvation。

### G2：把 best checkpoint 选择从 fee-first 升级为 multi-objective

正式评估所用模型应由多目标验证协议选择，而不是继续按单一 fee 选出。

### G3：把“综合最优”落成可验证主指标

实验输出中必须有一个**事先定义**的 composite score 或 Pareto 型综合判据，避免后验口头解释。

### G4：把公平性解释性分析纳入正式产物链路

编排脚本应能直接输出 oldest coverage / starvation / fairness decomposition 等结果，供论文正文和附录直接引用。

---

## 4. 需要改动的代码模块与方向

---

## 4.1 `experiments/env.py`：重构奖励定义与 STOP 机制

### 目标

把当前

\[
r_t = \alpha r_{fee} + \beta \Delta r_{fair} - \gamma_r r_{risk}
\]

重构为“**即时 age-aware + 终局 fairness-aware + starvation-aware**”的奖励体系。

### 建议改动

#### A. 保留 fee 项

- 保留当前 `r_fee = tx.fee / max_fee`；
- 作为收益主项，不建议删除。

#### B. 将当前 `ΔJain` 改为混合公平项

建议新增以下子项：

1. `r_age`：等待时间奖励
   - 直接基于 `t_now - tx.arrival_time` 归一化；
   - 用于鼓励服务更老交易。

2. `r_oldest_cover`：oldest-q 覆盖奖励
   - 若本步选择的交易属于当前 oldest 10% / 20% 候选集合，则给奖励；
   - 用于抑制 oldest backlog 被长期忽视。

3. `r_terminal_fair`：episode 终局 fairness 奖励
   - 在 episode 结束时一次性加入；
   - 使用与论文汇报一致的最终 fairness 指标（可先保留 Jain）。

4. `r_starvation`：oldest backlog 残留惩罚
   - episode 结束时统计 oldest-q 中未被服务的比例；
   - 残留越多，惩罚越大。

#### C. 风险项保留但重新参数化

- 保留当前头尾 / 中部位置敏感的风险惩罚；
- 但允许把 `gamma_r` 下调，并让 position penalty 更平滑，避免它在训练初期过度主导排序。

#### D. STOP 惩罚增加 fairness 约束

当前 STOP 惩罚主要看剩余 fee。建议改为：

- 剩余 fee 均值；
- oldest 未服务比例；
- oldest wait mass；
- 当前打包比例。

这样 STOP 不再只是“漏不漏收益”，而是同时考虑“会不会把最老交易留在池里”。

### 输出要求

- `info` 中新增每个 episode 的 fairness reward decomposition；
- 训练日志中新增 `proxy_age_reward / proxy_oldest_cover / proxy_starvation_penalty`。

---

## 4.2 `experiments/metrics.py`：补充公平性与综合性指标

### 目标

让论文中“公平性更优”“综合最优”的叙事有正式指标承载，而不是只靠现有 Jain 一项解释。

### 建议新增指标

#### A. `oldest_coverage_ratio(selected, pool, q=0.2)`

- 定义：候选池中 oldest 20% 交易被打包的比例；
- 作用：衡量是否真实服务了最老交易。

#### B. `starvation_gap(selected, pool, q=0.2)`

- 定义：oldest 20% 中未被服务的比例；
- 作用：直接量化饿死效应。

#### C. `tail_wait_reduction(selected, pool)`

- 定义：选中交易与全池交易在高分位等待时间上的改善程度；
- 作用：反映方法是否降低长尾等待压力。

#### D. `composite_score(metrics)`

建议预注册为：

\[
C = w_1 \tilde{fee} + w_2 \tilde{fairness} - w_3 \tilde{risk} + w_4 \tilde{oldestCoverage}
\]

其中 `tilde` 为标准化或 min-max 归一化。

### 输出要求

- `compute_all_metrics()` 支持上述新指标；
- 主实验、鲁棒性、消融结果都允许选择性输出 composite score；
- LaTeX 表格生成逻辑可按配置决定是否显示这些新指标。

---

## 4.3 `experiments/networks.py` 与 `experiments/env.py`：增强状态表示

### 目标

让策略网络显式接收 backlog fairness 相关上下文，而不是只依赖单笔交易 feature 与已选序列均值。

### 建议新增 block state summary

建议在 block state 中新增：

1. `oldest_wait_norm`
2. `p90_wait_norm`
3. `mean_wait_selected`
4. `std_wait_selected`
5. `oldest20_unserved_ratio`
6. `selected_oldest20_ratio`
7. `late_selected_ratio`
8. `sender_starvation_max`
9. `sender_starvation_mean`

### 网络层面建议

- 允许 `block_dim` 增长；
- 保持当前 MLP Actor-Critic 结构不变作为第一阶段；
- 如效果仍不理想，再评估是否引入 attention / set encoder 替代均值池化 critic。

### 设计原则

这一阶段优先做“低侵入增强”，先不大改 PPO 主体，先证明公平性上下文是否是瓶颈。

---

## 4.4 `experiments/train.py`：增加两阶段训练入口

### 目标

避免 PPO 从随机策略直接收敛到 fee-first 模式。

### 建议改动

#### A. 新增 imitation warm-start 模式

- 增加 `--pretrain-policy` 或 `--warmstart-fairness` 开关；
- 支持用 FIFO、Fair-Fee 或二者混合生成监督样本；
- 先做若干 epoch 的行为克隆，再进入 PPO。

#### B. 新增 curriculum 训练开关

训练分三阶段：

1. **阶段一：公平引导阶段**
   - 较高 `age / oldest_cover` 权重；
   - 较低风险惩罚。

2. **阶段二：风险对齐阶段**
   - 逐步提升 `gamma_r`；
   - 保持公平项仍有显式作用。

3. **阶段三：综合平衡阶段**
   - 用正式 reward 和正式验证协议选模。

#### C. 日志增强

- 记录每个阶段的平均 fee / fairness / risk / oldest coverage；
- 允许输出阶段切换点。

---

## 4.5 `experiments/evaluate.py`：支持 fairness 分解分析

### 目标

让正式评估不只输出“一个 Jain 值”，而能解释 fairness 为什么高或低。

### 建议新增输出

1. `fairness_decomposition.json`
   - Jain
   - oldest coverage
   - starvation gap
   - tail wait reduction
   - selected wait std

2. `main_episode_metrics.json` 中逐 episode 增加：
   - `oldest_coverage`
   - `starvation_gap`
   - `composite_score`

3. 新增图表输出：
   - wait distribution comparison；
   - oldest-decile service comparison；
   - episode-level fairness vs fee scatter。

---

## 4.6 `experiments/run_experiments.py`：升级实验编排与产物链路

### 目标

让新方法修改后的所有结果自动进入主流程，而不是人工拼接。

### 必改项

#### A. 支持新的 reward 配置组

新增 reward ablation 配置，例如：

- `Ours-AgeOnly`
- `Ours-Age+Risk`
- `Ours-Age+TerminalFair`
- `Ours-FullBalanced`

#### B. 支持新的 fairness 指标表格

新增：

- `table_fairness_decomp.tex`
- `table_composite_main.tex`

#### C. 支持新的 checkpoint selection policy

编排脚本应允许：

- `val_metric = composite_score`
- `val_metric = constrained_fee`
- `val_metric = hypervolume`

#### D. 支持“先超 heuristic，再冲 FIFO”的实验矩阵

编排上建议引入两类正式配置：

1. `fairness_recovery_track`
   - 目标：先把 fairness 拉回到 heuristic 以上；
2. `fifo_challenge_track`
   - 目标：在新 reward、新选模下冲击 FIFO fairness。

---

## 4.7 `experiments/config.py`：参数体系扩展

### 建议新增配置项

#### 奖励权重

- `BETA_AGE`
- `BETA_OLDEST_COVER`
- `BETA_TERMINAL_FAIR`
- `GAMMA_STARVATION`

#### 选模配置

- `VALIDATION_METRIC = composite_score | block_fee | constrained_fee`
- `VALIDATION_FAIRNESS_FLOOR`
- `VALIDATION_RISK_CEIL`

#### 公平性分析配置

- `FAIR_OLDEST_RATIO = 0.2`
- `FAIR_TAIL_QUANTILE = 0.9`

#### curriculum 配置

- `CURRICULUM_STAGE_EPISODES`
- `CURRICULUM_GAMMA_R_SCHEDULE`
- `CURRICULUM_BETA_AGE_SCHEDULE`

---

## 5. 推荐实施顺序（非常重要）

本轮改造建议严格按以下顺序推进：

### P0：先改协议与目标，不先改大模型

1. [ ] 在 `metrics.py` 新增 oldest coverage / starvation / composite score。
2. [ ] 在 `evaluate.py` 和 `run_experiments.py` 打通这些新指标的输出链路。
3. [ ] 将 checkpoint selection 从 `block_fee` 改为 `composite_score` 或约束式规则。

### P1：再改 reward

4. [ ] 在 `env.py` 中引入 `age + oldest_cover + terminal_fair + starvation` 组合奖励。
5. [ ] 保留现有 risk 项，但调成可独立开关和可独立记录。
6. [ ] 更新 reward ablation。

### P2：再补状态建模

7. [ ] 给 `block_state` 新增 backlog fairness summary。
8. [ ] 更新 `networks.py` 的 `block_dim` 及前向逻辑。

### P3：最后补训练策略增强

9. [ ] 增加 imitation warm-start。
10. [ ] 增加 curriculum 训练入口。

---

## 6. 建议的首批实验矩阵

在代码改完后，建议先跑一个“小正式版”实验矩阵，而不是直接上全量 5-seed 正式实验。

### 6.1 协议验证组

- 配置 A：旧 reward + 新 composite 选模
- 配置 B：旧 reward + 约束式选模
- 目的：先确认是不是仅换选模就能拉回部分 fairness

### 6.2 reward 验证组

- 配置 C：fee + age
- 配置 D：fee + age + risk
- 配置 E：fee + age + terminal fairness
- 配置 F：fee + age + oldest cover + terminal fairness + risk + starvation

### 6.3 状态增强组

- 配置 G：配置 F + fairness summaries
- 配置 H：配置 G + warm-start

### 6.4 验证目标

优先看三件事：

1. fairness 是否先超过 heuristic；
2. composite score 是否成为全体最优；
3. risk 是否仍显著优于 Gas Priority。

---

## 7. 论文叙事与代码产物的对齐要求

为了后续正文可以直接落地，本轮代码改造必须保证以下对齐：

### 叙事 A：综合最优

代码层需要正式输出：

- composite score；
- 其统计显著性；
- 其对应 LaTeX 表格。

### 叙事 B：fairness 改善

代码层需要正式输出：

- Jain；
- oldest coverage；
- starvation gap；
- fairness decomposition 图表。

### 叙事 C：为什么不是“靠少打包交易换 fairness”

代码层必须同时保留：

- packing ratio；
- gas utilization；
- block fee；
- risk exposure。

---

## 8. 风险提示

### 8.1 不能为了赢 FIFO 而偷偷改 metric

如果要扩展 fairness 指标，必须：

- 明确说明 Jain 仍保留；
- 新增指标是对 starvation / oldest service 的补充；
- 不是简单替换旧指标。

### 8.2 不能只改 reward，不改选模

否则训练阶段引入的新 fairness 目标会被 fee-only validation 再次冲掉。

### 8.3 不能同时大改 reward、网络、数据生成而不做分步实验

否则很难判断到底是哪一层改动真正起作用。

---

## 9. 最小可执行版本（MVP）

如果时间有限，建议先做下面这 6 项最关键改动：

1. [ ] `metrics.py` 新增 `oldest_coverage_ratio / starvation_gap / composite_score`。
2. [ ] `evaluate.py` 与 `run_experiments.py` 打通新指标落盘。
3. [ ] `checkpoint_meta.json` 选模协议改成 `composite_score`。
4. [ ] `env.py` 将 fairness reward 从 `ΔJain` 扩展为 `age + oldest_cover + terminal_fair`。
5. [ ] `env.py` 的 STOP 惩罚加入 oldest 未服务惩罚。
6. [ ] `run_experiments.py` 增加一套 fairness-oriented reward ablation。

如果这 6 项做完，哪怕暂时不改网络结构，也足以回答：

- 当前 fairness 差是不是目标函数错了；
- composite selection 能不能把“综合最优”先做出来；
- 方法是否有机会先超过 heuristic fairness。

---

## 10. 一句话执行建议

**先把“综合目标”和“公平目标”正式写进实验协议与代码接口，再去训练模型；否则后续无论结果多好，都很难稳地支撑你想要的论文叙事。**
