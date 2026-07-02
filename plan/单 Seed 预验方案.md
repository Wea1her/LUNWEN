# Agent 执行文档：单 Seed 预验方案（two_stage vs trade_score）

## 0. 文档目的

本文件供实验执行 Agent 阅读与执行，用于在正式 5-seed 实验前进行一次**单 seed 预验**，比较两种验证选模协议：

1. `two_stage`：可行性优先 + 折中分析；
2. `trade_score`：给定目标偏好下的折中函数选模。

本次预验的目标不是产出正式论文结论，而是决定正式 5-seed 实验采用哪一种 `val_metric`。

---

## 1. 当前实验背景

论文当前主叙事为：

> 区块构建交易排序是收益、等待公平与位置敏感风险之间的多目标冲突问题。本文不追求绝对全局最优，而是在预设目标偏好或约束条件下学习一个有效的交易排序 operating point。

当前代码已加入综合折中函数：

- `trade_score`
- `risk_aware_trade_score`
- `constrained_trade_score`

因此需要验证：

- 如果采用 `two_stage` 选模，是否能保持更好的可行性与风险边界；
- 如果采用 `trade_score` 选模，是否能提高综合折中得分且不破坏风险、公平和可行性。

---

## 2. 严格限制

Agent 执行时必须遵守：

1. **不要修改模型结构**；
2. **不要修改 reward 结构**；
3. **不要新增实验类型**；
4. **不要根据单 seed 结果反复调参**；
5. **不要把单 seed 结果写成正式结论**；
6. **只比较 `two_stage` 与 `trade_score` 两种选模协议**；
7. 其他实验参数必须保持一致。

---

## 3. 实验基本配置

统一配置如下：

```text
seed = 42
training episodes = 3000
evaluation episodes = 1000
pool_size = 300
operating_mode = balanced
device = cuda:0
workers = 1
max_gpu_workers = 1
```

如果运行环境没有 GPU，可将：

```bash
--device cuda:0
```

替换为：

```bash
--device cpu
```

但需要在运行报告中记录实际设备。

---

## 4. 实验 A：two_stage 选模

### 4.1 实验目的

验证“可行性优先 + 折中分析”的模型选择方案。

### 4.2 实验叙事

如果本方案表现更稳，正式论文可写：

> 本文采用可行性优先的 two-stage 规则进行模型选择，以避免策略仅追求综合得分而违反基本公平或风险约束；同时使用 \(S_{\text{trade}}\) 作为测试阶段辅助综合评价指标。

### 4.3 执行命令

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
  --output results_single_seed_twostage_check
```

---

## 5. 实验 B：trade_score 选模

### 5.1 实验目的

验证“给定目标偏好下的综合折中函数选模”是否能取得更好的综合表现。

### 5.2 实验叙事

如果本方案表现更好，正式论文可写：

> 本文采用 \(S_{\text{trade}}\) 作为验证集选模指标，使模型选择目标与测试阶段多目标折中评价保持一致。

### 5.3 执行命令

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
  --val-metric trade_score \
  --output results_single_seed_trade_score_check
```

---

## 6. 两组实验完成后必须检查的文件

两个输出目录均需检查以下文件：

```text
outputs_manifest.json
v5_protocol_manifest.json
checkpoint_meta.json
aggregated_main.json
table_main.tex
table_tradeoff_main.tex
narrative_guard.json
baseline_best_params.json
```

### 6.1 文件完整性要求

对于每个实验目录，必须满足：

```text
outputs_manifest.json 中 report_incomplete = false
table_tradeoff_main.tex 正常生成
aggregated_main.json 存在
narrative_guard.json 存在
checkpoint_meta.json 存在
baseline_best_params.json 存在
```

如任一文件缺失，需记录为运行失败或输出不完整。

---

## 7. 核心指标提取

从 `aggregated_main.json` 或对应汇总表中提取以下指标。

### 7.1 综合折中指标

```text
trade_score_mean
risk_aware_trade_score_mean
constrained_trade_score_mean
```

### 7.2 原始主指标

```text
block_fee_mean
fairness_mean
old_tx_pack_rate_mean
risk_exposure_mean
edge10_risk_mean
gas_util_mean
packing_ratio_mean
risky_inclusion_rate_mean
```

### 7.3 可行性指标

如代码已输出，提取：

```text
feasible_rate
feasible_fee_mean
two_stage_selection_score
violation_breakdown
```

如未直接输出，则从 `narrative_guard.json` 或 constraint summary 中提取对应信息。

---

## 8. 对比表模板

Agent 需要在最终报告中填充如下表格：

| 指标 | two_stage | trade_score | 判断 |
|---|---:|---:|---|
| block_fee_mean |  |  | 收益是否保持 |
| fairness_mean |  |  | 等待公平是否下降 |
| old_tx_pack_rate_mean |  |  | 长等待交易服务是否保持 |
| risk_exposure_mean |  |  | 总体风险是否恶化 |
| edge10_risk_mean |  |  | 头尾敏感区风险是否恶化 |
| gas_util_mean |  |  | Gas 利用是否稳定 |
| packing_ratio_mean |  |  | 打包效率是否稳定 |
| risky_inclusion_rate_mean |  |  | 是否通过少打包高风险交易伪造低风险 |
| trade_score_mean |  |  | 综合折中是否提升 |
| risk_aware_trade_score_mean |  |  | 风险增强折中是否提升 |
| constrained_trade_score_mean |  |  | 约束折中是否提升 |
| feasible_rate |  |  | 可行性是否下降 |

---

## 9. 判定规则

### 9.1 优先选择 two_stage 的条件

如果出现以下任意情况，建议正式 5-seed 实验采用 `two_stage`：

```text
trade_score 方案虽然 trade_score 更高，但 risk_exposure 明显恶化；
trade_score 方案 edge10_risk 明显恶化；
trade_score 方案 feasible_rate 明显下降；
trade_score 方案 packing_ratio 明显下降；
trade_score 方案 old_tx_pack_rate 明显下降；
trade_score 方案 risky_inclusion_rate 明显下降；
trade_score 方案只是提高收益，但牺牲公平或风险。
```

建议阈值：

```text
feasible_rate 下降 > 5 个百分点：不建议用 trade_score
risk_exposure 上升 > 10%：不建议用 trade_score
edge10_risk 上升 > 10%：不建议用 trade_score
packing_ratio 下降 > 5%：不建议用 trade_score
old_tx_pack_rate 明显下降：不建议用 trade_score
risky_inclusion_rate 明显下降：需警惕风险指标被选择性不打包掩盖
```

### 9.2 可以选择 trade_score 的条件

如果满足以下条件，可考虑正式 5-seed 实验采用 `trade_score`：

```text
trade_score_mean 明显高于 two_stage；
risk_exposure 没有明显恶化；
edge10_risk 没有明显恶化；
feasible_rate 基本不下降；
block_fee 保持稳定；
fairness 和 old_tx_pack_rate 保持稳定；
packing_ratio 不明显下降；
risky_inclusion_rate 没有明显下降。
```

建议阈值：

```text
trade_score_mean 提升 >= 1%~2%
risk_exposure 上升 <= 5%
edge10_risk 上升 <= 5%
feasible_rate 下降 <= 3 个百分点
block_fee 下降 <= 3%
packing_ratio 下降 <= 3%
old_tx_pack_rate 基本不下降
risky_inclusion_rate 基本不下降
```

### 9.3 如果两者差异不明显

如果两者差异不明显，默认选择：

```text
two_stage
```

理由：

> 当前论文强调风险感知与约束可行性，two_stage 更稳，并且更适合后续毕业论文中约束强化学习方向的衔接。

---

## 10. Agent 最终报告必须包含

Agent 完成两组实验后，需要输出一份 Markdown 报告，包含以下内容：

### 10.1 实验状态

```text
实验 A 是否完成
实验 B 是否完成
是否存在缺失文件
是否存在 NaN / inf / 异常指标
```

### 10.2 核心结果表

填写第 8 节中的对比表。

### 10.3 选模协议建议

必须明确给出：

```text
建议正式 5-seed 使用 two_stage
```

或：

```text
建议正式 5-seed 使用 trade_score
```

并说明理由。

### 10.4 论文叙事建议

根据结果，给出下列之一：

#### 若选择 two_stage

```text
建议论文主叙事为：可行性优先 + 多目标折中分析。
S_trade 作为测试阶段辅助综合评价指标。
```

#### 若选择 trade_score

```text
建议论文主叙事为：给定目标偏好下的多目标折中最优。
S_trade 同时作为验证集选模指标和测试集综合评价指标。
```

#### 若结果不稳定

```text
建议继续使用 two_stage，并将本次结果描述为单 seed 预验，不进入正式结论。
```

---

## 11. 注意事项

1. 本次实验是单 seed 预验，不具备正式统计意义；
2. 不允许使用“显著优于”；
3. 不允许把单 seed 结果写成论文正式主表；
4. 不允许根据本次结果多轮修改权重；
5. 不允许把 `S_trade` 解释为客观全局最优；
6. 如果结果不理想，优先报告 operating point 差异，而不是立即修改代码；
7. 正式 5-seed 启动前，必须冻结最终 `val_metric`。

---

## 12. 推荐默认结论

在没有强证据表明 `trade_score` 更稳之前，默认建议：

```text
正式主协议：two_stage
辅助评价指标：S_trade
```

对应论文叙事：

> 本文采用可行性优先的模型选择规则，以保证策略满足基本公平与风险约束；同时定义多目标折中得分 \(S_{\text{trade}}\)，用于评价不同排序方法在预设目标偏好下的综合 operating point。

---

## 13. 完成后下一步

如果预验通过，执行正式 5-seed 实验：

```bash
PYTHONPATH=experiments python3 experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --workers 1 \
  --max-gpu-workers 1 \
  --device cuda:0 \
  --operating-mode balanced \
  --val-metric <FINAL_VAL_METRIC> \
  --output results_formal_5seed_YYYYMMDD
```

其中 `<FINAL_VAL_METRIC>` 必须替换为本次预验选定的：

```text
two_stage
```

或：

```text
trade_score
```
