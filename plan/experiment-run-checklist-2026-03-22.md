# 实验启动清单（2026-03-22）

## 结论

**可以开始跑实验，但不建议一上来就全量跑完整矩阵。**

更稳妥的顺序是：
1. 先做单 seed 轻量主流程确认；
2. 再做单 seed 正式主实验；
3. 主实验稳定后再扩到 5 seeds；
4. 最后再跑三组鲁棒性与两类消融。

---

## 为什么现在可以开始跑

当前代码已经具备以下基础：
- 训练入口、评估入口和一键编排入口都可调用；
- 主实验、鲁棒性、消融的代码路径已经打通；
- 正式评估统一使用 `best_model.pt`，结果聚合与 LaTeX 表格生成链路已经存在。

因此，从“工程可启动”的角度看，现在已经可以进入正式实验阶段。

---

## 为什么不建议直接全量开跑

虽然可以开始，但还不建议直接执行“5 seeds + 三类鲁棒性 + 两类消融 + 显著性 + 全表格”的满配实验，原因有三点：

1. **先确认主场景结果是否有论文价值**  
   如果默认主场景（`N=300, p_risk=15%, kappa=2.0`）下本文方法没有形成足够清晰的综合优势，那么后面的鲁棒性和消融就没有必要立即全量展开。

2. **先确认训练波动和选模是否稳定**  
   当前 best checkpoint 仍按单回合奖励峰值选取，建议先看 1 个 seed 的训练曲线和主实验表现，再决定是否直接扩到 5 seeds。

3. **先确认资源与耗时**  
   `run_experiments.py` 默认会串起多 seed、鲁棒性与可选消融。如果不先做小规模试跑，很容易一口气跑很久后才发现设置或结果不理想。

---

## 推荐执行顺序

### 阶段 0：环境与脚本确认（半天内完成）

目标：确认主流程不报错、目录输出正常。

建议命令：

```bash
python experiments/train.py --episodes 20 --pool-size 100 --seed 42 --output results_smoke/seed42/checkpoints --device cpu
python experiments/evaluate.py --model results_smoke/seed42/checkpoints/best_model.pt --episodes 20 --pool-size 100 --seed 42 --output results_smoke/seed42/results --device cpu
```

通过标准：
- 能生成 checkpoint；
- 能输出主实验结果；
- 训练日志和评估 JSON 正常落盘。

### 阶段 1：单 seed 正式主实验

目标：先判断“这篇论文值不值得继续全量跑”。

建议命令：

```bash
python experiments/run_experiments.py \
  --seeds 42 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cpu \
  --output results_main_seed42
```

重点检查：
- RL 是否在收益、公平、风险控制上形成可写的折中优势；
- 与 Gas Priority 的收益差距是否可接受；
- 与启发式方法相比，风险暴露是否稳定更低；
- 训练曲线是否存在明显塌陷或极端波动。

### 阶段 2：5 seed 主实验

目标：拿到论文主表所需的第一版正式结果。

建议命令：

```bash
python experiments/run_experiments.py \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cpu \
  --output results_main_5seeds
```

这一阶段产物应优先用于：
- 回填主实验表；
- 判断是否值得继续保留当前奖励权重；
- 检查显著性结论是否至少在方向上成立。

### 阶段 3：三组鲁棒性

只有在阶段 2 结果可写时再继续。

当前脚本已经内置：
- 风险比例鲁棒性；
- 候选池规模鲁棒性；
- 风险手续费倍率鲁棒性。

建议：先保留默认脚本逻辑，不要一开始再扩更多 setting。

### 阶段 4：两类消融

只有在“主实验有效”已经成立时再跑。

建议命令：

```bash
python experiments/run_experiments.py \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cpu \
  --output results_full_ablation \
  --ablation
```

---

## 跑实验前必须先确认的 5 件事

1. **固定默认配置**  
   不要边跑边改 `pool_size`、奖励权重、基线集合和 checkpoint 规则。

2. **固定论文口径**  
   在正文中明确：正式评估使用 `best_model.pt`；共享评估池是在“同一 setting 内部共享”。

3. **先看主实验，再看鲁棒性**  
   主场景没有说服力时，不要急着跑完整矩阵。

4. **保留原始 JSON 结果**  
   不要只保留 LaTeX 表格，后续改统计口径时会需要原始结果文件。

5. **记录每次运行配置**  
   建议每个输出目录都带 seed、episodes、pool_size 等信息，避免后面混淆。

---

## 我对你当前阶段的直接建议

如果你现在就要推进，**可以马上开始跑，但应该先跑“单 seed 正式主实验”，而不是直接全量总实验**。

最推荐你立刻执行的是这一步：

```bash
python experiments/run_experiments.py \
  --seeds 42 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cpu \
  --output results_main_seed42
```

如果这个结果看起来：
- 收益没有崩；
- 公平性有提升；
- 风险暴露明显下降；
- 训练曲线稳定；

那就说明这条线值得继续扩到 5 seeds 和后续鲁棒性。

否则，应先回调奖励权重、基线配置或选模规则，再进入全量实验。
