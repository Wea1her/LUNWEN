# 实验预验与正式运行命令（2026-03-23）

## 目标

给出当前第四版实验框架在“先小规模预验、再正式运行”思路下的推荐命令，尽量减少无效长跑与中途返工。

---

## 一、运行前约定

默认约定如下：

- 当前工作目录为仓库根目录；
- 使用 `python` 直接运行 `experiments/run_experiments.py`；
- 当前脚本默认 `workers=1`，即串行运行；若要恢复并行，需显式传 `--workers`；
- 若环境尚未安装依赖，先执行：

```bash
pip install -r experiments/requirements.txt
```

---

## 二、推荐执行顺序

建议严格按以下顺序执行：

1. 主实验预验（1 seed，低 episodes）；
2. 主实验 + 鲁棒性预验；
3. 主实验 + 鲁棒性 + 消融全链路预验；
4. 5 seed 正式主实验；
5. 5 seed 正式主实验 + 鲁棒性；
6. 5 seed 正式消融；
7. 若中断，使用 `--resume` 继续。

---

## 三、预验命令

### 3.1 最小主实验预验

用途：

- 检查训练、评估、聚合、主表生成是否正常；
- 检查 `aggregated_main.json`、`table2_content.tex`、`paired_significance_tests.json` 是否生成。

```bash
python experiments/run_experiments.py \
  --stages main \
  --seeds 42 \
  --episodes 200 \
  --eval-episodes 64 \
  --pool-size 300 \
  --output results_dryrun_main
```

### 3.2 主实验 + 鲁棒性预验

用途：

- 检查三类鲁棒性 JSON 与 LaTeX 表格是否能正常输出；
- 检查共享评估池、聚合与 timing 输出是否正常。

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 \
  --episodes 200 \
  --eval-episodes 64 \
  --pool-size 300 \
  --output results_dryrun_robust
```

### 3.3 全链路预验（含消融）

用途：

- 检查奖励消融、结构消融是否能稳定结束；
- 检查 `ablation_reward.json`、`ablation_struct.json` 与对应表格是否生成。

```bash
python experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 \
  --episodes 150 \
  --eval-episodes 32 \
  --pool-size 300 \
  --workers 1 \
  --output results_dryrun_full
```

说明：

- 全链路预验建议显式使用 `--workers 1`，优先保证稳定性；
- 若只是检查主实验链路，不必急着把消融一起跑。

---

## 四、正式实验命令

### 4.1 正式主实验（推荐第一步先跑这个）

用途：

- 先获得主表、配对显著性和案例分析产物；
- 优先判断 RL 是否在默认场景下取得稳定综合优势。

```bash
python experiments/run_experiments.py \
  --stages main \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_main \
  --resume
```

### 4.2 正式主实验 + 鲁棒性

用途：

- 生成主实验与三类鲁棒性正式结果；
- 用于论文第 5 章主表与鲁棒性表回填。

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_main_robust \
  --resume
```

### 4.3 正式全链路（主实验 + 鲁棒性 + 消融）

用途：

- 一次性跑完论文当前 V4 所需主要实验；
- 适合确认主流程已稳定、机器资源允许长时间占用后使用。

```bash
python experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_full \
  --resume
```

---

## 五、可选命令

### 5.1 启用更强启发式基线

当需要提高对照强度时，可加入 `Center-Aware Greedy`：

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --enable-center-aware-baseline \
  --output results_formal_main_robust_strong \
  --resume
```

### 5.2 仅评估已有模型（跳过训练）

若各 seed 的 checkpoint 已存在，只想重做评估与聚合：

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 123 456 789 2025 \
  --skip-train \
  --pool-size 300 \
  --output results_formal_main_robust \
  --resume
```

### 5.3 跳过已有完整输出

若某些 seed 已完全跑完，可用：

```bash
python experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_full \
  --skip-existing
```

---

## 六、GPU 场景建议命令

### 6.1 单卡保守跑法

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cuda:0 \
  --max-gpu-workers 1 \
  --output results_formal_gpu \
  --resume
```

### 6.2 多卡映射跑法

示例：把不同 seed 映射到不同 GPU。

```bash
python experiments/run_experiments.py \
  --stages main robustness \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --device cuda \
  --device-map 42:cuda:0,123:cuda:1,456:cuda:0,789:cuda:1,2025:cuda:0 \
  --max-gpu-workers 2 \
  --output results_formal_multi_gpu \
  --resume
```

---

## 七、建议优先采用的实际方案

如果当前目标是“尽快得到论文可回填结果”，推荐按以下三条执行：

### 第一步：小规模全链路预验

```bash
python experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 \
  --episodes 150 \
  --eval-episodes 32 \
  --pool-size 300 \
  --workers 1 \
  --output results_dryrun_full
```

### 第二步：正式主实验

```bash
python experiments/run_experiments.py \
  --stages main \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_main \
  --resume
```

### 第三步：正式主实验 + 鲁棒性 + 消融

```bash
python experiments/run_experiments.py \
  --stages main robustness ablation \
  --seeds 42 123 456 789 2025 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --output results_formal_full \
  --resume
```

---

## 八、结果检查清单

每轮运行后建议至少检查以下文件是否存在：

- `run_summary.json`
- `timing.json`
- `aggregated_main.json`
- `table2_content.tex`
- `paired_significance_tests.json`
- `case_study.json`
- `behavior_probe.json`

若运行阶段包含 `robustness`，再检查：

- `aggregated_robustness_risk.json`
- `aggregated_robustness_pool.json`
- `aggregated_robustness_fee.json`
- `table3_risk_content.tex`
- `table3_pool_content.tex`
- `table3_fee_content.tex`

若运行阶段包含 `ablation`，再检查：

- `ablation_reward.json`
- `ablation_struct.json`
- `table_ablation_reward.tex`
- `table_ablation_struct.tex`

---

## 九、当前建议

当前阶段不建议继续大改实验代码，推荐：

1. 先按本文件执行 dry-run；
2. dry-run 无异常后直接启动正式实验；
3. 结果生成后立即回填论文第 5 章，并同步修正文中与最新代码实现不一致的描述。

---

## 十、2026-03-25 最小验证方案（已确认）

> 目标：将总耗时压到约 24 小时内，优先保证“可支撑论文核心叙事”的最小证据链。  
> 原则：主结论不减配，鲁棒性和消融降统计强度。

### 10.1 配置裁剪

- `main`：3 seeds（`42 123 789`），`episodes=3000`，`eval-episodes=1000`
- `robustness`：2 seeds（`42 123`），`eval-episodes=500`
- `ablation`：2 seeds（`42 123`），`episodes=1000`，`eval-episodes=300`
- GPU 并行建议：`workers=3`，`max-gpu-workers=3`

### 10.2 执行顺序与命令

#### 步骤 A：先跑主实验（3 seeds）

```bash
python experiments/run_experiments.py \
  --stages main \
  --seeds 42 123 789 \
  --episodes 3000 \
  --eval-episodes 1000 \
  --pool-size 300 \
  --workers 3 \
  --max-gpu-workers 3 \
  --device cuda:0 \
  --output results_formal_min_fast \
  --resume
```

#### 步骤 B：再跑鲁棒性（2 seeds，复用主实验 checkpoint）

```bash
python experiments/run_experiments.py \
  --stages robustness \
  --seeds 42 123 \
  --skip-train \
  --episodes 3000 \
  --eval-episodes 500 \
  --pool-size 300 \
  --workers 2 \
  --max-gpu-workers 2 \
  --device cuda:0 \
  --output results_formal_min_fast \
  --resume
```

#### 步骤 C：最后跑消融（2 seeds，降训练/评估规模）

```bash
python experiments/run_experiments.py \
  --stages ablation \
  --seeds 42 123 \
  --episodes 1000 \
  --eval-episodes 300 \
  --pool-size 300 \
  --workers 3 \
  --max-gpu-workers 3 \
  --device cuda:0 \
  --output results_formal_min_fast \
  --resume
```

### 10.3 RQ4 最小证据口径

- 保留：`case_study.json`、`behavior_probe.json`、`pareto_episode_analysis.json`
- 不新增额外长跑：线性拟合检验可先不做，放后续补强

### 10.4 风险提示

- 该方案可支撑方向性结论，但鲁棒性/消融属于“快速验证强度”；
- 若用于最终定稿，建议后续补齐到正式强度（至少提升鲁棒性和消融 seed 数）。
