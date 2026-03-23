# 实验框架代码改造建议（2026-03-23）

## 结论

如果你问的是“**实验框架代码本身，下一步最值得改什么**”，我的判断是：

> **先改“可复现性与统计产物”，再改“运行控制”，最后再改“方法增强”。**

也就是说，最优先的不是再堆更多实验设置，而是先让当前框架能稳定回答这三个问题：
1. 这次结果是由哪组配置跑出来的？
2. 这组结果能否被严格复现与重新检验？
3. 主实验是否值得扩到鲁棒性和消融？

---

## P0：建议在全量正式实验前优先修改

### 1. 保存逐 episode 原始结果，而不是只保存聚合均值

当前 `evaluate.py` / `run_experiments.py` 主要保存的是聚合后的 `mean/std` 结果，而不是每个 episode 的原始指标序列。

**为什么这是第一优先级：**
- 你现在很难做真正的 episode 级配对检验；
- 后续如果想改统计口径，只剩聚合均值会很被动；
- 审稿人若质疑某项指标，需要回到原始样本时，你手上没有最关键的数据层。

**建议怎么改：**
- `evaluate_rl()` 和 `evaluate_baseline()` 保持返回 `list[dict]` 不变；
- 在 `run_single_seed()` 中把每个方法的原始 episode 结果也落盘，例如：
  - `seed_x/results/main_episode_metrics.json`
  - `seed_x/results/robustness_risk_episode_metrics.json`
- JSON 结构建议保存为：
  - `setting`
  - `method`
  - `episode_id`
  - `metrics`
  - `shared_pool_id`

### 2. 主实验与鲁棒性评估池要像消融一样落盘

当前消融实验已经把共享评估池保存到了磁盘，但主实验和鲁棒性大多还是运行时现生成、跑完即失。

**问题：**
- 难以复核“这一版论文结果到底用了哪批池”；
- 统计分析无法在后续脚本中直接重用；
- 同一 seed 的结果若要重算，只能重新生成池，虽然理论上 seed 一致可复现，但管理上不够稳。

**建议怎么改：**
- 参照消融的 `save_shared_pools()` 逻辑；
- 为主实验和每组鲁棒性都保存共享评估池；
- 命名建议：
  - `shared_pools_main_seed42.json`
  - `shared_pools_robust_risk_seed42.json`
  - `shared_pools_robust_pool_seed42.json`
  - `shared_pools_robust_fee_seed42.json`

### 3. 统计检验应改为基于 episode 配对结果，而不是 seed 均值

当前 `run_significance_tests()` 是直接对各 seed 的方法均值做 Welch t-test。

**问题：**
- 样本量太小；
- 没有用上共享评估池带来的配对结构；
- 更像“训练稳定性比较”，不像“测试集表现比较”。

**建议怎么改：**
- 新增一个函数，例如 `run_paired_significance_tests()`；
- 输入改为“同一 shared pool 下，RL 与 baseline 的逐 episode 指标序列”；
- 默认使用：
  - 配对 t 检验；
  - 或 Wilcoxon signed-rank（更稳健）；
- seed 级 mean/std 继续保留在主表中，作为稳定性统计。

### 4. 模型不存在时不要默认回退到随机策略

当前 `evaluate.py` 和 `run_experiments.py` 在 checkpoint 不存在时，会打印 warning 后继续用随机策略评估。

**问题：**
- 对正式实验很危险；
- 结果文件可能被误当成正式结果；
- 后面如果只看 JSON，不一定能一眼看出该结果来自随机策略。

**建议怎么改：**
- 默认改为 fail-fast：模型不存在时直接报错退出；
- 仅在显式加参数时允许随机策略回退，例如：
  - `--allow-random-fallback`
- 并在结果 JSON 中写入 `policy_source=trained|random_fallback`。

### 5. 给 `run_experiments.py` 增加阶段开关，避免“一跑全跑”

当前 `run_experiments.py` 只要启动单 seed，就会默认执行主实验 + 3 类鲁棒性；加 `--ablation` 后再继续跑两类消融。

**问题：**
- 不利于分阶段推进；
- 想先只跑主实验时也会把鲁棒性一起跑掉；
- 调参成本偏高。

**建议怎么改：**
- 新增布尔开关：
  - `--main-only`
  - `--with-robustness`
  - `--with-ablation`
  - 或者统一成 `--stages main robustness ablation`
- 推荐默认行为改成：**只跑主实验**。

---

## P1：建议在主实验结果出来后尽快修改

### 6. 保存配置快照与环境元信息

当前结果目录里有 checkpoint meta，但缺少一份完整的实验配置快照。

**建议补充：**
- `config_snapshot.json`，记录：
  - 代码里关键超参数；
  - CLI 参数；
  - seeds；
  - 设备；
  - checkpoint 规则；
  - 时间戳；
  - 当前 git commit hash。

这样你后面看到一个 `results_main_5seeds/`，能立刻知道它到底对应哪一版代码和配置。

### 7. 统一方法命名，不要在不同模块里混用大写/小写/别名

现在主实验、鲁棒性和 LaTeX 表格生成里，方法名存在多套风格：
- `RL (Ours)`
- `RL`
- `fifo`
- `FIFO`
- `gas`
- `GAS`
- `fair_fee`
- `FairFee`

**问题：**
- 增加 JSON 后处理复杂度；
- 很容易在表格生成、统计检验、绘图时写错 key；
- 后续补新基线时会越来越乱。

**建议怎么改：**
- 在 `config.py` 或独立 `registry.py` 中定义唯一方法注册表；
- 每个方法包含：
  - `method_id`
  - `display_name`
  - `latex_name`
- 所有模块只传 `method_id`，显示时再映射。

### 8. 训练选模规则建议从“单回合最大奖励”升级为“固定验证池最优”

当前 best checkpoint 采用单回合奖励峰值保存。

**建议怎么改：**
- 划出一个固定小验证池，例如 100 个 shared pools；
- 每隔 50 或 100 个 episode 做一次验证；
- 按验证指标保存 best checkpoint；
- 正文里就能更有底气地说正式评估采用验证最优模型。

### 9. 结果 JSON 应显式写明“这是聚合结果还是原始结果”

现在 `main_results.json` 这种名字容易让人误解成完整结果。

**建议：**
- 改成更明确的双层命名：
  - `main_episode_metrics.json`
  - `main_aggregated_metrics.json`
- 鲁棒性和消融也采用同样命名风格。

### 10. 输出目录建议自动创建运行摘要索引

建议在每个总输出目录下自动生成 `run_summary.json`，概括：
- 跑了哪些阶段；
- 每个 seed 是否成功；
- 每个结果文件路径；
- 是否生成显著性结果；
- 是否生成 LaTeX 表格。

这样排查实验是否完整会省很多时间。

---

## P2：属于框架增强项，可在投稿前择优做

### 11. 增加更强基线的可插拔接口

现在基线是在多个脚本里手工列方法名。

**建议怎么改：**
- 新建统一基线注册表；
- 每个基线注册：
  - `id`
  - `callable`
  - `display_name`
  - `enabled_by_default`
- 这样以后加 `CenterAwareGreedy` 或别的启发式时，不需要改 4 个地方。

### 12. 增加运行耗时统计

当前几乎没有记录训练耗时、评估耗时、单 seed 总耗时。

**建议：**
- 对 train / eval / robustness / ablation 分阶段计时；
- 落盘到 `timing.json`；
- 后面写论文实验设置时，也能说明计算成本。

### 13. 增加失败恢复与断点续跑能力

如果 5 seed 长实验跑到一半中断，现在恢复体验一般。

**建议：**
- 若某 seed 已有完整结果，则自动跳过；
- 若已有 checkpoint 且指定 `--resume`，则继续训练或直接评估；
- 输出目录中为每个 seed 维护状态标记文件。

### 14. 设备调度与并行策略可以更保守

当前 `ProcessPoolExecutor` + 多 seed 并行对 CPU 还好，但如果后面改到 GPU，容易出现资源竞争。

**建议：**
- CPU 场景允许多进程；
- GPU 场景默认 `workers=1`；
- 或显式加入 `--device-map` / `--max-gpu-workers`。

### 15. 解释性实验（RQ4）最好补成正式脚本，而不是靠手工分析

如果论文还保留 RQ4，代码层面应该至少补两个正式产物：
- `case_study.json`：保存典型池上的各方法排序序列与关键指标；
- `linear_probe.json`：保存线性拟合结果与对比指标。

否则建议论文把 RQ4 降级为补充分析，不要继续作为独立研究问题。

---

## 我最建议你立刻动手改的 5 个点

如果只允许改 5 件事，我建议按这个顺序：

1. **保存 episode 级原始结果**；
2. **保存主实验/鲁棒性的 shared pools**；
3. **去掉默认随机策略回退，改成 fail-fast**；
4. **给 `run_experiments.py` 增加 `main-only / stages` 开关**；
5. **补 `config_snapshot.json + run_summary.json`**。

这 5 个改动不会改变你的方法本身，但会显著提高：
- 结果可复现性；
- 统计分析灵活性；
- 运行管理可控性；
- 论文结果的可信度。

---

## 一句话判断

**你现在最该改的，不是模型结构，而是“实验产物设计”和“运行控制逻辑”。**

只要这两块先补齐，后面不管是补 paired significance、加更强基线，还是继续写论文，都会顺很多。
