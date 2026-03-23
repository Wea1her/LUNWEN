# 实验框架修改清单（2026-03-23）

## 目标

将当前实验框架从“已具备完整主流程”进一步提升到“统计更可信、运行更稳、产物更利于论文复核”的状态。

---

## P0：建议优先修改

### 1. 将显著性检验升级为 episode 级配对检验

- [ ] 新增 `run_paired_significance_tests()`。
- [ ] 输入改为 `main_episode_metrics.json`，按 `shared_pool_id` 对齐 RL 与 baseline。
- [ ] 默认支持 paired t-test。
- [ ] 补充 Wilcoxon signed-rank test 作为稳健版本。
- [ ] 输出 `paired_significance_tests.json`。
- [ ] 输出 `table_significance_paired.tex`。

**原因：** 当前逐 episode 数据已经落盘，但正式显著性检验仍然是对 seed 级均值做 Welch t-test，没有充分利用共享评估池的配对结构。

### 2. 将 best checkpoint 改为固定验证池选模

- [ ] 预生成一批固定 validation pools。
- [ ] 每隔若干 episode 执行一次验证。
- [ ] 按验证指标保存 best checkpoint。
- [ ] 在 `checkpoint_meta.json` 中明确记录选模规则。

**原因：** 当前 best model 仍按单回合训练奖励峰值保存，容易受到训练波动影响；消融实验也沿用了相同逻辑。

### 3. 为多 seed 主流程补失败隔离与断点恢复

- [ ] 在 `future.result()` 外层增加异常捕获。
- [ ] 单个 seed 失败时不影响其它 seed 继续完成。
- [ ] 在 `run_summary.json` 中记录 `success / failed / skipped_existing / resumed` 等状态。
- [ ] 增加 `--resume`。
- [ ] 增加 `--skip-existing`。

**原因：** 当前任一 seed 失败都可能中断整批实验，长时间正式实验的容错性仍不足。

### 4. 收紧训练奖励与论文评价指标之间的口径差异

- [ ] 评估当前风险奖励项与 `risk_exposure` 指标的对应关系。
- [ ] 若暂不修改 reward，则额外输出训练同口径代理指标。
- [ ] 在论文与实验说明中明确区分训练代理目标与最终汇报指标。

**原因：** 环境中的风险位置惩罚与论文评估中的 `risk_exposure` 并非完全同构，可能影响结果解释的一致性。

---

## P1：建议尽快补充

### 5. 让并行策略按设备自适应

- [ ] `cpu` 场景默认允许多进程。
- [ ] `cuda` 场景默认将 `workers` 收缩到 1。
- [ ] 可选增加 `--max-gpu-workers`。
- [ ] 可选增加 `--device-map`。

**原因：** 当前 `workers` 默认固定为 5，没有根据设备类型自动调整，在 GPU 场景下存在资源竞争风险。

### 6. 增加正式运行耗时统计

- [ ] 记录 train 阶段耗时。
- [ ] 记录 eval 阶段耗时。
- [ ] 记录 robustness 阶段耗时。
- [ ] 记录 ablation 阶段耗时。
- [ ] 汇总输出 `timing.json`。

**原因：** 当前已有配置快照与运行摘要，但缺少正式的时间统计产物，不利于复现和论文中的计算成本说明。

### 7. 统一方法命名与基线注册方式

- [ ] 新建方法/基线 registry。
- [ ] 为每个方法统一维护 `method_id`。
- [ ] 为每个方法统一维护 `display_name`。
- [ ] 为每个方法统一维护 `latex_name`。
- [ ] 所有模块只传 `method_id`，显示层再做映射。

**原因：** 当前方法命名在不同模块中存在多套风格，后续增加新基线时维护成本会持续上升。

---

## P2：论文说服力增强项

### 8. 提升交易池生成的真实性

- [ ] 让 `fee / risk / arrival / hist_delay` 之间具有更合理的统计相关性。
- [ ] 不再仅靠统一费率倍率区分风险交易。
- [ ] 可选补一个“半真实分布/真实数据拟合”版本。

**原因：** 当前交易池生成仍以参数化随机仿真为主，适合验证机制有效性，但外部效度仍有提升空间。

### 9. 将 RQ4 解释性分析做成正式输出产物

- [ ] 新增 `case_study.json`。
- [ ] 新增 `behavior_probe.json`。
- [ ] 输出典型池排序案例与方法间排序差异。
- [ ] 输出关键行为指标对照。

**原因：** 指标层已具备解释性分析基础，但主编排流程中仍缺少对应的正式产物。

### 10. 增加一个更强但仍可解释的启发式基线

- [ ] 新增 `CenterAwareGreedy` 或类似基线。
- [ ] 显式考虑收益、风险头尾惩罚与 late-high-fee promotion。
- [ ] 通过统一 registry 接入现有评估流程。

**原因：** 当前基线已足够支撑第一版实验对照，但补一个更强启发式会进一步提高说服力。

---

## 建议实施顺序

1. [ ] paired significance
2. [ ] validation-pool checkpoint selection
3. [ ] seed failure isolation + resume
4. [ ] GPU-aware workers + timing
5. [ ] 方法/基线 registry 化
6. [ ] 更真实的数据生成
7. [ ] RQ4 正式产物
8. [ ] 更强启发式基线

---

## 最小执行版 TODO

如果只保留最核心的工作项，建议优先执行以下 6 条：

- [ ] 显著性检验改成 episode 配对检验。
- [ ] best checkpoint 改成固定验证池选模。
- [ ] 主流程补失败隔离、resume 与 skip-existing。
- [ ] 并行调度改成按设备自适应。
- [ ] 输出补 `timing.json`。
- [ ] 方法与基线改成统一 registry。
