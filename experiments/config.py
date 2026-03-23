"""超参数与全局配置"""

# ========== 仿真环境 ==========
MAX_BLOCK_GAS = 30_000_000
POOL_SIZE_MIN = 50
POOL_SIZE_MAX = 500
POOL_SIZE_DEFAULT = 300
GAS_TRANSFER = 21_000
GAS_CONTRACT_MIN = 50_000
GAS_CONTRACT_MAX = 500_000
FEE_LOG_MEAN = 1.5          # 对数正态分布 μ (Gwei)
FEE_LOG_STD = 0.8           # 对数正态分布 σ
RISK_RATIO = 0.15            # 默认风险交易比例
RISK_SCORE_HIGH = (0.5, 1.0) # 风险交易评分范围
RISK_SCORE_LOW = (0.0, 0.3)  # 普通交易评分范围
ARRIVAL_RATE = 1.0           # 泊松到达率 (笔/秒)
RISK_FEE_MULTIPLIER = 2.0    # 风险交易手续费倍率
POOL_REALISM_MODE = "correlated_v1"  # 交易池生成模式: iid_baseline / correlated_v1

# ========== 奖励函数 ==========
ALPHA = 1.0   # 手续费收益权重
BETA = 0.3    # 兼容旧版: 增量 Jain 公平性权重
GAMMA_R = 0.5 # 风险惩罚权重
ETA = 0.1     # 提前停止惩罚权重
BETA_AGE = 0.25
BETA_OLDEST_COVER = 0.20
BETA_TERMINAL_FAIR = 0.35
GAMMA_STARVATION = 0.35
RISK_CENTER = 0.5
RISK_POSITION_SIGMA = 0.22
STOP_FEE_WEIGHT = 1.0
STOP_OLDEST_UNSERVED_WEIGHT = 1.2
STOP_OLDEST_WAIT_WEIGHT = 0.8
STOP_PACKING_WEIGHT = 0.8
STOP_UNUSED_GAS_WEIGHT = 0.8
STOP_LATE_HIGH_FEE_UNSERVED_WEIGHT = 0.6

# fairness gate: 达到 fairness floor 后逐步释放 fee/risk 优化自由度
FAIRNESS_GATE_TYPE = "sigmoid"  # hard / sigmoid
FAIRNESS_GATE_THRESHOLD = 0.92
FAIRNESS_GATE_SHARPNESS = 18.0
FAIRNESS_GATE_MIN = 0.35
FAIRNESS_OLDEST_COVERAGE_FLOOR = 0.95
FAIRNESS_STARVATION_CEIL = 0.05

# reward 新增项
PACKING_REWARD_WEIGHT = 0.12
LATE_FEE_RECOVERY_WEIGHT = 0.10
UNUSED_GAS_PENALTY_WEIGHT = 0.25
TERMINAL_RISK_EXPOSURE_WEIGHT = 0.20
TERMINAL_TOP10_RISK_WEIGHT = 0.15
TERMINAL_RISKY_RANK_DEV_WEIGHT = 0.10

# ========== 网络结构 ==========
TX_FEATURE_DIM = 8       # 交易原始特征维度
HIDDEN_DIM = 128         # 隐藏层维度
BLOCK_BASE_DIM = 3       # rem_gas / selected_ratio / acc_fee
BLOCK_FAIRNESS_SUMMARY_DIM = 9
BLOCK_RISK_SUMMARY_DIM = 5
BLOCK_SEQ_SUMMARY_DIM = HIDDEN_DIM
BLOCK_STATE_DIM = (
    BLOCK_BASE_DIM
    + BLOCK_FAIRNESS_SUMMARY_DIM
    + BLOCK_RISK_SUMMARY_DIM
    + BLOCK_SEQ_SUMMARY_DIM
)

# ========== PPO 训练 ==========
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
DISCOUNT = 0.99          # 折扣因子 γ
GAE_LAMBDA = 0.95        # GAE λ
PPO_CLIP = 0.2           # 截断参数 ε
PPO_EPOCHS = 4           # 每回合更新轮数
ENTROPY_COEF = 0.01      # 熵正则系数

TOTAL_EPISODES = 3000  # 训练回合数
BATCH_SIZE = 64          # mini-batch 大小
LOG_INTERVAL = 100       # 日志打印间隔

# ========== 基线 ==========
HEURISTIC_RISK_THRESHOLD = 0.5  # 启发式风险阈值

# ========== 评估 ==========
EVAL_EPISODES = 1000
RISK_POSITION_RATIO = 0.1  # 风险暴露度: 头/尾 10%
FAIR_OLDEST_RATIO = 0.2
FAIR_TAIL_QUANTILE = 0.9
COMPOSITE_W_FEE = 0.40
COMPOSITE_W_FAIRNESS = 0.25
COMPOSITE_W_RISK = 0.20
COMPOSITE_W_OLDEST_COVERAGE = 0.15
RISK_ADJUSTED_FEE_LAMBDA = 0.40
VALIDATION_EPISODES = 64
VALIDATION_INTERVAL = 50
VALIDATION_METRIC = "composite_score"
LOWER_IS_BETTER_METRICS = ("risk_exposure", "starvation_gap", "top10_risk")
VALIDATION_FAIRNESS_FLOOR = 0.0
VALIDATION_RISK_CEIL = 1.0
VALIDATION_OLDEST_COVERAGE_FLOOR = 0.0
VALIDATION_TOP10_RISK_CEIL = 1.0
VALIDATION_SEED_OFFSET = 10000
BEST_CHECKPOINT_NAME = "best_model.pt"
FINAL_CHECKPOINT_NAME = "final_model.pt"
FORMAL_EVAL_CHECKPOINT_NAME = BEST_CHECKPOINT_NAME
FORMAL_EVAL_CHECKPOINT_RULE = "best_fixed_validation_pool_metric"

# ========== 训练策略 ==========
PRETRAIN_POLICY = "none"  # none / fifo / fair_fee / mixed
PRETRAIN_EPOCHS = 0
PRETRAIN_EPISODES_PER_EPOCH = 8
CURRICULUM_ENABLED = False
CURRICULUM_STAGE_EPISODES = (0.3, 0.7, 1.0)

# ========== 新增基线参数 ==========
LINEAR_LAMBDA_F = 1.0   # Fee-Risk 线性评分: 手续费权重
LINEAR_LAMBDA_R = 0.5   # Fee-Risk 线性评分: 风险惩罚权重
FAIR_LAMBDA_F = 1.0      # Fair-Fee 双目标: 手续费权重
FAIR_LAMBDA_W = 0.5      # Fair-Fee 双目标: 等待时间权重
CENTER_AWARE_LAMBDA_F = 1.0
CENTER_AWARE_LAMBDA_W = 0.4
CENTER_AWARE_LATE_PROMO = 0.2

# ========== 多种子与鲁棒性 ==========
SEEDS = [42, 123, 456, 789, 2025]
ROBUSTNESS_RISK_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30]
ROBUSTNESS_POOL_SIZES = [100, 200, 300, 500]
ROBUSTNESS_FEE_MULTIPLIERS = [1.2, 1.5, 2.0, 3.0]


def validate_pool_size(value: int) -> int:
    """校验 CLI / 运行时传入的候选池大小。"""
    pool_size = int(value)
    if not (POOL_SIZE_MIN <= pool_size <= POOL_SIZE_MAX):
        raise ValueError(
            f"pool_size must be in [{POOL_SIZE_MIN}, {POOL_SIZE_MAX}], "
            f"got {pool_size}"
        )
    return pool_size
