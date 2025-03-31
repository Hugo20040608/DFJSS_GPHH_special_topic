# config.py

# Global parameters
TOTAL_RUNS = 10  # 總運行次數
RANDOM_SEED = 42
SELECTION_METHOD = "BEST"
# SELECTION_METHOD = "TOURNAMENT"
# OBJECTIVE = "MAKESPAN"
# OBJECTIVE = "MEAN-FLOWTIME"
OBJECTIVE = "MAX-FLOWTIME"

# General parameters
NUMBER_MACHINES = 10  # 機器數量
NUMBER_JOBS = 40  # 工作數量 (其中包含warm-up的工作數量)
WARM_UP = 10  # 熱身階段的工作數量
DUE_DATE_TIGHTNESS = 4  # 交期緊迫度
UTILIZATION = 0.8  # 設備利用率 (0.6~0.9)
MISSING_OPERATION = False  # 是否允許缺少操作
MEAN_PROCESSING_TIME = 25

# GP parameters
POPULATION_SIZE = 200  # 種群大小
HALL_OF_FAME_SIZE = 1  # 榮譽堂大小（保留的最佳解數量）
GENERATIONS = 1  # 進化代數
CX_PROB = 0.9  # 交叉概率
MUT_PROB = 0.1  # 突變概率

# Simulation parameters
# RANDOM_SEEDS_FOR_SIMULATION = [75]  # 隨機種子列表
# RANDOM_SEEDS_FOR_SIMULATION = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]  # 隨機種子列表
RANDOM_SEEDS_FOR_SIMULATION = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78]  # 隨機種子列表
