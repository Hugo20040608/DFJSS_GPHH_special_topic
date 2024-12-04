# config.py

# Global parameters
RANDOM_SEED = 42
# SELECTION_METHOD = "BEST"
SELECTION_METHOD = "TOURNAMENT"

# General parameters
NUMBER_MACHINES = 5  # 機器數量
NUMBER_JOBS = 100  # 工作數量
WARM_UP = 10  # 熱身階段的工作數量
DUE_DATE_TIGHTNESS = 4  # 交期緊迫度
UTILIZATION = 0.80  # 設備利用率
MISSING_OPERATION = True  # 是否允許缺少操作

# GP parameters
POPULATION_SIZE = 200  # 種群大小
HALL_OF_FAME_SIZE = 1  # 榮譽堂大小（保留的最佳解數量）
GENERATIONS = 20  # 進化代數
CX_PROB = 0.8  # 交叉概率
MUT_PROB = 0.05  # 突變概率

# Simulation parameters
RANDOM_SEEDS_FOR_SIMULATION = [int(i) for i in range(20)]  # 隨機種子列表