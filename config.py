# config.py

import operator
import random
# ---------------------------
# Random seed settings (隨機種子設定)
# ---------------------------
# RANDOMSEED = [42, 42, 41, 40]
RANDOMSEED = [42]
SIMULATION_RANDSEED = [40,41,42,43,44,45,46,47,48,49]    # 只有給模擬的隨機函數

# ---------------------------
# GP 演化參數
# ---------------------------
POP_SIZE = 400          # 族群大小 need to be divisible by 4
GENERATIONS = 100         # 演化代數
CX_PROB = 0.8           # 交配機率
MUT_PROB = 0.2          # 突變機率
# SELECTION_METHOD = "TOURNAMENT"   # 永遠是錦標賽選擇
TOURNAMENT_SIZE = 7     # 選擇算子：錦標賽選擇中的競爭者數量
HALL_OF_FAME_SIZE = None    # 榮譽堂大小（多目標不需要）
VERBOSE = 1             # 是否顯示演化過程詳細資料（打開才會畫圖）
OBJECTIVE_TYPE = ("MEANFLOWTIME", "MAXFLOWTIME", "TREE_SIZE") # 多目標可選擇: "MAXFLOWTIME", "MEANFLOWTIME", "MAKESPAN", "TREE_SIZE"

# for PC processing
IND_DISTANCE_THRESHOLD = 15  # 個體距離閾值 (IND)
PC_SIMILARITY_THRESHOLD = 80  # 表型特徵相似度閾值 (用percentage表示)
PC_PROCESSING_GENERATIONS_UPPERBOUND = 70
PC_EVALUATION_INTERVAL = 10000

# ---------------------------
# Job Shop settings (工廠設定)
# ---------------------------
MACHINE_NUM = 10             # 機台數量
WORKPIECE_NUM = 600         # 工件數量
PROCESSES_RANGE = (1,10)     # 工件製程數量 [min, MAX]
FLEXIBLE_RANGE = (1,10)      # 製程可選擇機台數量 [min, MAX]
WARM_UP = 100                 # 熱場階段工作數量
UTILIZATION_RATE = 0.95      # 工廠利用率
PROCESSING_TIME_UPPER = 249
PROCESSING_TIME_LOWER = 1   # "製程"操作時長範圍 [min, MAX]
MEAN_PROCESSING_TIME = (PROCESSING_TIME_UPPER+PROCESSING_TIME_LOWER)/2  # "製程"操作時長之平均值 (min)
# SD_PROCESSING_TIME = 30     # "製程"操作時長之標準差
SIMULATION_END_TIME = None  # 最大工廠模擬時間點

DUE_DATE_MULTIPLIER = 2.0   # 預期的期限為工作執行時間的幾倍

NUM_OF_CUTPOINTS = 10       # 要被記錄的工廠時刻
SNAPSHOT_RANDSEED = [40, 50, 60]      # 只有給快照的隨機函數

# ---------------------------
# Function set (基本運算子)
# ---------------------------
def div(x, y):
    """
    保護性除法，避免除以 0 的情況
    """
    try:
        return x / y
    except ZeroDivisionError:
        return 1.0

GP_FUNCTIONS = [
    operator.add,     # 加法
    operator.sub,     # 減法
    operator.mul,     # 乘法
    div,              # 保護性除法
    min,              # 取最小值
    max,              # 取最大值
]

# ---------------------------
# Terminal set (輸入變數)
# ---------------------------
# 這裡設定的 terminal 名稱，代表每個個體中可用到的變數，
# 順序將對應到輸入資料的順序。
TERMINALS = [
    'CT', 'PT', 'NPT', 'OWT', 'TIM',
    'NOR', 'WKR', 'TIS',
    'NIQ', 'WIQ', 'MWT', 'APTQ',
    'DD', 'SL'
]
"""
ZFF: Representations with Multi-tree and Cooperative Coevolution 6.3.3.1
Yi Mei: Evolving Time-Invariant Dispatching Rules in Job Shop Scheduling with Genetic Programming
參數    意義
CT      當前時間
-
PT      製程在候選機器上的加工時間
NPT     下一個製程在候選機器上的加工時間之中位數 (如果為最後一項製程，則為 0)
TIM     製程抵達機台時間
OWT     製程在候選機器上的等待時間 (CT-TIM)
-
NOR     工件剩餘製程數量
WKR     工件完成剩餘製程所需加工時間的中位數
TIS     工件抵達工廠時間
-
NIQ     機台佇列中的製程數量
WIQ     機台佇列中的製程所需的總處理時間
MWT     表示等待機台再次空閒的時間 (目前加工的製程結束時間 - CT) -> 在 routing 有用, squencing 沒有用
APTQ    機台佇列中的製程的平均加工時間 (WIQ/NIQ)
-
DD       工件距離截止有多久
SL       鬆弛時間 (假設DD有6hr, 之後的製程要花4hr, 那麼SL=2hr)
"""

# ---------------------------
# Output settings (輸出設定)
# ---------------------------
LOGBOOK_ON_SIMULATION = 0
LOGBOOK_ON_TERMINAL = True
LOGBOOK_SAVEON = "./CSVs/" # None for not saving
PLOT_PARETO_X_SCALE = (100, 1500)
PLOT_PARETO_Y_SCALE = (0, 30)

# ---------------------------
# 列舉所有的績效指標 別改
# ---------------------------
PI_mapping = {
    "MEANFLOWTIME": 0 ,
    "MAXFLOWTIME": 1 ,
    "MAKESPAN": 2,
    "TREE_SIZE": 3
}