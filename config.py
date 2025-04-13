# config.py

import operator
import random

# ---------------------------
# GP 演化參數
# ---------------------------
POP_SIZE = 500          # 族群大小
GENERATIONS = 100        # 演化代數
CX_PROB = 0.8           # 交配機率
MUT_PROB = 0.2          # 突變機率
# SELECTION_METHOD = "TOURNAMENT"   # 永遠是錦標賽選擇
TOURNAMENT_SIZE = 3     # 選擇算子：錦標賽選擇中的競爭者數量
HALL_OF_FAME_SIZE = None    # 榮譽堂大小（多目標不需要）
VERBOSE = 0             # 是否顯示演化過程詳細資料（打開才會畫圖）
# RANDOMSEED = [42, 42, 41, 40]
RANDOMSEED = [40]

# ---------------------------
# Job Shop settings (工廠設定)
# ---------------------------
MACHINE_NUM = 2             # 機台數量
WORKPIECE_NUM = 10          # 工件數量
PROCESSES_RANGE = (5,5)     # 工件製程數量 [min, MAX]
FLEXIBLE_RANGE = (1,2)      # 製程可選擇機台數量 [min, MAX]
WARM_UP = 0                 # 熱場階段工作數量
UTILIZATION_RATE = 0.8      # 工廠利用率
MEAN_PROCESSING_TIME = 100  # 平均"製程"操作時長
SD_PROCESSING_TIME = 30     # "製程"操作時長之標準差
SIMULATION_RANDSEED = 42    # 只有給模擬的隨機函數
SIMULATION_END_TIME = 2000  # 最大工廠模擬時間點

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
    'PT', 'RT', 'RPT', 'RNO', 'DD', 'RTO', 
    'PTN', 'SL', 'WT', 'APTQ', 'NJQ', 'WINQ', 'CT'
]

# ---------------------------
# Output settings (輸出設定)
# ---------------------------
LOGBOOK_ON_TERMINAL = True
LOGBOOK_SAVEON = "./CSVs//logbook.csv" # None for not saving
