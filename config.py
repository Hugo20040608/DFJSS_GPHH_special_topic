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
SIMULATION_END_TIME = 1000  # 最大工廠模擬時間點

DUE_DATE_MULTIPLIER = 2.0   # 預期的期限為工作執行時間的幾倍

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
LOGBOOK_ON_TERMINAL = True
LOGBOOK_SAVEON = "./CSVs//logbook.csv" # None for not saving
