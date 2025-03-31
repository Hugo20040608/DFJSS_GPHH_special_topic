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


# TODO：
"""
加上多目標評估的部分(makespan + treesize) ✅

config 加上隨機初始化族群(不同的randseed) ✅

輸出演化曲線圖、初始族群、最終族群 ✅ - 初始族群 Not Yet

確認錦標賽選擇是不是真的有在錦標賽選擇 
確認演化的邏輯
確認名人堂的邏輯(是不是有需要？)  -ㄟ恭喜，沒有需要

跑了酷酷的 gif ✅

加上 simulation 到新的 GP
simulation 要加上 flexible
(以上都要確認config有更新)
"""