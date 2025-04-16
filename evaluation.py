# evaluation.py

import math
import config

def evaluate_individual(individual, toolbox):
    """
    評估個體：
      1. 先將 GP tree 編譯成可呼叫的函數，並使用一組測試資料計算 error。
      2. 同時計算該個體的樹大小(例如: len(individual))。
      
    回傳 (error, tree_size) 兩目標，皆希望越小越好。
    """
    # 編譯個體，產生一個 callable function
    func = toolbox.compile(expr=individual)
    
    # ---------------------------
    # 取得測試資料
    # ---------------------------
    # 範例：建立一個 dummy dataset，每筆資料皆為一個 tuple，
    # tuple 中元素的數量必須與 TERMINALS 數量相符。
    dummy_data = [
        (1,  2,   4,    8,    16,  32,   64,   128, 256, 512,   1024, 2048,  4096),
      # 'PT','RT','RPT','RNO','DD','RTO','PTN','SL','WT','APTQ','NJQ','WINQ','CT'
        # 可加入更多測試資料...
    ]
    
    error = 0.0
    for data in dummy_data:
        try:
            result = func(*data)
            # ---------------------------
            # 計算誤差
            # ---------------------------
            # 請根據實際問題提供正確的期望值。
            expected = 255  # TODO: 替換為真實期望值
            error += abs(result - expected)
        except Exception as e:
            # 若在計算時發生例外，則給予一個很高的誤差值
            error += 1e8

    tree_size = len(individual)

    if config.OBJECTIVE_TYPE == "SINGLE":
        if config.SINGLE_OBJECTIVE_TYPE == "ERROR":
            return (error,)
        elif config.SINGLE_OBJECTIVE_TYPE == "TREE_SIZE":
            return (tree_size,)
        elif config.SINGLE_OBJECTIVE_TYPE == "COMBINED":
            combined_score = config.ERROR_WEIGHT * error + config.SIZE_WEIGHT * tree_size
            return (combined_score,)
    else:
        return error, tree_size  # 返回兩個目標
