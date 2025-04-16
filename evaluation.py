# evaluation.py

import math
import config
import simulation

def evaluate_individual(individual, toolbox):
    """
    評估個體：
      1. 先將 GP tree 編譯成可呼叫的函數，並使用一組測試資料計算 error。
      2. 同時計算該個體的樹大小(例如: len(individual))。
      
    回傳 (fitness, tree_size) 兩目標，皆希望越小越好。
    """
    # 編譯個體，產生一個 callable function
    routing_func = toolbox.compile(expr=individual[0])
    sequencing_func = toolbox.compile(expr=individual[1])

    # 用simulation算出實際的fitness value
    fitness = simulation.simulate(routing_func, sequencing_func)
    
    tree_size = len(individual)

    if config.OBJECTIVE_TYPE == "SINGLE":
        if config.SINGLE_OBJECTIVE_TYPE == "FITNESS":
            return (fitness,)
        elif config.SINGLE_OBJECTIVE_TYPE == "TREE_SIZE":
            return (tree_size,)
        elif config.SINGLE_OBJECTIVE_TYPE == "COMBINED":
            combined_score = config.ERROR_WEIGHT * fitness + config.SIZE_WEIGHT * tree_size
            return (combined_score,)
    else:
        return fitness, tree_size  # 返回兩個目標
