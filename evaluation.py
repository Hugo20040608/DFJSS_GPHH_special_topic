# evaluation.py

import config
import simulation
import something_cool
import numpy as np
from deap import gp
from gp_setup import create_primitive_set, setup_toolbox

def evaluate_individual(individual, toolbox):
    """
    評估個體：
      1. 先將 GP tree 編譯成可呼叫的函數，並使用一組測試資料計算 error。
      2. 同時計算該個體的樹大小(例如: len(individual))。
      
    回傳 (fitness, tree_size) 兩目標，皆希望越小越好。
    """
    # 編譯個體，產生一個 callable function
    routing_func = toolbox.compile(expr=gp.PrimitiveTree(individual[0]))
    sequencing_func = toolbox.compile(expr=gp.PrimitiveTree(individual[1]))
    tree_size = len(individual[0]) + len(individual[1])  # 計算樹的大小

    # 用simulation算出實際的fitness value
    # ToDo：改成使用 config 中的 SIMULATION_RANDSEED (list)
    fitness_list = []
    for seed in config.SIMULATION_RANDSEED:
        fit = simulation.simulate(routing_func, sequencing_func, random_seed=seed)
        fitness_list.append(fit)
    # 對每個目標做平均
    avg_fitness = tuple(float(np.mean([f[i] for f in fitness_list])) for i in range(len(fitness_list[0])))
    fitness = avg_fitness + (tree_size,)
    
    # 根據 config.OBJECTIVE_TYPE 回傳對應的目標值
    result_objective = ()
    for i in config.OBJECTIVE_TYPE:
        result_objective += (fitness[config.PI_mapping[i]],)
    return result_objective

def evaluate_individual_no_toolbox(individual):
    # 這裡 toolbox 可以是全域的，或每次都新建
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)
    return evaluate_individual(individual, toolbox)

def test_specific_rule():
    # 1. 建立原始集合與工具箱
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)
    
    # 2. 手動定義想要測試的規則
    # 建立 routing tree (手動或解析字符串)
    routing_str = "PT"  # 可替換為你想測試的規則
    routing_tree = gp.PrimitiveTree.from_string(routing_str, pset)
    
    # 建立 sequencing tree
    sequencing_str = "add(PT,DD)"  # 可替換為你想測試的規則
    sequencing_tree = gp.PrimitiveTree.from_string(sequencing_str, pset)
    
    # 3. 創建一個測試個體
    from gp_setup import MultiTreeIndividual
    test_individual = MultiTreeIndividual(routing_tree, sequencing_tree)
    
    # 4. 評估此個體
    # random.seed(42)  # 設定隨機種子以獲得可重複的結果
    fitness_values = evaluate_individual(test_individual, toolbox)
    
    something_cool.double_border_my_word(
        "Rules for testing individuals:",
        f"- Routing rule: {routing_str}",
        f"- Sequencing rule: {sequencing_str}",
        f"Fitness value: {fitness_values}"
    )

if __name__ == "__main__":
    test_specific_rule()
