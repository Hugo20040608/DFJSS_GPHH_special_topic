# evaluation.py

import math
import config
import simulation
import random
import something_cool
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

    # 用simulation算出實際的fitness value
    fitness = simulation.simulate(routing_func, sequencing_func)

    tree_size = len(individual[0]) + len(individual[1])  # 計算樹的大小

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

def test_specific_rule():
    # 1. 建立原始集合與工具箱
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)
    
    # 2. 手動定義想要測試的規則
    # 例如：routing_rule = PT + WINQ（處理時間 + 工作中心等待時間）
    # sequencing_rule = 1/PT（最短處理時間優先）
    
    # 建立 routing tree (手動或解析字符串)
    routing_str = "add(mul(mul(PT, add(add(add(add(add(add(MWT, sub(add(add(sub(sub(add(APTQ, PT), div(APTQ, NIQ)), div(add(sub(add(add(add(sub(mul(mul(NIQ, mul(add(add(mul(DD, DD), PT), div(WIQ, TIM)), add(PT, div(WIQ, NIQ)))), div(TIS, add(mul(APTQ, APTQ), div(mul(DD, DD), sub(add(add(PT, sub(mul(add(PT, add(sub(PT, div(sub(div(sub(PT, TIS), add(div(sub(mul(mul(NIQ, TIS), div(TIS, OWT)), mul(add(TIS, NIQ), add(APTQ, MWT))), add(sub(PT, TIS), APTQ)), add(add(WIQ, mul(add(mul(div(mul(NIQ, add(add(PT, sub(mul(WIQ, mul(NIQ, PT)), mul(add(add(add(sub(add(add(mul(add(add(mul(DD, DD), CT), div(DD, add(add(add(APTQ, DD), sub(mul(PT, OWT), SL)), add(add(WIQ, TIM), add(SL, NIQ))))), DD), PT), WKR), NIQ), sub(WKR, NIQ)), sub(PT, WKR)), NIQ), APTQ))), PT)), sub(NOR, NIQ)), TIM), mul(CT, CT)), mul(div(NOR, SL), mul(WKR, OWT)))), add(sub(NOR, NOR), NIQ)))), sub(PT, add(MWT, sub(PT, WKR)))), sub(NOR, NOR))), APTQ)), SL), PT)), DD), NIQ))))), mul(add(TIS, NIQ), add(APTQ, MWT))), PT), APTQ), div(DD, PT)), NIQ), TIM), sub(NOR, NOR))), APTQ), div(SL, PT)), NIQ)), PT), APTQ), add(MWT, sub(add(TIS, SL), add(APTQ, PT)))), sub(NOR, TIM)), PT)), add(PT, add(TIS, WIQ))), WIQ)"  # 可替換為你想測試的規則
    routing_tree = gp.PrimitiveTree.from_string(routing_str, pset)
    
    # 建立 sequencing tree
    sequencing_str = "sub(add(SL, MWT), add(mul(OWT, div(WKR, NIQ)), add(mul(TIM, add(APTQ, div(sub(sub(div(div(sub(OWT, WKR), WKR), SL), sub(div(div(sub(OWT, WKR), div(mul(APTQ, MWT), NIQ)), PT), div(SL, NPT))), add(mul(OWT, div(WKR, NIQ)), add(mul(TIM, add(APTQ, div(sub(sub(div(div(sub(OWT, WKR), WKR), SL), sub(div(div(sub(OWT, WKR), div(mul(APTQ, MWT), NIQ)), PT), div(sub(sub(div(TIM, SL), sub(NOR, WKR)), mul(sub(SL, DD), mul(APTQ, CT))), NPT))), div(OWT, mul(OWT, NOR))), sub(WIQ, mul(sub(SL, DD), mul(div(sub(sub(div(div(sub(OWT, WKR), WKR), SL), sub(div(div(sub(OWT, WKR), div(mul(APTQ, MWT), NIQ)), PT), div(sub(sub(div(TIM, CT), sub(NOR, WKR)), mul(sub(SL, DD), mul(APTQ, CT))), NPT))), div(OWT, mul(OWT, NOR))), sub(WIQ, mul(sub(SL, DD), mul(APTQ, CT)))), CT)))))), sub(NPT, NIQ)))), sub(sub(NPT, sub(NOR, APTQ)), mul(sub(SL, DD), APTQ))))), sub(NPT, NIQ))))"  # 可替換為你想測試的規則
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
        f"Fitness value: {fitness_values}")
    
    # 5. 可選：直接使用模擬
    routing_func = toolbox.compile(expr=test_individual[0])
    sequencing_func = toolbox.compile(expr=test_individual[1])

if __name__ == "__main__":
    test_specific_rule()