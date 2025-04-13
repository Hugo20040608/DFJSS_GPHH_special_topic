# gp_setup.py

import operator
import random
from deap import gp, base, creator, tools
import config

class MultiTreeIndividual(creator.Individual):
    """
    自訂個體類別，包含兩棵樹：routing_tree 和 sequencing_tree。
    """
    def __init__(self, routing_tree, sequencing_tree):
        super().__init__([routing_tree, sequencing_tree])
        self.routing = None      # 編譯後的 routing 函數
        self.sequencing = None   # 編譯後的 sequencing 函數


def create_primitive_set():
    """
    建立並回傳一個 GP 的 primitive set，
    將 function 與 terminal 都加入到 set 中。
    """
    # 這裡設定 pset 的 arity 為 terminals 數量，代表該 GP 模型輸入的變數數量
    pset = gp.PrimitiveSet("MAIN", arity=len(config.TERMINALS))
    
    # 重新命名預設參數名稱 (ARG0, ARG1, …) 成實際變數名稱
    for i, term in enumerate(config.TERMINALS):
        pset.renameArguments(**{f"ARG{i}": term})
    
    # 加入 function 集合，預設 arity 固定為2 (二元運算)，請依實際需求調整
    for func in config.GP_FUNCTIONS:
        pset.addPrimitive(func, 2, name=func.__name__)

    
    # 若需要加入隨機常數 (ephemeral constant)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-10, 10))
    
    return pset

def setup_toolbox(pset):
    """
    根據 primitive set 建立 toolbox，
    包含個體定義、種群初始化、編譯、評估及基因操作子。
    """
    # # ---------------------------
    # # 建立 Fitness 與 Individual 類別
    # # ---------------------------
    # # 建立多目標 Fitness，兩個目標皆最小化
    # try:
    #     creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    # except Exception as e:
    #     # 若已經建立過就忽略
    #     pass
    # try:
    #     creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, phenotypic=None)
    # except Exception as e:
    #     pass
    
    # toolbox = base.Toolbox()
    
    # # ---------------------------
    # # 個體與種群初始化
    # # ---------------------------
    # toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # # ---------------------------
    # # 編譯：將樹狀結構轉換成可呼叫的函數
    # # ---------------------------
    # toolbox.register("compile", gp.compile, pset=pset)
    
    """
    根據 primitive set 建立 toolbox，並支援 MultiTreeIndividual。
    """
    # ---------------------------
    # 建立 Fitness 與 MultiTreeIndividual 類別
    # ---------------------------
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    except Exception:
        pass
    try:
        creator.create("MultiTreeIndividual", MultiTreeIndividual, fitness=creator.FitnessMin)
    except Exception:
        pass

    toolbox = base.Toolbox()

    # ---------------------------
    # 定義 routing 和 sequencing 的樹
    # ---------------------------
    toolbox.register("expr_routing", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("expr_sequencing", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

    # ---------------------------
    # 初始化 MultiTreeIndividual
    # ---------------------------
    def init_individual():
        routing_tree = toolbox.expr_routing()
        sequencing_tree = toolbox.expr_sequencing()
        return creator.MultiTreeIndividual(routing_tree, sequencing_tree)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ---------------------------
    # 編譯：將樹轉換成可呼叫的函數
    # ---------------------------
    toolbox.register("compile", gp.compile, pset=pset)
    
    # ---------------------------
    # 註冊評估函數
    # ---------------------------
    # 這裡採用 lambda 將 toolbox 參數傳入評估函數
    from evaluation import evaluate_individual
    toolbox.register("evaluate", lambda ind: evaluate_individual(ind, toolbox))
    
    # ---------------------------
    # 註冊選擇、交配、突變操作子
    # ---------------------------
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # 限制樹的最大深度，避免樹過度膨脹 (bloat)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    
    return toolbox
