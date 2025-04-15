# gp_setup.py

import operator
import random
from deap import gp, base, creator, tools
import config
from deap.gp import PrimitiveTree

# ---------------------------
# 建立 Fitness 與 Individual 類別
# ---------------------------
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # 多目標最小化
except Exception:
    pass

try:
    creator.create("Individual", list, fitness=creator.FitnessMin)  # 基於 list 的 Individual 類別
except Exception:
    pass

class MultiTreeIndividual:
    """
    自訂個體類別，包含兩棵樹：routing_tree 和 sequencing_tree。
    """
    def __init__(self, routing_tree, sequencing_tree):
        self.individual = creator.Individual([routing_tree, sequencing_tree])  # 使用 creator.Individual
        self.routing = None      # 編譯後的 routing 函數
        self.sequencing = None   # 編譯後的 sequencing 函數

    @property
    def fitness(self):
        return self.individual.fitness

    @fitness.setter
    def fitness(self, value):
        self.individual.fitness = value

    def __getitem__(self, index):
        return self.individual[index]

    def __setitem__(self, index, value):
        self.individual[index] = value

    def __len__(self):
        return len(self.individual)

    def __iter__(self):
        return iter(self.individual)


def cxMultiTree(ind1, ind2, toolbox):
    """
    自定義交叉操作子，分別對 MultiTreeIndividual 的 routing_tree 和 sequencing_tree 進行交叉。
    """
    def subtree_crossover(tree1, tree2):
        """
        對兩棵樹進行子樹交換操作。
        """
        # 隨機選擇第一棵樹的子樹
        index1 = random.randint(0, len(tree1) - 1)
        subtree1 = tree1.searchSubtree(index1)

        # 隨機選擇第二棵樹的子樹
        index2 = random.randint(0, len(tree2) - 1)
        subtree2 = tree2.searchSubtree(index2)

        # 交換子樹
        new_tree1 = tree1[:subtree1.start] + tree2[subtree2] + tree1[subtree1.stop:]
        new_tree2 = tree2[:subtree2.start] + tree1[subtree1] + tree2[subtree2.stop:]

        return new_tree1, new_tree2

    # 對 routing_tree 進行交叉
    routing_tree1 = PrimitiveTree(ind1[0])
    routing_tree2 = PrimitiveTree(ind2[0])
    new_routing_tree1, new_routing_tree2 = subtree_crossover(routing_tree1, routing_tree2)

    # 對 sequencing_tree 進行交叉
    sequencing_tree1 = PrimitiveTree(ind1[1])
    sequencing_tree2 = PrimitiveTree(ind2[1])
    new_sequencing_tree1, new_sequencing_tree2 = subtree_crossover(sequencing_tree1, sequencing_tree2)

    # 更新個體的樹
    ind1[0], ind2[0] = new_routing_tree1, new_routing_tree2
    ind1[1], ind2[1] = new_sequencing_tree1, new_sequencing_tree2

    return ind1, ind2
    
def mutMultiTree(individual, toolbox):
    """
    自定義突變操作子，分別對 MultiTreeIndividual 的 routing_tree 和 sequencing_tree 進行突變。
    """
    def subtree_mutation(tree, expr_generator):
        """
        對單棵樹進行子樹突變操作。
        """
        # 隨機選擇樹中的一個節點
        index = random.randint(0, len(tree) - 1)
        subtree = tree.searchSubtree(index)

        # 生成新的子樹
        new_subtree = expr_generator()

        # 替換舊的子樹
        mutated_tree = tree[:subtree.start] + new_subtree + tree[subtree.stop:]
        return mutated_tree

    # 對 routing_tree 進行突變
    routing_tree = PrimitiveTree(individual[0])
    mutated_routing_tree = subtree_mutation(routing_tree, toolbox.expr_routing)

    # 對 sequencing_tree 進行突變
    sequencing_tree = PrimitiveTree(individual[1])
    mutated_sequencing_tree = subtree_mutation(sequencing_tree, toolbox.expr_sequencing)

    # 更新個體的樹
    individual[0] = mutated_routing_tree
    individual[1] = mutated_sequencing_tree

    return individual,

def create_primitive_set():
    """
    建立並回傳一個 GP 的 primitive set，
    將 function 與 terminal 都加入到 set 中。
    """
    # 設定 pset 的 arity 為 TERMINALS 的數量
    pset = gp.PrimitiveSet("MAIN", arity=len(config.TERMINALS))

    # 重新命名參數名稱
    for i, term in enumerate(config.TERMINALS):
        pset.renameArguments(**{f"ARG{i}": term})

    # 加入基本運算符
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(config.div, 2, name="div")  # 確保 `div` 是正確定義的函數

    # 若需要加入隨機常數
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
    根據 primitive set 建立 toolbox，
    包含個體定義、種群初始化、編譯、評估及基因操作子。
    """
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
        return MultiTreeIndividual(routing_tree, sequencing_tree)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ---------------------------
    # 編譯：將樹轉換成可呼叫的函數
    # ---------------------------
    toolbox.register("compile", gp.compile, pset=pset)

    # ---------------------------
    # 註冊評估函數
    # ---------------------------
    from evaluation import evaluate_individual
    toolbox.register("evaluate", lambda ind: evaluate_individual(ind, toolbox))

    # ---------------------------
    # 註冊選擇、交配、突變操作子
    # ---------------------------
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", cxMultiTree, toolbox=toolbox)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", mutMultiTree, toolbox=toolbox)
    
    # 限制樹的最大節點數量，避免樹過度膨脹 (bloat)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=100))  # 限制最大節點數量為 100
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=100))  # 限制最大節點數量為 100
    
    return toolbox
