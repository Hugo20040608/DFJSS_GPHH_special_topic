from deap import base, creator, gp, tools
from operator import add, sub, mul
from functools import partial

def div(left, right):
    if right == 0:
        return 1
    else:
        return left / right

# 定義原語集合（Primitives Set）
pset = gp.PrimitiveSet("MAIN", 2)  # 2個參數的原語集合
pset.addPrimitive(add, 2)           # 加法 (兩個參數)
pset.addPrimitive(sub, 2)           # 減法 (兩個參數)
pset.addPrimitive(mul, 2)           # 乘法 (兩個參數)
pset.addPrimitive(div, 2)           # 除法 (兩個參數)
pset.addPrimitive(min, 2)           # 除法 (兩個參數)
pset.addPrimitive(max, 2)           # 除法 (兩個參數)
pset.renameArguments(ARG0='first')  # 可以改成job的其他參數 such as PT、RT....
pset.renameArguments(ARG1='second')


# 定義適應度與個體類別（單目標最小化）
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 初始化個體和群體
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定義評估函數
def evaluate(individual):
    func = toolbox.compile(expr=individual)
    #如果 func 的結果存在隨機性，多次執行並平均是更好的選擇
    return sum(func(1, 2) for _ in range(10)),  # 返回 tuple (適應度值,)

toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)

# 初始化群體並評估
population = toolbox.population(n=10)
fitnesses = map(toolbox.evaluate, population)

for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# 輸出結果
for ind in population:
    print(ind, ind.fitness.values)
