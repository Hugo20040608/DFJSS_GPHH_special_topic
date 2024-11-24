from deap import base, creator, tools, gp
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

# 隨機生成個體
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 使用 partial 將 pset 傳入 genHalfAndHalf
# min_, max_ 樹的最小/最大深度(depth=[min_, max_])
generate_expr = partial(gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
individual = tools.initIterate(creator.Individual, generate_expr)

print("Generated Expression:", individual)

# 編譯並執行這個表達式
compiled = gp.compile(expr=individual, pset=pset)

# 測試表達式的運算
result = compiled(3, 5)  # 傳入 3 和 5 作為參數
print("Result of the expression:", result)
