import multiprocessing
import numpy as np
from simulation import simulation
from statistics import mean
import operator
import random
import algorithms
from deap import base, creator, tools, gp
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import config  # 導入配置文件
import pydot # type: ignore
from collections import Counter

def div(left, right):
    if right == 0:
        return 1
    else:
        return left / right

def ifte(condition, return_if_true, return_if_not_true):
    if condition >= 0:
        argument = return_if_true
    else:
        argument = return_if_not_true
    return argument

# define the functions and terminals
pset = gp.PrimitiveSet("MAIN", 13)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(ifte, 3)

# rename the terminals
# job/operation
pset.renameArguments(ARG0='PT')
pset.renameArguments(ARG1='RT')
pset.renameArguments(ARG2='RPT')
pset.renameArguments(ARG3='RNO')
pset.renameArguments(ARG4='DD')
pset.renameArguments(ARG5='RTO')
pset.renameArguments(ARG6='PTN')
pset.renameArguments(ARG7='SL')
pset.renameArguments(ARG8='WT')
# work center
pset.renameArguments(ARG9='APTQ')
pset.renameArguments(ARG10='NJQ')
# global
pset.renameArguments(ARG11='WINQ')
pset.renameArguments(ARG12='CT')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, phenotypic=None)

# set some GP parameters
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

##----------------------
# 目前是指用這個函數來評估個體的適應度
def simpleEvalgenSeed(input):
    func = toolbox.compile(expr=input[0]) # Transform the tree expression in a callable function
    # random_seed_current_run = input[1] # 來自於 invalid_ind 中的 random_seed (random.randint(1,300))
    # 寫死20個種子
    random_seed_current_run = [69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]
    all_mean_flowtime, all_makespan, all_max_flowtime = [], [], []
    for s in random_seed_current_run:
        current_mean_flowtime, current_makespan, current_max_flowtime = simulation(number_machines=config.NUMBER_MACHINES, number_jobs=config.NUMBER_JOBS, warm_up=config.WARM_UP,
                                                                               func=func, random_seed=s, 
                                                                               due_date_tightness=config.DUE_DATE_TIGHTNESS, utilization=config.UTILIZATION, missing_operation=config.MISSING_OPERATION)
        all_mean_flowtime.append(current_mean_flowtime)
        all_makespan.append(current_makespan)
        all_max_flowtime.append(current_max_flowtime)
    if config.OBJECTIVE == "MEAN-FLOWTIME":
        return mean(all_mean_flowtime),
    if config.OBJECTIVE == "MAKESPAN":
        return mean(all_makespan),
    if config.OBJECTIVE ==  "MAX-FLOWTIME":
        return mean(all_max_flowtime),
    assert("No Objective!")
    return 

# def simpleEvalfixSeed(input):
#     func = toolbox.compile(expr=input)  # Transform the tree expression in a callable function
#     random_seed_current_gen = 41
#     current_mean_flowtime, current_mean_tardiness, current_max_tardiness = simulation(number_machines=10, number_jobs=2500, warm_up=500,
#                                                                                func=func, random_seed=random_seed_current_gen,
#                                                                                due_date_tightness=4, utilization=0.80, missing_operation=True)
#     return current_mean_flowtime,

# def fullEval(input):
#     func = toolbox.compile(expr=input)  # Transform the tree expression in a callable function
#     makespan, max_tardiness, waiting_time, mean_flowtime = [], [], [], []
#     random_seed = [4, 15, 384, 199, 260]
#     for s in random_seed:
#         current_max_tardiness, current_mean_flowtime, current_max_tardiness = \
#             simulation(number_machines=10, number_jobs=2500, warm_up=500, func=func, random_seed=s,
#                        due_date_tightness=4, utilization=0.80, missing_operation=True)
#         mean_flowtime.append(current_mean_flowtime)
#     current_mean_flowtime = np.mean(mean_flowtime)
#     return current_mean_flowtime, 

##----------------------

# initialize GP and set some parameter
toolbox.register("evaluate", simpleEvalgenSeed)
if config.SELECTION_METHOD == "BEST":
    toolbox.register("select", tools.selBest)
elif config.SELECTION_METHOD == "TOURNAMENT":
    toolbox.register("select", tools.selTournament, tournsize=3)
else:
    assert("No selection method!")
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=1)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

def analyze_terminal_usage(population):
    terminal_counts = Counter()

    for individual in population:
        # 遍歷個體的樹結構，記錄 Terminal
        for node in individual:
            if isinstance(node, gp.Terminal):  # 判斷是否是 Terminal
                readable_name = node.value
                terminal_counts[readable_name] += 1

    # 將結果打印並繪圖
    print("Terminal Usage:")
    for terminal, count in terminal_counts.items():
        print(f"{terminal}: {count}")

    # 繪製柱狀圖
    plt.bar(terminal_counts.keys(), terminal_counts.values())
    plt.xlabel("Terminal")
    plt.ylabel("Frequency")
    plt.title("Terminal Set Usage in Final Generation")
    plt.show()

def main(run):
    # Enable multiprocessing using all CPU cores
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # define population and hall of fame (size of the best kept solutions found during GP run)
    pop = toolbox.population(n=config.POPULATION_SIZE)
    hof = tools.HallOfFame(config.HALL_OF_FAME_SIZE)

    # 將第一代族群存成檔案
    path = "./initial_population"
    os.makedirs(path, exist_ok=True)
    pop_df = pd.DataFrame([[str(i)] for i in pop], columns=['Expression'])
    pop_df.to_excel(path+f"/initial_population_run{run}.xlsx")

    # define statistics for the GP run to be measured
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    pop, log = algorithms.GPHH_new(population=pop, toolbox=toolbox, cxpb=config.CX_PROB, mutpb=config.MUT_PROB, 
                                   ngen=config.GENERATIONS, rand_seed=run, stats=mstats, halloffame=hof, verbose=True)

    # 假設 `final_population` 是最後一代的個體列表
    analyze_terminal_usage(pop)
    nodes, edges, labels = gp.graph(pop[0])
    # 創建 Pydot 圖形
    graph = pydot.Dot(graph_type="graph")
    graph.set_graph_defaults(dpi=300)  # 設置解析度為 300 DPI
    # 添加節點並設置標籤
    for node in nodes:
        graph.add_node(pydot.Node(node, label=labels.get(node, node)))
    # 添加邊
    for edge in edges:
        graph.add_edge(pydot.Edge(edge[0], edge[1]))
    # 設置布局方式並保存圖形
    graph.write("./DFJSS_results/tree.dot")  # 保存為 Graphviz DOT 文件
    graph.write_pdf("./DFJSS_results/tree.pdf")  # 保存為 PDF 文件
    graph.write_png("./DFJSS_results/tree.png")  # 保存為 PNG 文件
    print("圖形已成功生成並保存為 PDF 和 PNG 格式！")

    # define the path where the results are supposed to be saved
    path = "./DFJSS_results"
    # create the new folder for each run
    try:
        os.makedirs(path, exist_ok=True)
        print("Successfully created the directory %s " % path)
    except OSError as e:
        print(f"Creation of the directory {path} failed due to {e}")


    pop_df = pd.DataFrame([[str(i), (i.fitness).values[0]] for i in pop])
    pop_df.to_excel(path+"/final_population_run{run_num}.xlsx".format(run_num=run))

    # extract statistics:
    avgFitnessValues  = log.chapters['fitness'].select("avg")
    minFitnessValues = log.chapters['fitness'].select("min")
    maxFitnessValues = log.chapters['fitness'].select("max")
    stdFitnessValues = log.chapters['fitness'].select("std")
    nb_generation = log.select("gen")
    nevals = log.select('nevals')

    # transform statistics into numpy arrays
    minFitnessValues = np.array(minFitnessValues)
    maxFitnessValues = np.array(maxFitnessValues)
    avgFitnessValues = np.array(avgFitnessValues)
    stdFitnessValues = np.array(stdFitnessValues)
    nb_generation = np.array(nb_generation)

    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(nb_generation, maxFitnessValues, label="Max Fitness", color="orange")
    plt.plot(nb_generation, avgFitnessValues, label="Average Fitness", color="blue")
    plt.plot(nb_generation, minFitnessValues, label="Min Fitness", color="green")
    plt.title("Fitness Convergence", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Fitness", fontsize=14)
    plt.legend()
    plt.grid(True)
    # plt.close()
    plt.show()

if __name__ == '__main__':
    import time
    for i in range(config.TOTAL_RUNS):  # 根據需要調整重複次數
        start = time.time()
        random.seed(i+config.RANDOM_SEED)
        main(run=i)
        end = time.time()
        print(f'Execution time simulation: {end - start}')
