import multiprocessing
import numpy as np
from simulation import simulation
import operator
import random
import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import config  # 導入配置文件

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
    random_seed_current_run = input[1] # 來自於 invalid_ind 中的 random_seed (random.randint(1,300))
    current_mean_flowtime, current_mean_tardiness, current_max_tardiness = simulation(number_machines=config.NUMBER_MACHINES, number_jobs=config.NUMBER_JOBS, warm_up=config.WARM_UP,
                                                                               func=func, random_seed=random_seed_current_run, 
                                                                               due_date_tightness=config.DUE_DATE_TIGHTNESS, utilization=config.UTILIZATION, missing_operation=config.MISSING_OPERATION)
    return current_mean_flowtime,

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

def main(run):
    # Enable multiprocessing using all CPU cores
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # define population and hall of fame (size of the best kept solutions found during GP run)
    pop = toolbox.population(n=config.POPULATION_SIZE)
    hof = tools.HallOfFame(config.HALL_OF_FAME_SIZE)

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
    for i in range(1, 2):  # 根據需要調整重複次數
        start = time.time()
        random.seed(i+config.RANDOM_SEED)
        main(run=i)
        end = time.time()
        print(f'Execution time simulation: {end - start}')
