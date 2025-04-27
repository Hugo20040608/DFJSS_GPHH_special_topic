# main.py

import os
import random
import numpy as np
import pandas as pd
from deap import tools, algorithms
import config
import global_vars
from gp_setup import create_primitive_set, setup_toolbox
from perato import plot_pareto_front, print_pareto_front

def output_logbook(logbook):
    # 假設 logbook 已經定義並產生，並且有以下各項統計資料：
    ngen = logbook.select("gen")          # generations，假設欄位名稱為 "gen"
    nevals = logbook.select("nevals")      # 每一代的個體評估次數

    if config.OBJECTIVE_TYPE == "SINGLE":
        fitness_max = logbook.select("max")  # 最大 fitness 值
        fitness_min = logbook.select("min")  # 最小 fitness 值  
        fitness_avg = logbook.select("avg")  # 平均 fitness 值
        fitness_std = logbook.select("std")  # 標準差 fitness 值

        # 建立 DataFrame，每一列代表一個 generation
        df = pd.DataFrame({
            "ngen": ngen,
            "nevals": nevals,
            "fitness_max": fitness_max,
            "fitness_min": fitness_min,
            "fitness_avg": fitness_avg,
            "fitness_std": fitness_std,
        })
    else:
        # 對於 fitness 章節，取得各統計量，注意每筆資料通常是 numpy array
        fitness_max = [float(x[0].item()) for x in logbook.chapters["fitness"].select("max")]
        fitness_min = [float(x[0].item()) for x in logbook.chapters["fitness"].select("min")]
        fitness_avg = [float(x[0].item()) for x in logbook.chapters["fitness"].select("avg")]
        fitness_std = [float(x[0].item()) for x in logbook.chapters["fitness"].select("std")]

        # 對於 size 章節，取得各統計量
        size_max = [float(x.item()) for x in logbook.chapters["size"].select("max")]
        size_min = [float(x.item()) for x in logbook.chapters["size"].select("min")]
        size_avg = [float(x.item()) for x in logbook.chapters["size"].select("avg")]
        size_std = [float(x.item()) for x in logbook.chapters["size"].select("std")]

        # 建立 DataFrame，每一列代表一個 generation
        df = pd.DataFrame({
            "ngen": ngen,
            "nevals": nevals,
            "fitness_max": fitness_max,
            "fitness_min": fitness_min,
            "fitness_avg": fitness_avg,
            "fitness_std": fitness_std,
            "size_max": size_max,
            "size_min": size_min,
            "size_avg": size_avg,
            "size_std": size_std,
        })

    # 固定每個數值欄位小數點後四位輸出
    df = df.round(4)
    print(df)

    if config.LOGBOOK_SAVEON is not None:
        # 確保資料夾存在，若不存在則建立
        actual_path = config.LOGBOOK_SAVEON.format(global_vars.run)
        save_dir = os.path.dirname(actual_path)
    
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # 儲存 logbook，文件名包含 run 編號
        run_specific_path = actual_path.replace(".csv", f"_run{global_vars.run:02d}.csv")
        df.to_csv(run_specific_path, index=False)
        print(f"Successfully saved logbook to {run_specific_path}!")

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 評估初始族群
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    # 對初始族群使用 NSGA-II 排序 (排序依據多目標)
    population = toolbox.select(population, len(population))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # 開始世代演化流程
    for gen in range(1, config.GENERATIONS+1):
        global_vars.gen = gen

        # 產生子代（selTournamentDCD 是 NSGA-II 常用的擴充版本，可以更好地保留多樣性）。
        offspring = tools.selTournamentDCD(population, lambda_)
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # 交配與突變
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < config.MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # 評估無效個體
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # 合併父代與子代，並使用 NSGA-II 選出 mu 個體作為下一代族群
        population = toolbox.select(population + offspring, mu)
        
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)
            if config.OBJECTIVE_TYPE == "MULTI":
                plot_pareto_front(population)
            # else:
    
    return population, logbook

def tree_sizes(ind):
    return (len(ind[0])+len(ind[1]))

def main():
    # 1. 建立 primitive set 與 toolbox
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)

    # 設定隨機種子，方便重現結果
    for run in range(len(config.RANDOMSEED)):
        global_vars.run = run
        random.seed(config.RANDOMSEED[run])
        
        # 2. 初始化種群
        population = toolbox.population(n=config.POP_SIZE)
        
        # 3. 統計資訊設定：收集 fitness 與個體大小的統計數據 (如果需要記錄多目標統計，可依需求修改)
        if config.OBJECTIVE_TYPE == "SINGLE":
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
        else:
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(tree_sizes)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", np.mean, axis=0)
            mstats.register("std", np.std, axis=0)
            mstats.register("min", np.min, axis=0)
            mstats.register("max", np.max, axis=0)
        
        # 4. 執行演化流程 (eaSimple：基本演化程序)
        if config.OBJECTIVE_TYPE == "SINGLE":
            population, logbook = algorithms.eaSimple(
                population, toolbox,
                cxpb=config.CX_PROB, mutpb=config.MUT_PROB,
                ngen=config.GENERATIONS,
                stats=stats, halloffame=None,
                verbose=config.VERBOSE
            )
        # --------------------------------------
        else:
            # (eaMuPlusLambda：(μ+λ) 演化演算法)
            # mu: 族群大小, lambda_: 從父代產生的子代數量 (可自行設定，通常 lambda_ = mu)
            population, logbook = eaMuPlusLambda(
                population, toolbox, mu=config.POP_SIZE, lambda_=config.POP_SIZE,
                cxpb=config.CX_PROB, mutpb=config.MUT_PROB,
                ngen=config.GENERATIONS,
                stats=mstats, halloffame=None,
                verbose=config.VERBOSE
            )

        # 確認 logbook 是否要輸出
        if(config.LOGBOOK_ON_TERMINAL):
            output_logbook(logbook)

        # 輸出最佳個體結果，取得非支配前沿的第一層解
        print_pareto_front(population)
        
        # TODO: 可將最佳個體存檔、繪圖、或進行後續分析

    #end for(run)
# end main

if __name__ == "__main__":
    main()
