# main.py

import os
import random
import numpy as np
import pandas as pd
from deap import tools, algorithms, gp
import config
import global_vars
from gp_setup import create_primitive_set, setup_toolbox
from perato import plot_pareto_front, print_pareto_front
from something_cool import double_border_my_word

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
        # 對於 fitness 章節，取得各統計量，注意每筆資料通常是標量
        fitness1_max = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[0]}"].select("max")
        fitness1_min = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[0]}"].select("min")
        fitness1_avg = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[0]}"].select("avg")
        fitness1_std = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[0]}"].select("std")

        fitness2_max = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[1]}"].select("max")
        fitness2_min = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[1]}"].select("min")
        fitness2_avg = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[1]}"].select("avg")
        fitness2_std = logbook.chapters[f"{config.MULTI_OBJECTIVE_TYPE[1]}"].select("std")

        # 建立 DataFrame，每一列代表一個 generation
        df = pd.DataFrame({
            "ngen": ngen,
            "nevals": nevals,
            f"{config.MULTI_OBJECTIVE_TYPE[0]}_max": fitness1_max,
            f"{config.MULTI_OBJECTIVE_TYPE[0]}_min": fitness1_min,
            f"{config.MULTI_OBJECTIVE_TYPE[0]}_avg": fitness1_avg,
            f"{config.MULTI_OBJECTIVE_TYPE[0]}_std": fitness1_std,
            f"{config.MULTI_OBJECTIVE_TYPE[1]}_max": fitness2_max,
            f"{config.MULTI_OBJECTIVE_TYPE[1]}_min": fitness2_min,
            f"{config.MULTI_OBJECTIVE_TYPE[1]}_avg": fitness2_avg,
            f"{config.MULTI_OBJECTIVE_TYPE[1]}_std": fitness2_std,
        })

    # 固定每個數值欄位小數點後四位輸出
    df = df.round(4)
    print(df)

    if config.LOGBOOK_SAVEON is not None:
        # 確保資料夾存在，若不存在則建立
        save_dir = os.path.dirname(config.LOGBOOK_SAVEON)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        # 儲存 logbook
        df.to_csv(config.LOGBOOK_SAVEON, index=False)
        print(f"Successfully saved logbook to {config.LOGBOOK_SAVEON}!")

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 用於存儲每一代的樹和 fitness 值
    generation_data = []

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
    global_vars.gen = 0
    plot_pareto_front(population, objective_labels=config.MULTI_OBJECTIVE_TYPE, title="Initial Pareto Front")  # 繪製初始族群的 Pareto 前沿

    # 紀錄初始族群的樹和 fitness 值
    generation_data.append({
        "generation": 0,
        "individuals": [
            {
                "routing": str(gp.PrimitiveTree(ind[0])),
                "sequencing": str(gp.PrimitiveTree(ind[1])),
                "fitness": ind.fitness.values
            }
            for ind in population
        ]
    })

    # 開始世代演化流程
    for gen in range(1, ngen+1):
        global_vars.gen = gen

        # 產生子代（selTournamentDCD 是 NSGA-II 常用的擴充版本，可以更好地保留多樣性）。
        offspring = tools.selTournamentDCD(population, lambda_)
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # 交配與突變
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
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

        # 紀錄當前族群的樹和 fitness 值
        generation_data.append({
            "generation": gen,
            "individuals": [
                {
                    "routing": str(gp.PrimitiveTree(ind[0])),
                    "sequencing": str(gp.PrimitiveTree(ind[1])),
                    "fitness": ind.fitness.values
                }
                for ind in population
            ]
        })

        if verbose:
            print(logbook.stream)
            if config.OBJECTIVE_TYPE == "MULTI":
                    plot_pareto_front(population, objective_labels=config.MULTI_OBJECTIVE_TYPE)  # 繪製初始族群的 Pareto 前沿
            # else:
    
    return population, logbook, generation_data

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
            stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[config.PI[config.MULTI_OBJECTIVE_TYPE[0]]])
            stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[config.PI[config.MULTI_OBJECTIVE_TYPE[1]]])
            # 修正多目標統計的初始化
            mstats = tools.MultiStatistics(**{
                f"{config.MULTI_OBJECTIVE_TYPE[0]}": stats_fit1,
                f"{config.MULTI_OBJECTIVE_TYPE[1]}": stats_fit2
            })
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
        elif config.OBJECTIVE_TYPE == "MULTI":
            # (eaMuPlusLambda：(μ+λ) 演化演算法)
            # mu: 族群大小, lambda_: 從父代產生的子代數量 (可自行設定，通常 lambda_ = mu)
            population, logbook, generation_data = eaMuPlusLambda(
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
        
        # 儲存 generation_data 到檔案
        file_path = os.path.join(".", "Raw_Data")
        if not os.path.exists(file_path):
            # 如果路徑不存在，則創建它
            os.makedirs(file_path)
            print(f"創建資料夾: {file_path}")
        file_name = f"generation_data_run{run}.json"
        full_path = os.path.join(file_path, file_name)
        with open(full_path, "w") as f:
            import json
            json.dump(generation_data, f, indent=4)
        double_border_my_word(f"Saved generation data to {full_path}")

    #end for(run)
# end main

if __name__ == "__main__":
    main()
