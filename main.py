# main.py

import multiprocessing
import os
import random
import numpy as np
import pandas as pd
from deap import tools, algorithms, gp
import dill as pickle
import config
import itertools
from gp_setup import create_primitive_set, setup_toolbox
from gp_setup import MultiTreeIndividual
from perato import print_pareto_front
from something_cool import double_border_my_word
from evaluation_PC import compute_correct_rate
from phenotypic import Event, Workpiece, Factory


def output_logbook(logbook, run):
     # 假設 logbook 已經定義並產生，並且有以下各項統計資料：
    ngen = logbook.select("gen")          # generations，欄位名稱為 "gen"
    nevals = logbook.select("nevals")      # 每一代的個體評估次數

    # 預設欄位與資料
    data = {
        "ngen": ngen,
        "nevals": nevals,
    }

    for obj in config.OBJECTIVE_TYPE:
        prefix = f"{obj}_"
        chapter = logbook.chapters[obj]
        data.update({
            f"{prefix}max": chapter.select("max"),
            f"{prefix}min": chapter.select("min"),
            f"{prefix}avg": chapter.select("avg"),
            f"{prefix}std": chapter.select("std"),
        })

    df = pd.DataFrame(data).round(4)
    print(df)

    if config.LOGBOOK_SAVEON is not None:
        save_dir = os.path.dirname(config.LOGBOOK_SAVEON)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        file_name = f"logbook_run{run}.csv"
        save_dir = os.path.join(save_dir, file_name)
        df.to_csv(save_dir, index=False)
        print(f"Successfully saved logbook to {save_dir}!")

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 用於存儲每一代的樹和 fitness 值
    generation_data = []

    # 評估初始族群
    # 改為使用 toolbox.map 來並行評估個體，多核心處理
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 對初始族群使用 NSGA-II 排序 (排序依據多目標)
    population = toolbox.select(population, len(population))

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
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

        # 產生子代
        offspring = toolbox.select(population, lambda_)
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
        # 這裡使用 toolbox.map 來並行評估個體，多核心處理
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # ---------------------- 表現型評估 ----------------------
        # 記錄0-5%相似、5~10%相似...每5%做分隔
        # generation_similarity = {f"{i*5}-{(i+1)*5}%": 0 for i in range(20)}
        # ind_PClist = {}
        # # 把 population + offspring 轉換成 MultiTreeIndividual
        # PCeval_ind = [MultiTreeIndividual(gp.PrimitiveTree(ind[0]), gp.PrimitiveTree(ind[0])) for ind in (population + offspring)]
        # ind_PC = list(map(toolbox.phenotypic_evaluate, PCeval_ind))
        # for ind, pc in zip(PCeval_ind, ind_PC):
        #     ind_PClist[ind] = pc
        
        # for indA, indB in itertools.combinations(PCeval_ind, 2):
        #     similarity = compute_correct_rate(ind_PClist[indA], ind_PClist[indB]) * 100
        #     # 根據相似度分組
        #     for i in range(20):
        #         if similarity >= i * 5 and similarity <= (i + 1) * 5:
        #             generation_similarity[f"{i*5}-{(i+1)*5}%"] += 1
        #             break
        # # 輸出
        # double_border_my_word(
        #     f"Generation {gen} similarity distribution:",
        #     f"{generation_similarity}"
        # )
        # --------------------------------------------------------
        
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
    
    return population, logbook, generation_data

def tree_sizes(ind):
    return (len(ind[0])+len(ind[1]))

def main():
    # 1. 建立 primitive set 與 toolbox
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)

    # 設定隨機種子，方便重現結果
    for run in range(len(config.RANDOMSEED)):
        random.seed(config.RANDOMSEED[run])
        
        with multiprocessing.Pool() as pool:
            # 將 map 函數替換為 pool.map，讓評估過程並行執行
            toolbox.register("map", pool.map)

            # 2. 初始化種群
            population = toolbox.population(n=config.POP_SIZE)

            # 3. 統計資訊設定：收集 fitness 與個體大小的統計數據 (如果需要記錄多目標統計，可依需求修改)
            stats = {}
            for idx, obj in enumerate(config.OBJECTIVE_TYPE):
                stats[obj] = tools.Statistics(lambda ind, i=idx: ind.fitness.values[i])
            mstats = tools.MultiStatistics(**stats)
            mstats.register("avg", np.mean, axis=0)
            mstats.register("std", np.std, axis=0)
            mstats.register("min", np.min, axis=0)
            mstats.register("max", np.max, axis=0)
            
            # # 4. 執行演化流程 (eaSimple：基本演化程序)
            # if len(config.OBJECTIVE_TYPE) == 0:
            #     population, logbook = algorithms.eaSimple(
            #         population, toolbox,
            #         cxpb=config.CX_PROB, mutpb=config.MUT_PROB,
            #         ngen=config.GENERATIONS,
            #         stats=mstats, halloffame=None,
            #         verbose=config.VERBOSE
            #     )
            # # --------------------------------------
            # else:
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
                output_logbook(logbook, run)

            # 輸出最佳個體結果，取得非支配前沿的第一層解
            print_pareto_front(population)
            
            # 儲存 generation_data 到檔案
            file_path = os.path.join(".", "RawData")
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
    multiprocessing.set_start_method("spawn")
    main()
