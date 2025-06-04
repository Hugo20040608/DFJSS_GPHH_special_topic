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

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# def analyze_fitness_distances(population, gen_idx=0):
#     """分析族群中個體在適應度空間的距離分佈，並提供閾值選擇建議
    
#     Args:
#         population: 要分析的族群
#         gen_idx: 當前世代索引，用於圖片命名
#     """
#     # 建立儲存圖片的資料夾
#     histogram_dir = os.path.join(".", "DistanceHistograms")
#     threshold_dir = os.path.join(".", "ThresholdAnalysis")
    
#     # 確保資料夾存在
#     for dir_path in [histogram_dir, threshold_dir]:
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#             print(f"創建資料夾: {dir_path}")
    
#     # 計算所有個體對之間的距離
#     distances = []
#     for ind1, ind2 in combinations(population, 2):
#         # 計算歐氏距離
#         dist = np.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(ind1.fitness.values, ind2.fitness.values)))
#         distances.append(dist)
    
#     # 繪製直方圖
#     plt.figure(figsize=(10, 6))
#     plt.hist(distances, bins=30, edgecolor='black')
#     plt.title('Fitness Distance Distribution')
#     plt.xlabel('Distance between individuals')
#     plt.ylabel('Frequency')
#     plt.grid(True, alpha=0.3)
    
#     # 添加統計資訊
#     min_dist = min(distances)
#     max_dist = max(distances)
#     mean_dist = np.mean(distances)
#     median_dist = np.median(distances)
    
#     plt.axvline(min_dist, color='r', linestyle='--', alpha=0.7, label=f'Min: {min_dist:.4f}')
#     plt.axvline(max_dist, color='g', linestyle='--', alpha=0.7, label=f'Max: {max_dist:.4f}')
#     plt.axvline(mean_dist, color='b', linestyle='--', alpha=0.7, label=f'Avg: {mean_dist:.4f}')
#     plt.axvline(median_dist, color='m', linestyle='--', alpha=0.7, label=f'Mid: {median_dist:.4f}')
    
#     plt.legend()
#     plt.tight_layout()
    
#     # 儲存直方圖到指定資料夾
#     histogram_path = os.path.join(histogram_dir, f"distance_histogram_gen{gen_idx}.png")
#     plt.savefig(histogram_path)
#     # plt.show()
#     plt.close()
    
#     # 分析不同閾值的影響
#     thresholds = np.linspace(min_dist, max_dist, 20)
#     below_threshold = [sum(d < t for d in distances) for t in thresholds]
    
#     # 繪製閾值分析圖
#     plt.figure(figsize=(10, 6))
#     plt.plot(thresholds, below_threshold, 'o-')
#     plt.title('Distance Threshold Analysis')
#     plt.xlabel('distance threshold')
#     plt.ylabel('Number of pairs below threshold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     # 儲存閾值分析圖到指定資料夾
#     threshold_path = os.path.join(threshold_dir, f"threshold_analysis_gen{gen_idx}.png")
#     plt.savefig(threshold_path)
#     # plt.show()
#     plt.close()
    
#     return distances

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
        
        # 確保 offspring 中的每個個體都被評估過
        # 這裡使用 toolbox.map 來並行評估個體，多核心處理
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 先把 offspring 加入 population，再進行相似度分析
        population = population + offspring

        # ---------------------- 表現型評估 ----------------------
        if gen % 5 == 0 and gen <= config.PC_PROCESSING_GENERATIONS_UPPERBOUND:
            to_remove_indices = set()
            for (idx1, ind1), (idx2, ind2) in combinations(enumerate(population), 2):
                if idx1 in to_remove_indices or idx2 in to_remove_indices:
                    continue

                # 計算個體之間的距離
                dist = np.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(ind1.fitness.values, ind2.fitness.values)))
                if dist < config.IND_DISTANCE_THRESHOLD:
                    print(f"個體 {idx1} 和 {idx2} 的距離小於閾值，進行相似度檢查...")
                    if not hasattr(ind1, 'cached_PC'):
                        mti1 = MultiTreeIndividual(gp.PrimitiveTree(ind1[0]), gp.PrimitiveTree(ind1[1]))
                        ind1.cached_PC = toolbox.phenotypic_evaluate(mti1)
                
                    if not hasattr(ind2, 'cached_PC'):
                        mti2 = MultiTreeIndividual(gp.PrimitiveTree(ind2[0]), gp.PrimitiveTree(ind2[1]))
                        ind2.cached_PC = toolbox.phenotypic_evaluate(mti2)

                    similarity = compute_correct_rate(ind1.cached_PC, ind2.cached_PC) * 100
                    if similarity > config.PC_SIMILARITY_THRESHOLD:
                        to_remove_indices.add(idx2)
                        print(f"標記移除: 個體 {idx2}，與個體 {idx1} 相似度為 {similarity:.2f}%")
            # 移除標記的個體
            population = [ind for idx, ind in enumerate(population) if idx not in to_remove_indices]
        
        # population < mu + lambda 時，隨機生成個體補齊數量
        while len(population) < mu + lambda_:
            print(f"Population size {len(population)} is less than mu + lambda_ ({mu + lambda_}), generating new individuals...")
            new_ind = toolbox.individual()
            fit_values = toolbox.evaluate(new_ind)
            new_ind.fitness.values = fit_values
            population.append(new_ind)
        # -------------------------------------------------------

        # ---------------------- 表現型評估 ----------------------
        # 記錄0-5%相似、5~10%相似...每5%做分隔
        # generation_similarity = {f"{i*5}-{(i+1)*5}%": 0 for i in range(20)}
        # ind_PClist = {}
        # # 把 population + offspring 轉換成 MultiTreeIndividual
        # PCeval_ind = [MultiTreeIndividual(gp.PrimitiveTree(ind[0]), gp.PrimitiveTree(ind[1])) for ind in (population + offspring)]
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
        population = toolbox.select(population, mu)
        
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
