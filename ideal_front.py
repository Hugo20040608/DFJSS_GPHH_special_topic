import json
import os
import glob
import time
import multiprocessing
import numpy as np
from functools import partial

def is_dominated_np(fitness1, fitness2):
    """使用numpy加速的支配檢查"""
    return np.all(fitness2 <= fitness1) and np.any(fitness2 < fitness1)

def load_run_data(file_path):
    """載入單一執行檔案的資料"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"載入檔案 {file_path} 時發生錯誤: {e}")
        return []

def process_run_file(file_path):
    """處理單一執行檔案並收集不重複個體"""
    print(f"處理檔案: {file_path}")
    start_time = time.time()
    
    run_data = load_run_data(file_path)
    unique_individuals_dict = {}
    
    for generation in run_data:
        for ind in generation.get("individuals", []):
            if "routing" in ind and "sequencing" in ind and "fitness" in ind and ind["fitness"]:
                # 建立個體的唯一識別碼（使用routing, sequencing和fitness）
                fitness_tuple = tuple(ind["fitness"])
                individual_key = (ind["routing"], ind["sequencing"], fitness_tuple)
                
                # 只收集不重複的個體
                if individual_key not in unique_individuals_dict:
                    unique_individuals_dict[individual_key] = ind
    
    elapsed = time.time() - start_time
    print(f"  從檔案 {file_path} 收集到 {len(unique_individuals_dict)} 個不重複個體 (耗時: {elapsed:.2f}秒)")
    return unique_individuals_dict

def fast_non_dominated_sort(individuals):
    """快速非支配排序算法，只返回第一層（Pareto前緣）"""
    n = len(individuals)
    dominated_by = [[] for _ in range(n)]
    domination_count = [0] * n
    pareto_front = []
    
    # 轉換適應度為numpy數組以加速計算
    fitness_arrays = [np.array(ind["fitness"]) for ind in individuals]
    
    # 計算支配關係
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_dominated_np(fitness_arrays[i], fitness_arrays[j]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        # 找出第一層的個體（不被任何人支配）
        if domination_count[i] == 0:
            pareto_front.append(individuals[i])
    
    return pareto_front

def save_pareto_front(pareto_front, output_file):
    """儲存最前緣個體到檔案"""
    pareto_data = [
        {
            "generation": "global_pareto_front",
            "individuals": pareto_front
        }
    ]
    
    with open(output_file, 'w') as f:
        json.dump(pareto_data, f, indent=4)
    
    print(f"找到 {len(pareto_front)} 個全域最前緣個體並儲存至 {output_file}")

def main():
    start_time = time.time()
    
    # 資料檔案的路徑模式
    file_pattern = "RawData/generation_data_run*.json"
    
    # 尋找所有符合模式的檔案
    run_files = glob.glob(file_pattern)
    print(f"找到 {len(run_files)} 個執行資料檔案")
    
    # 並行處理檔案
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(run_files))) as pool:
        results = pool.map(process_run_file, run_files)
    
    # 合併所有處理結果
    unique_individuals_dict = {}
    for result_dict in results:
        unique_individuals_dict.update(result_dict)
    
    # 轉換回列表
    all_individuals = list(unique_individuals_dict.values())
    print(f"總共收集了 {len(all_individuals)} 個不重複個體")
    
    # 找出全域最前緣個體
    print("計算全域最前緣...")
    global_pareto_front = fast_non_dominated_sort(all_individuals)
    
    # 儲存結果
    save_pareto_front(global_pareto_front, "global_pareto_front.json")
    
    total_time = time.time() - start_time
    print(f"總共耗時: {total_time:.2f}秒")

if __name__ == "__main__":
    main()