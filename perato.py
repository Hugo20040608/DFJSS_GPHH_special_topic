import matplotlib.pyplot as plt
import numpy as np
from deap import tools, gp
import global_vars
import os
import config

def plot_pareto_front(population, objective_labels=("Fitness", "Tree Size"), title="Pareto Front"):
    """
    根據最後一代族群，畫出所有個體的散點圖，
    並用線連接第一層非支配解 (Pareto front)。
    最後將圖形儲存為 .png 到指定路徑:
        "./Graph/Run{global_vars.run:02d}/generation_{global_vars.gen:02d}.png"

    參數:
      population: 最後一代族群，每個個體需具備 fitness.values，格式為 (error, tree_size)
      objective_labels: 一個 tuple，分別為 X 與 Y 軸的標籤。預設為 ("Fitness Error", "Tree Size")
      title: 圖表標題
    """
    # 取得所有個體的 fitness 值
    all_points = [ind.fitness.values for ind in population]
    if not all_points:
        print("族群中沒有個體可繪圖")
        return

    errors = [pt[0] for pt in all_points]
    tree_sizes = [pt[1] for pt in all_points]

    plt.figure(figsize=(8, 6))
    # 繪製整個族群的散點圖 (藍色點，設定透明度，確保不會被遮蔽)
    plt.scatter(errors, tree_sizes, c='blue', edgecolors='k', s=50, alpha=0.6, label="Population", zorder=1)
    
    # 取得非支配前沿的解 (第一層)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_points = [ind.fitness.values for ind in pareto_front]
    
    if pareto_points:
        # 根據第一個目標排序，使連線順序正確
        pareto_points = sorted(pareto_points, key=lambda x: x[0])
        pareto_errors = [pt[0] for pt in pareto_points]
        pareto_tree_sizes = [pt[1] for pt in pareto_points]
        
        # 用紅色線連接非支配解
        plt.plot(pareto_errors, pareto_tree_sizes, color='red', lw=2, label="Pareto Front", zorder=2)
        # 再以紅色散點標出非支配解
        plt.scatter(pareto_errors, pareto_tree_sizes, c='red', edgecolors='k', s=70, zorder=3)
    
    plt.xlabel(objective_labels[0])
    plt.ylabel(objective_labels[1])
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # # 固定 x 軸和 y 軸的範圍
    plt.xlim(config.PLOT_PARETO_X_SCALE[0], config.PLOT_PARETO_X_SCALE[1])
    plt.ylim(config.PLOT_PARETO_Y_SCALE[0], config.PLOT_PARETO_Y_SCALE[1])  
    
    # 使用全域變數 RUN_NUMBER 與 GEN_NUMBER 來決定儲存路徑
    file_path = f"./Graph/Run{global_vars.run:02d}/generation_{global_vars.gen:02d}.png"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)
    plt.close()  # 關閉圖形以釋放記憶體
    

def print_pareto_front(population):
    if config.OBJECTIVE_TYPE == "MULTI":
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 過濾重複的 (fitness error, tree size) 組合
        unique_pf = {}
        for ind in pareto_front:
            key = tuple(ind.fitness.values)  # 假設這是一個 (error, tree_size) tuple
            if key not in unique_pf:
                unique_pf[key] = ind

        keyIdx = 1
        print("\nUnique Pareto Front Solutions:")
        for key, ind in unique_pf.items():
            print(f"Ind#{keyIdx:02d}: {gp.PrimitiveTree(ind[0])}  |  {gp.PrimitiveTree(ind[1])}  |  {key}")
            keyIdx += 1
        print()
    else:
        best_ind = tools.selBest(population, 1)[0]
        print("\nBest Individual:")
        print(f"Fitness: {best_ind.fitness.values[0]}")
        print(f"Expression: {gp.PrimitiveTree(best_ind[0])}  |  {gp.PrimitiveTree(best_ind[1])}")