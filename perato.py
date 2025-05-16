import matplotlib.pyplot as plt
import numpy as np
from deap import creator, base, tools, gp
import os
import json
import config
from something_cool import double_border_my_word

def create_deap_individual():
    """
    定義 DEAP 的個體和 fitness，如果尚未定義。
    """
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # 假設是雙目標最小化問題
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def load_population_from_json(data, generation):
    """
    從 JSON 資料中重建 DEAP 的 population。

    參數:
        data: 已解析的 JSON 資料。
        generation: 要重建的世代索引。

    返回:
        population: 重建的 DEAP population。
    """
    create_deap_individual()

    # 確保指定的世代存在
    if generation >= len(data):
        raise IndexError(f"Generation {generation} 不存在於資料中。")

    # 取得對應世代的資料
    generation_data = data[generation]

    # 重建 population
    population = []
    for ind_data in generation_data["individuals"]:
        ind = creator.Individual()  # 空的個體
        ind.fitness.values = tuple(ind_data["fitness"])  # 設定 fitness 值
        population.append(ind)

    return population


def plot_pareto_front(population, output_dir, generation, objective_labels=("Fitness 1", "Fitness 2")):
    """
    根據族群資料繪製 Pareto Front，並儲存為圖片。

    參數:
        population: DEAP 的 population。
        output_dir: 圖片輸出的目錄。
        generation: 當前世代索引。
        objective_labels: X 和 Y 軸的標籤。
    """
    # 取得所有個體的 fitness 值
    all_points = [ind.fitness.values for ind in population]
    if not all_points:
        print(f"Generation {generation}: 無個體可繪圖")
        return

    errors = [pt[0] for pt in all_points]
    tree_sizes = [pt[1] for pt in all_points]

    plt.figure(figsize=(8, 6))
    plt.scatter(errors, tree_sizes, c='blue', edgecolors='k', s=50, alpha=0.6, label="Population", zorder=1)

    # 取得非支配前沿的解 (第一層)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_points = [ind.fitness.values for ind in pareto_front]

    if pareto_points:
        pareto_points = sorted(pareto_points, key=lambda x: x[0])
        pareto_errors = [pt[0] for pt in pareto_points]
        pareto_tree_sizes = [pt[1] for pt in pareto_points]

        plt.plot(pareto_errors, pareto_tree_sizes, color='red', lw=2, label="Pareto Front", zorder=2)
        plt.scatter(pareto_errors, pareto_tree_sizes, c='red', edgecolors='k', s=70, zorder=3)

    plt.xlabel(objective_labels[0])
    plt.ylabel(objective_labels[1])
    plt.title(f"Pareto Front (Generation {generation})")
    plt.grid(True)
    plt.legend()

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 儲存圖片
    output_file = os.path.join(output_dir, f"generation_{generation:02d}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Generation {generation} 圖片已儲存至: {output_file}")
    

def print_pareto_front(population):
    """
    印出 Pareto Front 的解。
    這個函數會列印出每個解的 fitness 值和對應的樹結構。

    參數:
        population: DEAP 的 population。
    """
    # 多目標的情況下，使用非支配排序
    if len(config.OBJECTIVE_TYPE) > 1:
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # 過濾重複的 (fitness error, tree size) 組合
        unique_pf = {}
        for ind in pareto_front:
            key = tuple(ind.fitness.values)
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


def plot_all_generations(json_file, output_dir):
    """
    讀取 JSON 檔案，為每個 generation 繪製 Pareto Front。

    參數:
        json_file: JSON 檔案的路徑。
        output_dir: 圖片輸出的目錄。
    """
    # 讀取 JSON 檔案
    if not os.path.exists(json_file):
        print(f"檔案不存在: {json_file}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    for gen in range(len(data)):
        population = load_population_from_json(data, gen)
        plot_pareto_front(population, output_dir, gen, objective_labels=config.OBJECTIVE_TYPE)


def main():
    """
    主程式，處理多個 Run 的資料並繪製 Pareto Front。
    """
    for run in range(len(config.RANDOMSEED)):
        json_file = os.path.join(".", "RawData", f"generation_data_run{run}.json")
        output_dir = os.path.join(".", "Graph", f"Run{run:02d}")

        if not os.path.exists(json_file):
            double_border_my_word(f"Run {run}: JSON doesn't exist, terminate.")
            break

        print(f"> 處理 Run {run} 的資料...")
        plot_all_generations(json_file, output_dir)


if __name__ == "__main__":
    main()