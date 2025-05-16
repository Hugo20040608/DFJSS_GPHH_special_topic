import json
import plotly.graph_objects as go
import os
import config

def get_pareto_front_indices(fitness_values, objectives_directions):
    """
    找出給定 fitness 值列表中的柏拉圖前緣點的索引。

    Args:
        fitness_values (list of lists): 每個內部列表代表一個個體的 fitness 值，例如 [[f1,f2,f3], [f1,f2,f3], ...]。
        objectives_directions (list of str): 指示每個目標是 'min' (最小化) 還是 'max' (最大化)。
                                           例如: ['min', 'min', 'min']

    Returns:
        list: 柏拉圖前緣點在原始列表中的索引。
    """
    num_individuals = len(fitness_values)
    if num_individuals == 0:
        return []

    num_objectives = len(objectives_directions)
    pareto_indices = []

    for i in range(num_individuals):
        is_dominated_by_others = False
        for j in range(num_individuals):
            if i == j:
                continue

            # 檢查 individual j 是否支配 individual i
            j_dominates_i = True
            j_is_strictly_better_in_one_obj = False

            for k in range(num_objectives):
                val_i = fitness_values[i][k]
                val_j = fitness_values[j][k]

                if objectives_directions[k] == 'min':
                    if val_j > val_i:  # j 在這個最小化目標上比 i 差
                        j_dominates_i = False
                        break
                    if val_j < val_i:  # j 在這個最小化目標上嚴格比 i 好
                        j_is_strictly_better_in_one_obj = True
                elif objectives_directions[k] == 'max':
                    if val_j < val_i:  # j 在這個最大化目標上比 i 差
                        j_dominates_i = False
                        break
                    if val_j > val_i:  # j 在這個最大化目標上嚴格比 i 好
                        j_is_strictly_better_in_one_obj = True
                else:
                    raise ValueError(f"未知的目標方向: {objectives_directions[k]}")
            
            if j_dominates_i and j_is_strictly_better_in_one_obj:
                is_dominated_by_others = True
                break
        
        if not is_dominated_by_others:
            pareto_indices.append(i)
            
    return pareto_indices


def plot_generation_3d_with_pareto(data, 
                                   output_dir="3d_plots_pareto", 
                                   objectives_directions=['min', 'min', 'min']):
    """
    讀取演化運算資料，為每一代繪製 3D fitness 散佈圖，並標示柏拉圖前緣。

    Args:
        data (list): 包含一個或多個世代資料的列表。
        output_dir (str): 儲存 HTML 圖檔的目錄名稱。
        objectives_directions (list of str): 指示 fitness 值的最佳化方向，
                                             例如 ['min', 'min', 'min']。
    """
    if not isinstance(data, list):
        data = [data]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立目錄: {output_dir}")

    for generation_data in data:
        generation_number = generation_data.get("generation", "UnknownGeneration")
        individuals = generation_data.get("individuals", [])

        if not individuals:
            print(f"第 {generation_number} 代沒有個體資料可供繪製。")
            continue

        all_fitness_values = []
        all_hover_texts = []
        valid_individual_indices = [] # 儲存那些 fitness 格式正確的個體在原 individuals 列表中的索引

        for idx, ind in enumerate(individuals):
            fitness = ind.get("fitness", [])
            if len(fitness) == len(objectives_directions):
                all_fitness_values.append(fitness)
                hover_text = (f"Routing: {ind.get('routing', 'N/A')}<br>"
                              f"Sequencing: {ind.get('sequencing', 'N/A')}<br>"
                              f"Fitness: ({fitness[0]:.2f}, {fitness[1]:.2f}, {fitness[2]:.2f})")
                all_hover_texts.append(hover_text)
                valid_individual_indices.append(idx)
            else:
                print(f"警告：在第 {generation_number} 代中，個體 {idx} 的 fitness 值數量 ({len(fitness)}) 與目標方向數量 ({len(objectives_directions)}) 不符。將忽略此個體。")

        if not all_fitness_values:
            print(f"第 {generation_number} 代沒有有效的 fitness 資料可供繪製。")
            continue

        # 找出柏拉圖前緣點的索引 (基於 all_fitness_values 這個列表)
        pareto_indices_in_filtered_list = get_pareto_front_indices(all_fitness_values, objectives_directions)

        # 分離柏拉圖前緣點和非柏拉圖前緣點的座標和懸停文字
        pareto_x, pareto_y, pareto_z, pareto_hover = [], [], [], []
        non_pareto_x, non_pareto_y, non_pareto_z, non_pareto_hover = [], [], [], []

        for i in range(len(all_fitness_values)):
            fitness = all_fitness_values[i]
            hover_text = all_hover_texts[i]
            if i in pareto_indices_in_filtered_list:
                pareto_x.append(fitness[0])
                pareto_y.append(fitness[1])
                pareto_z.append(fitness[2])
                pareto_hover.append(hover_text)
            else:
                non_pareto_x.append(fitness[0])
                non_pareto_y.append(fitness[1])
                non_pareto_z.append(fitness[2])
                non_pareto_hover.append(hover_text)
        
        # 建立 3D 散佈圖物件
        fig_traces = []
        if non_pareto_x: # 如果有非柏拉圖點
            fig_traces.append(go.Scatter3d(
                x=non_pareto_x, y=non_pareto_y, z=non_pareto_z,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.6),
                text=non_pareto_hover,
                hoverinfo='text',
                name='被支配的點 (Dominated)'
            ))
        if pareto_x: # 如果有柏拉圖點
            fig_traces.append(go.Scatter3d(
                x=pareto_x, y=pareto_y, z=pareto_z,
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.9, symbol='diamond'), # 使用菱形標記
                text=pareto_hover,
                hoverinfo='text',
                name='柏拉圖前緣 (Pareto Front)'
            ))
        
        if not fig_traces:
            print(f"第 {generation_number} 代沒有點可繪製。")
            continue

        fig = go.Figure(data=fig_traces)

        fig.update_layout(
            title=f"第 {generation_number} 代 - Fitness 3D 散佈圖 (含柏拉圖前緣)",
            scene=dict(
                xaxis_title=f"{config.OBJECTIVE_TYPE[0]} ({objectives_directions[0]})",
                yaxis_title=f"{config.OBJECTIVE_TYPE[1]} ({objectives_directions[1]})",
                zaxis_title=f"{config.OBJECTIVE_TYPE[2]} ({objectives_directions[2]})"
            ),
            legend_title_text='圖例',
            margin=dict(r=20, b=10, l=10, t=50)
        )

        file_path = os.path.join(output_dir, f"generation_{generation_number}_pareto_plot.html")
        fig.write_html(file_path)
        print(f"第 {generation_number} 代的柏拉圖 3D 圖已儲存至: {file_path}")

# --- 主程式執行部分 ---
if __name__ == "__main__":
    # 設定目標方向(越大越好或越小越好):
    optimization_directions = ['min', 'min', 'min']

    # 從檔案讀取:
    for run in range(len(config.RANDOMSEED)):
        base_directory = "."
        data_subdirectory = "RawData"
        file_name = f"generation_data_run{run}.json"
        full_file_path = os.path.join(base_directory, data_subdirectory, file_name)
        try:
            with open(full_file_path, 'r', encoding='utf-8') as f:
                all_generations_data = json.load(f)
            print("\n--- 從檔案讀取資料並處理 (含柏拉圖前緣) ---")
            plot_generation_3d_with_pareto(all_generations_data, 
                                        output_dir="3D_Graphs",
                                        objectives_directions=optimization_directions)
        except FileNotFoundError:
            print("\n錯誤：找不到 'evolution_data.json' 檔案。")
        except json.JSONDecodeError:
            print("\n錯誤：'evolution_data.json' 檔案格式不正確。")