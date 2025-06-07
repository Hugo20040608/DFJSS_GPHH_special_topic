import json
import plotly.graph_objects as go
import os
import config

def get_pareto_front_indices(fitness_values, objectives_directions):
    """找出給定 fitness 值列表中的柏拉圖前緣點的索引。"""
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

def extract_pareto_front_from_file(file_path, objectives_directions=['min', 'min', 'min']):
    """從指定檔案中提取Pareto前緣點"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"錯誤：檔案格式不正確 {file_path}")
        return [], []

    if not isinstance(data, list):
        data = [data]
    
    # 收集所有世代的所有個體
    all_fitness_values = []
    all_hover_texts = []
    
    for generation_data in data:
        individuals = generation_data.get("individuals", [])
        for ind in individuals:
            fitness = ind.get("fitness", [])
            if len(fitness) == len(objectives_directions):
                all_fitness_values.append(fitness)
                hover_text = (f"Routing: {ind.get('routing', 'N/A')}<br>"
                              f"Sequencing: {ind.get('sequencing', 'N/A')}<br>"
                              f"Fitness: ({fitness[0]:.2f}, {fitness[1]:.2f}, {fitness[2]:.2f})<br>"
                              f"From: {os.path.basename(file_path)}")
                all_hover_texts.append(hover_text)
    
    if not all_fitness_values:
        print(f"檔案 {file_path} 中沒有有效的適應度資料")
        return [], []
        
    # 找出所有個體中的Pareto前緣
    pareto_indices = get_pareto_front_indices(all_fitness_values, objectives_directions)
    
    # 提取Pareto前緣點
    pareto_fitness_values = [all_fitness_values[i] for i in pareto_indices]
    pareto_hover_texts = [all_hover_texts[i] for i in pareto_indices]
    
    print(f"從 {file_path} 中提取了 {len(pareto_fitness_values)} 個Pareto前緣點")
    return pareto_fitness_values, pareto_hover_texts

def plot_combined_pareto_fronts(file_paths, output_dir="Combined_Pareto_Plots", 
                               output_file="combined_pareto_front.html", 
                               objectives_directions=['min', 'min', 'min']):
    """將多個檔案的Pareto前緣點繪製在同一張圖上"""
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立目錄: {output_dir}")
    
    # 六個不同顏色
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    fig_traces = []
    
    # 處理每個檔案
    for i, file_path in enumerate(file_paths):
        color = colors[i % len(colors)]  # 循環使用顏色
        file_name = os.path.basename(file_path)
        
        # 提取Pareto前緣
        pareto_values, pareto_texts = extract_pareto_front_from_file(file_path, objectives_directions)
        
        if pareto_values:
            # 分離x, y, z座標
            x = [val[0] for val in pareto_values]
            y = [val[1] for val in pareto_values]
            z = [val[2] for val in pareto_values]
            
            # 添加此檔案的Pareto前緣到圖表
            fig_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.9, symbol='diamond'),
                text=pareto_texts,
                hoverinfo='text',
                name=f'檔案 {i+1} Pareto前緣'
            ))
    
    if not fig_traces:
        print("沒有找到任何Pareto前緣點可供繪製")
        return
    
    # 建立圖表
    fig = go.Figure(data=fig_traces)
    
    # 設定版面配置
    fig.update_layout(
        title="多檔案Pareto前緣比較",
        scene=dict(
            xaxis_title=f"{config.OBJECTIVE_TYPE[0]} ({objectives_directions[0]})",
            yaxis_title=f"{config.OBJECTIVE_TYPE[1]} ({objectives_directions[1]})",
            zaxis_title=f"{config.OBJECTIVE_TYPE[2]} ({objectives_directions[2]})"
        ),
        legend_title_text='圖例',
        margin=dict(r=20, b=10, l=10, t=50)
    )
    
    # 保存HTML檔案
    file_path = os.path.join(output_dir, output_file)
    fig.write_html(file_path)
    print(f"已將多檔案Pareto前緣比較圖儲存至: {file_path}")

# --- 主程式執行部分 ---
if __name__ == "__main__":
    # 設定目標方向(越大越好或越小越好)
    optimization_directions = ['min', 'min', 'min']
    
    # 設定檔案路徑
    base_directory = "."
    data_subdirectory = "RawData"
    
    # 要繪製的6個檔案列表
    file_names = [f"{i}.json" for i in range(1, 3)]
    file_paths = [os.path.join(base_directory, data_subdirectory, file_name) for file_name in file_names]
    
    # 繪製合併的Pareto前緣
    plot_combined_pareto_fronts(
        file_paths, 
        output_dir="Combined_Pareto_Plots",
        output_file="combined_pareto_front_1_to_6.html",
        objectives_directions=optimization_directions
    )