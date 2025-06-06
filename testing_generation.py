import json
import os
from deap import gp
import config
import simulation
from gp_setup import create_primitive_set, setup_toolbox, MultiTreeIndividual

# 測試設定
TEST_SETTINGS = [
    {"run": 0, "mean_time": 50,  "seeds": list(range(6))},
    {"run": 1, "mean_time": 100, "seeds": list(range(6))},
    {"run": 2, "mean_time": 200, "seeds": list(range(6))}
]

def load_last_generation(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    last_gen = data[-1]["individuals"]
    return last_gen

def set_mean_processing_time(mean_time):
    config.PROCESSING_TIME_LOWER = 1
    config.PROCESSING_TIME_UPPER = mean_time * 2 - 1
    config.MEAN_PROCESSING_TIME = mean_time

def test_individual(routing_str, sequencing_str, mean_time, seeds):
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)
    routing_tree = gp.PrimitiveTree.from_string(routing_str, pset)
    sequencing_tree = gp.PrimitiveTree.from_string(sequencing_str, pset)
    routing_func = toolbox.compile(expr=routing_tree)
    sequencing_func = toolbox.compile(expr=sequencing_tree)
    tree_size = len(routing_tree) + len(sequencing_tree)  # 計算樹的大小
    fitness_list = []
    for seed in seeds:
        fit = simulation.simulate(routing_func, sequencing_func, random_seed=seed)
        fitness_list.append(fit)
    # 取平均
    avg_fitness = tuple(float(sum(f[i] for f in fitness_list)/len(fitness_list)) for i in range(len(fitness_list[0])))
    avg_fitness += (tree_size,)  # 加入樹的大小到 fitness 中
    # 只回傳 OBJECTIVE_TYPE 指定的指標
    selected_fitness = [avg_fitness[config.PI_mapping[obj]] for obj in config.OBJECTIVE_TYPE]
    return selected_fitness

def main():
    # 設定輸入和輸出目錄(記得改)
    input_dir = os.path.join("FeatureSelection", "delete NIQ, WIQ, MWT", "RawData")
    output_dir = os.path.join("FeatureSelection", "delete NIQ, WIQ, MWT", "TestingRawData")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No json files found in {input_dir}")
        return

    for input_file in json_files:
        input_json = os.path.join(input_dir, input_file)
        last_gen = load_last_generation(input_json)
        for setting in TEST_SETTINGS:
            set_mean_processing_time(setting["mean_time"])
            result = [{
                "generation": "test",
                "individuals": []
            }]
            for ind in last_gen:
                routing = ind["routing"]
                sequencing = ind["sequencing"]
                fitness = test_individual(routing, sequencing, setting["mean_time"], setting["seeds"])
                result[0]["individuals"].append({
                    "routing": routing,
                    "sequencing": sequencing,
                    "fitness": list(fitness)
                })
            # 儲存結果
            base = os.path.splitext(input_file)[0]
            outname = f"{base}_test{setting['run']}.json"
            outpath = os.path.join(output_dir, outname)
            with open(outpath, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
