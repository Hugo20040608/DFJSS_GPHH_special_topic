import json
import math
import os

def dominates(a, b):
    """判斷 a 是否支配 b（所有目標都不差，且至少一個更好）"""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def is_pareto_front(fitness_list):
    """傳回 fitness_list 中的 Pareto front（非支配解，且去除重複）"""
    front = []
    for i, fit_i in enumerate(fitness_list):
        if not any(dominates(fit_j, fit_i) for j, fit_j in enumerate(fitness_list) if j != i):
            front.append(fit_i)
    # 去除重複點
    unique_front = []
    seen = set()
    for fit in front:
        t = tuple(fit)
        if t not in seen:
            unique_front.append(fit)
            seen.add(t)
    return unique_front

def load_front(path):
    """從 json 檔讀取所有世代，回傳最後一代的 Pareto front fitness list"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 如果 json 是一個世代列表，取最大 generation
    if isinstance(data, list) and all('generation' in gen for gen in data):
        last_gen = max(data, key=lambda g: g['generation'])
        fits = [ind['fitness'] for ind in last_gen['individuals']]
    else:
        # 否則假設直接就是個體列表
        fits = [ind['fitness'] for ind in data]
    # 只取 Pareto front
    front = is_pareto_front(fits)
    return front

def euclidean(a, b):
    """計算兩個向量的歐氏距離"""
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def igd(approx, ideal):
    """
    計算 IGD：
    對 ideal 中每個點 v，找出 approx 中最近的距離，然後平均
    """
    total = 0.0
    for v in ideal:
        dmin = min(euclidean(v, u) for u in approx)
        total += dmin
        print(dmin)
    return total / len(ideal)

def main():
    for run in range(1):
        for i in range(1):
            # 理想前緣檔案
            ideal_path = f"global_pareto_front_test{i}.json"
            if not os.path.isfile(ideal_path):
                print(f"找不到 {ideal_path}")
                return

            ideal_front = load_front(ideal_path)
            print(f"載入理想前緣，共 {len(ideal_front)} 個點")

            # 要計算的測試檔案
            test_path = f'generation_run{run}_test{i}.json'
            if not os.path.isfile(test_path):
                print(f"找不到 {test_path}，跳過")
                continue
            approx_front = load_front(test_path)
            print(approx_front)
            value = igd(approx_front, ideal_front)
            print(f"IGD({test_path}) = {value:.6f}")

if __name__ == '__main__':
    main()
