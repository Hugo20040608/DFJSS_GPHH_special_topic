import config
from deap import gp
import os
import sys
import dill as pickle
import glob
import something_cool
from gp_setup import MultiTreeIndividual
from phenotypic import evaluate_cutpoint_decision
from gp_setup import create_primitive_set, setup_toolbox
from phenotypic import Event, Workpiece, Factory
import traceback

def compute_correct_rate(list_A, list_B):
    """
    計算兩個評估列表的正確率。
    假設每個列表都是一個包含評估結果的字典，並且有一個 'correct' 鍵。
    """
    if len(list_A) != len(list_B):
        raise ValueError("The two lists must have the same length.")

    correct_count = sum(1 for a, b in zip(list_A, list_B) if a == b)
    return correct_count / len(list_A) if list_A else 0.0

def evaluate_PC(individual):
    """
    評估個體在所有 factory_snapshots 下的快照表現。
    input: 
    individual type => MultiTreeIndividual
    """
    result = []
    SNAPSHOT_DIR = os.path.abspath("./factory_snapshots")
    pkl_files = glob.glob(f"{SNAPSHOT_DIR}/*.pkl")
    for fileName in pkl_files:
        try:
            pset = create_primitive_set()
            result.append(evaluate_cutpoint_decision(fileName, individual, pset))
        except Exception as e:
            print(f"Error evaluating {fileName}: {e}")
            traceback.print_exc()
            continue

    if len(result) == 0:
        something_cool.double_border_my_word(
            "No valid evaluations found.",
            "Please check your factory snapshots or individual definitions."
        )
        sys.exit(1)

    return result

def test_specific_PC():
    # 1. 建立原始集合與工具箱
    pset = create_primitive_set()
    
    # 2. 手動定義想要測試的規則
    routing_str_A = "PT"  # 可替換為你想測試的規則
    sequencing_str_A = "PT"  # 可替換為你想測試的規則
    routing_tree_A = gp.PrimitiveTree.from_string(routing_str_A, pset)
    sequencing_tree_A = gp.PrimitiveTree.from_string(sequencing_str_A, pset)
    individual_A = MultiTreeIndividual(routing_tree_A, sequencing_tree_A)
    
    routing_str_B = "add(OWT, OWT)"  # 可替換為你想測試的規則
    sequencing_str_B = "sub(APTQ, OWT)"  # 可替換為你想測試的規則
    routing_tree_B = gp.PrimitiveTree.from_string(routing_str_B, pset)
    sequencing_tree_B = gp.PrimitiveTree.from_string(sequencing_str_B, pset)
    individual_B = MultiTreeIndividual(routing_tree_B, sequencing_tree_B)  # 可以是不同的個體
    
    list_A = evaluate_PC(individual_A)
    list_B = evaluate_PC(individual_B)
    correctRate = compute_correct_rate(list_A, list_B)

    something_cool.border_my_word(
        "Rules for testing individuals:",
        f"- Rule A: {routing_str_A} | {sequencing_str_A}",
        f"- Rule B: {routing_str_B} | {sequencing_str_B}",
        f"- Correct Rate: {correctRate:.2%}"
    )


def main():
    """
    主函數，執行 PC 評估。
    """
    test_specific_PC()

if __name__ == "__main__":
    main()
