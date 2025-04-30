# 程式運作環境創建的作業講義程式系統

## 專案簡介

此專案使用遺傳規劃演算法 (Genetic Programming, GP) 來解決動態彈性零工式工廠排程問題 (Dynamic Flexible Job Shop Scheduling Problem, DFJSSP)。我們透過 DEAP (Distributed Evolutionary Algorithms in Python) 函式庫完成遺傳規劃演算法，針對機台分配 (routing) 和製程排序 (sequencing) 問題進行運算。

## 系統架構

本專案主要由下列幾個模組組成：

- **main.py**：主程式，執行遺傳演算法的演化過程
- **config.py**：配置文件，包含各種參數設定
- **simulation.py**：模擬工廠離散事件，評估個體表現用
- **gp_setup.py**：設置遺傳個體規則，定義每個個體長度和特徵
- **evaluation.py**：評估個體適應度能力
- **pareto.py**：產生並分析柏拉圖前沿 (Pareto Front)
- **global_vars.py**：存儲全局變數數據

## 功能特點

- **多樹遺傳規劃演算法(Multi-Tree Genetic Programming, MTGP)**：每個個體包含兩種表現形式，分別用於分配機台 (routing rule) 和製程排序 (sequencing rule)
- **多目標最佳化**：可以同時檢查多個指標，如平均完工時間 (Mean Flow Time) 和樹大小 (Tree Size)
- **模擬性**：透過模擬動態彈性零工式工廠的離散事件，實際分析最佳結果
- **實驗設定調整**：通過 **config.py**，可以調整整個系統和多數模擬環境設定

## 使用方法

1. 確保你已經安裝所需的依賴套件：
   ```bash
   pip install deap numpy pandas matplotlib
   ```

2. 根據需求調整改進 **config.py** 中的參數設定：
   - 工廠問題參數 (機台數量、工件數量等)
   - 演化演算法參數 (族群大小、世代數等)
   - 最佳化目標 (單目標或多目標)

3. 執行主程式：
   ```bash
   python main.py
   ```

4. 結果輸出，統計結果會存在 **CSVs** 資料夾中，個體分布和柏拉圖前緣的圖則會放在 **Graph**資料夾中

## 參考文獻

- Yi Mei: *Evolving Time-Invariant Dispatching Rules in Job Shop Scheduling with Genetic Programming*
- ZFF: *Representations with Multi-tree and Cooperative Coevolution*