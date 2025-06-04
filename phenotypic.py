from gp_setup import create_primitive_set, setup_toolbox, MultiTreeIndividual
import something_cool
from deap import gp
import dill as pickle
import os
import heapq
import random
import operator
import sys
import statistics
import matplotlib.pyplot as plt
import config

SIMULATION_RAND_PC = random.Random()
CUT_POINT_RNG = random.Random(25)  # 用於 cutpoint 的隨機數生成器

# ---------------------------
# Event 類別
# ---------------------------
class Event:
    def __init__(self, time, event_type, workpiece=None, process_index=None):
        """
        time: 事件發生時間
        event_type: 事件類型，如 'arrival', 'assign_process', 'end_process', 'machine_check'
        workpiece: 相關工件 (Workpiece 物件)
        process_index: 工件當前製程索引
        """
        self.time = time
        self.event_type = event_type
        self.workpiece = workpiece
        self.machine_id = None  # 事件發生的機台(routing role 決定)
        self.process_index = process_index
        
    def __lt__(self, other):
        # 定義事件類型的排序：時間相同時， end_process 最優先，最後才執行 machine_check
        order = {"end_process": 0, "arrival": 1, "assign_process": 2, "machine_check": 3}
        return (self.time, order.get(self.event_type, 99)) < (other.time, order.get(other.event_type, 99))


# ---------------------------
# Workpiece 類別
# ---------------------------
class Workpiece:
    def __init__(self, wp_id, processes):
        """
        wp_id: 工件識別編號
        processes: 製程列表，每個製程為一個字典，例如 {'machine_options': [1,2,3], ''process_time': [10, 20, 30]}
        """
        self.wp_id = wp_id
        self.processes = processes
        self.current_process = 0  # 當前製程索引
        self.arrival_time = None  # 工件抵達時間
        self.arrival_machine_time = None # 工件抵達機台時間
        self.due_date = None      # 工件的要求期限

    def __str__(self):
        return f"Workpiece {self.wp_id} (Process {self.current_process}/{len(self.processes)})"

# ---------------------------
# Factory 類別 (包含機台狀態管理)
# ---------------------------
class Factory:
    def __init__(self, machine_count, workpiece_count, utilization_rate, warmup_count, routing_rule, sequencing_rule, cutpoints=None, rule_name="", random_seed=None):
        """
        machine_count: 機台數量
        workpiece_count: 工件數量
        utilization_rate: 機台利用率設定（影響工件進廠時間）
        warmup_count: 暖場工件數量
        """
        self.machine_count = machine_count
        self.workpiece_count = workpiece_count
        self.utilization_rate = utilization_rate
        self.warmup_count = warmup_count
        self.current_time = 0.0
        self.event_queue = []   # 儲存所有事件的 priority queue
        self.completed_workpieces = []
        self.schedule_records = []  # 記錄每個工件各製程的排程記錄
        self.last_event = None  # 儲存最後處理的事件 (用於 cutpoint 快照)
        # self._ongoing_tasks = {}    # 暫存進行中的任務 (key: (wp_id, process_index))
        # 新增：機台狀態管理，記錄每台機台最早可用時間
        self.machine_status = {machine_id: 0.0 for machine_id in range(1, machine_count+1)}
        # 新增：machine queue，記錄每台機台可被執行的製程
        self.machine_queues = {machine_id: [] for machine_id in range(1, machine_count+1)}

        # 新增：routing rule 和 sequencing rule
        self.routing_rule = routing_rule
        self.sequencing_rule = sequencing_rule

        # 新增：cutpoints
        self.cutpoints = cutpoints if cutpoints is not None else set()
        self.rule_name = rule_name
        self.random_seed = random_seed

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def run(self, simulation_end_time):
        """
        以中央時間管理方式執行模擬，直到 simulation_end_time 或事件耗盡
        """
        while self.event_queue and (simulation_end_time is None or self.current_time < simulation_end_time):
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            self.process_event(event)
        if config.LOGBOOK_ON_SIMULATION:
            if self.event_queue is None or simulation_end_time is None:
                something_cool.double_border_my_word("", f"Simulation ended at time {self.current_time:.2f} with no more events in queue", "")
            elif self.current_time >= simulation_end_time:
                something_cool.double_border_my_word("", f"Simulation ended at time {self.current_time:.2f}, terminate by the time limit", "")
        return self.current_time

    def process_event(self, event):
        # **當前** 事件的工件資訊
        wp = event.workpiece

        if event.event_type == 'arrival':   # 工件抵達工廠
            wp.arrival_time = event.time
            # due date setting (期限設定)，指數分布，平均值約為 1/lambd
            lambd = 1.0 / config.get_mean_processing_time()
            wp.due_date = event.time + len(wp.processes) * SIMULATION_RAND_PC.expovariate(lambd) * config.DUE_DATE_MULTIPLIER
            # log(f"Time {event.time:.2f}: {wp} arrived,  due date: Time {wp.due_date:.2f}")

            # 加入該工件的第0項製程事件
            new_event = Event(event.time, 'assign_process', workpiece=wp, process_index=0)
            self.schedule_event(new_event)

        elif event.event_type == 'assign_process':  # 將製程使用 routing rule 分派至機台
            # log(f"Time {event.time:.2f}: {wp} wants to assign process {event.process_index} to a machine")

            # ------------------------------------- cutpoint -------------------------------------
            if (wp.wp_id, event.process_index) in self.cutpoints:
                self.last_event = event
                filename = f"./factory_snapshots/wp{wp.wp_id}_proc{event.process_index}_seed{self.random_seed}_rule{self.rule_name}_1.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(self, f)
            # ----------------------------------------------------------------------------------

            # 確保製程的機台選項跟對應時間的串列長度一致
            if len(wp.processes[event.process_index]['machine_options']) != len(wp.processes[event.process_index]['process_time']):
                something_cool.double_border_my_word(
                    "[ERROR]: MACHINE OPTIONS ERROR",
                    f"Time {event.time:.2f}: Machine options and process time mismatch for {wp} process {event.process_index}")
                sys.exit(1)

            # ------------------------------------ TERMINAL SETTING ------------------------------------ 
            CT = event.time  # 當前時間
            PT = -1 # setting below
            NPT = 0 if event.process_index+1 == len(wp.processes) else statistics.median(wp.processes[event.process_index+1]['process_time'])
            TIM = CT
            OWT = CT - TIM
            #-
            NOR = len(wp.processes) - event.process_index - 1
            WKR = 0 if event.process_index+1 >= len(wp.processes) else statistics.median([pt for proc in wp.processes[event.process_index+1:] for pt in proc['process_time']])
            # NOR*WKR 為剩下製程的中位數預期時間
            TIS = wp.arrival_time
            #-
            NIQ = -1 # setting below
            WIQ = -1 # setting below
            MWT = -1 # setting below
            APTQ = -1 # setting below
            #-
            DD = wp.due_date - event.time
            SL = 0 if DD < 0 else DD - (NOR * WKR)
            # ------------------------------------ TERMINAL SETTING ------------------------------------ 

            # ------------------------------------ 選擇機台start ------------------------------------
            available_machine = None
            corresponding_time = None
            earliest_available = float('inf')
            best_fitness = float('inf')
            for cadiate_machine, t in zip(wp.processes[event.process_index]['machine_options'], wp.processes[event.process_index]['process_time']):
                # ------------------------------------ TERMINAL SETTING ------------------------------------ 
                PT = t # 製程執行時間
                NIQ = len(self.machine_queues[cadiate_machine]) # 機台佇列中的製程數量
                WIQ = sum([self.machine_queues[cadiate_machine][i][0] for i in range(NIQ)]) # 機台佇列中的製程所需的總處理時間
                MWT = self.machine_status[cadiate_machine] - event.time # 表示等待機台再次空閒的時間 (目前加工的製程結束時間 - CT)
                APTQ = WIQ / NIQ if NIQ > 0 else 0 # 機台佇列中的製程的平均加工時間 (WIQ/NIQ)
                # ------------------------------------ TERMINAL SETTING ------------------------------------ 
                if self.routing_rule is not None:
                    fitness = self.routing_rule(CT, PT, NPT, TIM, OWT, NOR, WKR, TIS, NIQ, WIQ, MWT, APTQ, DD, SL)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        available_machine = cadiate_machine
                        corresponding_time = t
                else:
                    if t < earliest_available:
                        earliest_available = t
                        available_machine = cadiate_machine
                        corresponding_time = t

            if available_machine is None or corresponding_time is None:
                # 理論上不應該發生，但以防萬一
                something_cool.double_border_my_word(
                    "[ERROR]:  MACHINE ASSIGNMENT ERROR",
                    f"Time {event.time:.2f}: No available machine for {wp} process {event.process_index}")
                sys.exit(1)
            chosen_machine = available_machine      # 選擇的機台編號
            event.machine_id = chosen_machine       # 設定機台 ID
            event.workpiece.arrival_machine_time = event.time  # 設定工件抵達機台時間
            duration = corresponding_time           # 選擇的機台加工時間
            # ------------------------------------ 選擇機台end ------------------------------------


            # ------------------------------------ 加入目標機台佇列start ------------------------------------
            if self.machine_status[chosen_machine] > event.time:
                line1 = f"Machine {chosen_machine} busy until {self.machine_status[chosen_machine]:.2f}."
                line2 = f"Adding {wp} to waiting queue with processing time {duration:.4f}."
                # something_cool.border_my_word(line1, line2)
            else :
                line2 = f"> Adding {wp} to machine {chosen_machine} with processing time {duration:.4f}."
                # log(line2)
            
            # 將「事件」與其「對應加工時間」加入對應的 waiting_queue
            self.machine_queues[chosen_machine].append((duration, event))

            new_event = Event(event.time, 'machine_check', workpiece=wp, process_index=event.process_index)
            new_event.machine_id = chosen_machine  # 設定機台 ID
            # 排程機台確認 waiting queue 事件
            self.schedule_event(new_event)
            # ------------------------------------ 加入目標機台佇列end ------------------------------------
            return chosen_machine
        
        elif event.event_type == 'machine_check':   # 有製程的加入/離開時，檢查機台狀態
            if event.machine_id is None:
                # 理論上不應該發生，但以防萬一
                something_cool.double_border_my_word(
                    f"[ERROR]: MACHINE CHECK ERROR",
                    f"Time {event.time:.2f}: No machine ID for {wp}")
                sys.exit(1)

            # 查看 machine 現在的狀態
            machine_id = event.machine_id
            machine_status = self.machine_status[machine_id]
            if machine_status <= event.time and self.machine_queues[machine_id] != []:
                # ------------------------------------- cutpoint -------------------------------------
                if (wp.wp_id, event.process_index) in self.cutpoints:
                    self.last_event = event
                    filename = f"./factory_snapshots/wp{wp.wp_id}_proc{event.process_index}_seed{self.random_seed}_rule{self.rule_name}_2.pkl"
                    with open(filename, "wb") as f:
                        pickle.dump(self, f)
                # ----------------------------------------------------------------------------------

                # ------------------------------------ 選擇製程start ------------------------------------
                best_fitness = float('inf')
                best_index = None
                machine_queue_index = 0

                # 列出機台的等待序列
                for (event_duration, waiting_event) in self.machine_queues[machine_id]:
                    waiting_wp = waiting_event.workpiece
                    # ------------------------------------ TERMINAL SETTING ------------------------------------ 
                    CT = event.time  # 當前時間
                    PT = event_duration # 製程執行時間
                    NPT = 0 if waiting_event.process_index+1 == len(waiting_wp.processes) else statistics.median(waiting_wp.processes[waiting_event.process_index+1]['process_time'])
                    TIM = waiting_wp.arrival_machine_time
                    OWT = CT - TIM
                    #-
                    NOR = len(waiting_wp.processes) - waiting_event.process_index - 1
                    WKR = 0 if waiting_event.process_index+1 >= len(waiting_wp.processes) else statistics.median([pt for proc in waiting_wp.processes[waiting_event.process_index+1:] for pt in proc['process_time']])
                    # NOR*WKR 為剩下製程的中位數預期時間
                    TIS = waiting_wp.arrival_time
                    #-
                    NIQ = len(self.machine_queues[machine_id]) # 機台佇列中的製程數量
                    WIQ = sum([self.machine_queues[machine_id][i][0] for i in range(NIQ)]) # 機台佇列中的製程所需的總處理時間
                    MWT = self.machine_status[machine_id] - CT # 表示等待機台再次空閒的時間 (目前加工的製程結束時間 - CT)
                    APTQ = WIQ / NIQ if NIQ > 0 else 0 # 機台佇列中的製程的平均加工時間 (WIQ/NIQ)
                    #-
                    DD = waiting_wp.due_date - waiting_event.time
                    SL = 0 if DD < 0 else DD - (NOR * WKR)
                    # ------------------------------------ TERMINAL SETTING ------------------------------------
                    
                    if self.sequencing_rule is not None:
                        fitness = self.sequencing_rule(CT, PT, NPT, TIM, OWT, NOR, WKR, TIS, NIQ, WIQ, MWT, APTQ, DD, SL)
                        if fitness < best_fitness:
                            best_fitness = fitness
                            best_index = machine_queue_index

                    machine_queue_index += 1
                
                next_duration = 0
                next_event = 0 
                # # 簡單的做法：從 waiting queue 中選擇最短的製程
                if self.sequencing_rule is None:
                    waiting = self.machine_queues[machine_id]
                    waiting.sort(key=operator.itemgetter(0))
                    best_index = 0
                if best_index is not None:
                    best_tuple = self.machine_queues[machine_id].pop(best_index)
                    (next_duration,next_event) = best_tuple
                else:
                    something_cool.double_border_my_word(
                        "[ERROR]: MACHINE CHECK ERROR",
                        f"Time {event.time:.2f}: No best event found for {wp}")
                    sys.exit(1)
                # ------------------------------------ 選擇製程end ------------------------------------

                next_event.time = event.time + next_duration
                next_event.event_type = 'end_process'
                self.schedule_event(next_event)
                self.machine_status[machine_id] = event.time + next_duration

                return best_index

        elif event.event_type == 'end_process':     # 製程結束加工
            # log(f"Time {event.time:.2f}: {wp} ends process {event.process_index} on machine {event.machine_id}")
            
            # key = (wp.wp_id, event.process_index)
            # if key in self._ongoing_tasks:
            #     self._ongoing_tasks[key]['end_time'] = event.time
            #     del self._ongoing_tasks[key]

            # 換到下一個製程
            wp.current_process += 1
            # 加入剛完成的製程的下一個製程事件
            if wp.current_process < len(wp.processes):
                new_event = Event(event.time, 'assign_process', workpiece=wp, process_index=wp.current_process)
                self.schedule_event(new_event)
                del new_event
            else:
                # log(f"Time {event.time:.2f}: {wp} completed all processes")
                self.completed_workpieces.append(wp)
            # 再確認 mahcine queue 裡面有沒有製程要執行
            new_event = Event(event.time, 'machine_check', workpiece=wp, process_index=wp.current_process)
            new_event.machine_id = event.machine_id  # 設定機台 ID
            self.schedule_event(new_event)
                
        else: 
            # 未知事件類型
            something_cool.double_border_my_word(
                "[ERROR]: UNKNOWN EVENT TYPE",
                f"Time {event.time:.2f}: Unknown event type {event.event_type} for {wp}")
            sys.exit(1)

# ---------------------------
# 產生不同類型的工廠
# ---------------------------
def createFactoryReference():
    """
    Generates a reference for the factory.
    Returns:
        list: A list containing the reference scheduling records for the factory.
    """
    pset = create_primitive_set()
    toolbox = setup_toolbox(pset)
    routing_str = "PT"
    routing_tree = gp.PrimitiveTree.from_string(routing_str, pset)
    routing_rule = toolbox.compile(expr=routing_tree)
    sequencing_str = "PT"
    sequencing_tree = gp.PrimitiveTree.from_string(sequencing_str, pset)
    sequencing_rule = toolbox.compile(expr=sequencing_tree)
    for seed in config.SNAPSHOT_RANDSEED:
       simulate_PC(routing_rule, sequencing_rule, random_seed=seed, rule_name="PT-PT")

    routing_str = "MWT"
    routing_tree = gp.PrimitiveTree.from_string(routing_str, pset)
    routing_rule = toolbox.compile(expr=routing_tree)
    sequencing_str = "TIS"
    sequencing_tree = gp.PrimitiveTree.from_string(sequencing_str, pset)
    sequencing_rule = toolbox.compile(expr=sequencing_tree)
    for seed in config.SNAPSHOT_RANDSEED:
       simulate_PC(routing_rule, sequencing_rule, random_seed=seed, rule_name="MWT-TIS")
    

# ---------------------------
# 產生 cutpoints 關鍵點
# ---------------------------
def generate_cutpoints(warmup, total, num_cutpoints, workpieces, rng):
    """
    隨機選出 num_cutpoints 個 cutpoint，每個 cutpoint是 (wp_id, process_index)
    """
    selected_wp_ids = rng.sample(range(warmup, total), num_cutpoints)
    print(f"Selected workpieces for cutpoints: {selected_wp_ids}")
    cutpoints = []
    for wp_id in selected_wp_ids:
        num_proc = len(workpieces[wp_id].processes)
        proc_idx = rng.randint(0, num_proc - 1)
        cutpoints.append((wp_id, proc_idx))

        something_cool.border_my_word(
            f"Cutpoint generated for Workpiece {wp_id}",
            f"Process Index: {proc_idx} (Total Processes: {num_proc})"
        )
    return set(cutpoints)

# ---------------------------
# 評估 cutpoints
# ---------------------------
def evaluate_cutpoint_decision(pkl_path, individual, pset):
    """
    載入 cutpoint snapshot，將個體的 rule 放回工廠，並回傳該狀態下的決策結果。
    individual: [routing_tree, sequencing_tree]
    pset: primitive set
    回傳: (chosen_machine, other_info)
    """
    with open(pkl_path, "rb") as f:
        factory = pickle.load(f)

    routing_rule = gp.compile(expr=individual[0], pset=pset)
    sequencing_rule = gp.compile(expr=individual[1], pset=pset)

    factory.routing_rule = routing_rule
    factory.sequencing_rule = sequencing_rule

    event = getattr(factory, "last_event", None)
    if event is None:
        raise RuntimeError("No event found in snapshot!")
    result = factory.process_event(event)
    return result

# ---------------------------
# 隨機產生工件
# ---------------------------
def generate_random_workpieces(count, min_processes=config.PROCESSES_RANGE[0], max_processes=config.PROCESSES_RANGE[1], machine_count=config.MACHINE_NUM):
    if (min_processes > max_processes) or (min_processes <= 0):
        something_cool.double_border_my_word(
            f"[ERROR]: process number range error [{min_processes}~{max_processes}]")
        sys.exit(1)
    min_machine_flex = config.FLEXIBLE_RANGE[0]
    max_machine_flex = config.FLEXIBLE_RANGE[1]
    if (min_machine_flex > max_machine_flex) or (min_machine_flex <= 0) or (min_machine_flex > machine_count)or (max_machine_flex > machine_count):
        something_cool.double_border_my_word(
            "[ERROR]: machine flexibility range error",
            f"machine flexibility range [{min_machine_flex}~{max_machine_flex}], machine count {machine_count}"
        )
        sys.exit(1)

    workpieces = []
    for i in range(count):
        num_processes = SIMULATION_RAND_PC.randint(min_processes, max_processes)
        processes = []
        for _ in range(num_processes):
            option_count = SIMULATION_RAND_PC.randint(min_machine_flex, max_machine_flex) 
            options = SIMULATION_RAND_PC.sample(range(1, machine_count+1), option_count)

            # ------------------------------------ 機台加工時間start ------------------------------------
            process_time_list = []
            for _ in range(option_count):
                process_time = SIMULATION_RAND_PC.uniform(config.PROCESSING_TIME_LOWER, config.PROCESSING_TIME_UPPER)
                process_time_list.append(process_time)
            # ------------------------------------ 機台加工時間end ------------------------------------

            processes.append({'machine_options': options, 'process_time': process_time_list})
        workpiece = Workpiece(i, processes)
        workpieces.append(workpiece)
    return workpieces

# ---------------------------
# 主程式
# ---------------------------
def simulate_PC(routing_rule=None, sequencing_rule=None, random_seed=None, rule_name=""):
    SIMULATION_RAND_PC.seed(random_seed)
    machine_count = config.MACHINE_NUM
    workpiece_count = config.WORKPIECE_NUM
    utilization_rate = config.UTILIZATION_RATE
    warmup_count = config.WARM_UP

    if warmup_count > workpiece_count:
        something_cool.double_border_my_word(
            "[ERROR]: WARM UP ERROR",
            f"WARM_UP {warmup_count} cannot be greater than WORKPIECE_NUM {workpiece_count}")
        sys.exit(1)
    
    workpieces = generate_random_workpieces(workpiece_count)
    cutpoints = generate_cutpoints(warmup_count, workpiece_count, config.NUM_OF_CUTPOINTS, workpieces, CUT_POINT_RNG)
    factory = Factory(machine_count, workpiece_count, utilization_rate, warmup_count, routing_rule, sequencing_rule, cutpoints=cutpoints,rule_name=rule_name,random_seed=random_seed)
    avg_job_length = (config.get_mean_processing_time()*(config.PROCESSES_RANGE[0]+config.PROCESSES_RANGE[1])/2)
    interarrival_rate = avg_job_length / (config.MACHINE_NUM * config.UTILIZATION_RATE)
    arrival_time = 0.0
    for wp in workpieces:
        arrival_time += SIMULATION_RAND_PC.expovariate( 1 / interarrival_rate )
        event = Event(arrival_time, 'arrival', workpiece=wp)
        factory.schedule_event(event)
    simulation_end_time = config.SIMULATION_END_TIME
    total_makespan = factory.run(simulation_end_time)
    
def main():
    """
    主程式入口，清空原本的 pkl 檔案，然後建立工廠參考。
    """
    # 先清空原本的 pkl 檔案
    if os.path.exists("factory_snapshots"):
        for file in os.listdir("factory_snapshots"):
            file_path = os.path.join("factory_snapshots", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs("factory_snapshots")

    # 5個時間點 
    # 2種dispatcing rule (SPT,FIFO)
    # 10間工廠(randseed) 
    createFactoryReference()

if __name__ == "__main__":
    main()