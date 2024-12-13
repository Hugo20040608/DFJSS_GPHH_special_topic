import time
import random
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
import config
import os

# this simulation has been validated and works
def simulation(number_machines, number_jobs, warm_up, func, due_date_tightness, utilization, random_seed=None, decision_situtation=None, missing_operation=None):

    # Initialize Lists
    schedule, jobs, jobs_var, jobs_finished = [], [], [], []

    # Initialize global clock
    global_clock = 0

    # Initialize random seed if applicable
    if random_seed != None:
        random.seed(a=random_seed)

    # Set number of operations equal to number of machines (full shop mode)
    number_operations = number_machines
    # Calculate the inter-arrival time (based on utilization and mean processing time)
    mean_processing_time = config.MEAN_PROCESSING_TIME # mean processing time (!!!!! may need to change !!!!!)
    
    if missing_operation==True:
        mean_number_operations = (number_machines+2) / 2
    else:
        mean_number_operations = number_machines

    # 工作到達間隔時間（inter-arrival time）的計算公式
    interarrival_time = (mean_processing_time*mean_number_operations)/(number_machines*utilization)

    # Initialize global parameters
    SPT, TRNO = 0, 0

# ------------------------------------ Class Job ------------------------------------ #

    class Job():
        def __init__(self):
            self.start = 0
            self.end = 0
            self.clock = 0
            self.operations = []
            self.number_operations = 0
            self.RPT = 0
            self.RNO = 0
            self.DD = 0
            self.operation_to_release = int(0)
            self.next_operation = int(1)
            self.release_status = 'no'
            self.t_event = 0
            self.number = 0
            self.release_time = 0

        class Operation():
            def __init__(self):
                self.number = 0             # above
                self.start = 0              # above
                self.end = 0                # above
                self.clock = 0              # above
                self.PT = 0                 
                self.machine = int(999999)
                self.release_time = 0       # above

# ------------------------------------ Class Machine ------------------------------------ #

    class Machine():
        def __init__(self, id):
            self.id = id
            self.queue = {'Job':[], 'Operation':[], 'Priority':[]}
            self.job_to_release = []
            self.num_in_system = 0
            self.clock = 0.0
            self.t_depart = float('inf')
            self.t_event = 0
            self.status = 'Idle'
            self.current_job_finish = 0
            self.counter = 0

        def execute(self):
            # update priority
            self.update_priority()
            # select the waiting operation with the lowest priority value
            min_priority = min(self.queue["Priority"])
            index_job = self.queue["Priority"].index(min_priority)
            next_job = self.queue['Job'][index_job].number
            
            # 更新作業與操作的狀態
            self.queue['Operation'][index_job].start = self.clock
            self.queue['Operation'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].t_event = self.queue["Operation"][index_job].PT    # t_event：當前操作的處理時間
            self.queue['Job'][index_job].clock += self.queue["Operation"][index_job].PT     # clock：作業已消耗的總時間
            self.queue['Job'][index_job].RPT -= self.queue["Operation"][index_job].PT       # RPT：作業剩餘的處理時間，減去當前操作的處理時間
            self.queue['Job'][index_job].RNO -= 1       # 作業剩餘的操作數量，減少 1
            self.queue['Job'][index_job].end = self.clock + self.queue["Operation"][index_job].PT

            # 判斷作業的整體狀態
            # 如果是第一個操作
            if self.queue['Operation'][index_job].number == 0:
                self.queue['Job'][index_job].start = self.clock
            # 如果是最後一個操作
            if self.queue['Operation'][index_job].number == (self.queue['Job'][index_job].number_operations-1):
                self.queue['Job'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
                jobs_var.remove(self.queue['Job'][index_job])
                jobs_finished.append(self.queue['Job'][index_job])
                # 活動作業列表（jobs_var）移除，並加入完成作業列表（jobs_finished）
            
            # 更新機器的時鐘與事件
            self.t_event = self.queue["Operation"][index_job].PT
            self.clock += self.t_event
            self.current_job_finish = self.clock

            # set job status to 'release'
            self.queue['Job'][index_job].operation_to_release += 1      # operation_to_release：標記下一個需要釋放的操作
            self.queue['Job'][index_job].next_operation += 1            # next_operation：更新下一步操作的索引
            self.queue['Job'][index_job].release_status = 'yes'
            self.queue['Job'][index_job].clock = self.clock

            schedule.append({
                "machine": self.id,  # 機器的唯一標識
                "job": self.queue['Job'][index_job].number,  # 作業編號
                "operation": self.queue['Operation'][index_job].number,  # 操作編號
                "start": self.queue['Operation'][index_job].start,  # 開始時間
                "end": self.queue['Operation'][index_job].end  # 結束時間
            })

            # remove operation from queue
            del self.queue["Job"][index_job]
            del self.queue["Operation"][index_job]
            del self.queue["Priority"][index_job]

            # set status to 'running'  |  更新機器狀態，表示機器正在執行作業
            self.status = 'Running'


        def update_priority(self):
            PT_list = []
            for i in range(len(self.queue['Job'])):
                PT_list.append(self.queue['Operation'][i].PT)
            APTQ = mean(PT_list)
            NJQ = len(self.queue['Job'])

            for i in range(len(self.queue['Job'])):
                PT = self.queue['Operation'][i].PT
                RT = self.queue['Job'][i].release_time
                RPT = self.queue['Job'][i].RPT
                RNO = self.queue['Job'][i].RNO
                DD = self.queue['Job'][i].DD
                RTO = self.queue['Operation'][i].release_time
                CT = self.clock     # 當前時間
                SL = DD-(CT+RPT)    # 截止時間與完成剩餘所有操作的時間之間的差值（剩餘時間餘裕）
                WT = max(0, CT-RTO) # 等待時間
                next_operation_1 = self.queue['Job'][i].next_operation
                if next_operation_1 >= len(self.queue['Job'][i].operations):
                    PTN = 0         # Processing Time of Next operation 下一步操作的處理時間
                    WINQ = 0        # Work in Next Queue 下一個機器的隊列工作量總和
                else:
                    next_operation_1 = self.queue['Job'][i].operations[next_operation_1]
                    PTN = next_operation_1.PT
                    machine_next_operation = next_operation_1.machine
                    queue_next_operation = machines[machine_next_operation].queue
                    WINQ = sum(queue_next_operation['Operation'][i].PT for i in range(len(queue_next_operation['Job']))) \
                           + max(machines[machine_next_operation].clock - CT, 0)

                expected_waiting_time = 0
                next_operation_2 = self.queue['Job'][i].next_operation
                while next_operation_2 < len(self.queue['Job'][i].operations):
                    next_operation = self.queue['Job'][i].operations[next_operation_2]
                    machine_next_operation = next_operation.machine
                    queue_next_operation = machines[machine_next_operation].queue
                    expected_waiting_time += (sum(queue_next_operation['Operation'][i].PT for i in range(len(queue_next_operation['Job']))) \
                                              - max(machines[machine_next_operation].clock - CT, 0)) / 2
                    next_operation_2 += 1

                # 使用 func 計算動態優先級
                if func is None:
                    priority = (2*PT + WINQ + PTN)
                else:
                    priority = func(PT, RT, RPT, RNO, DD, RTO, PTN, SL, WT, APTQ, NJQ, WINQ, CT)
                self.queue["Priority"][i] = -priority  # 設定優先級，but why negative？ -> 因為選擇 min_priority

# ----------------------------------------------------------- Class Job Generator ----------------------------------------------------------- #

    class JobGenerator():
        def __init__(self):
            self.clock = 0.0
            self.number = 1

        def execute(self):
            job = Job()                     # 生成一個job
            job.release_time = self.clock   # 生成工作的時間
            allowed_values = list(range(0, number_machines)) # job可以執行的machine範圍
            total_processing_time = 0       # 所有operation所需要的時間
            # 隨機生成operation的數量
            if missing_operation == True:  # 如果允許缺少operation (目前沒有使用)
                job.number_operations = random.randint(2, number_machines) 
            else:
                job.number_operations = number_machines

            number_operations = job.number_operations # 用來做最後回傳的操作數量
            job.operations = [job.Operation() for _ in range(job.number_operations)] # 依據前面隨機產生的job.number_operations生成對應數量的operation
            for oper in job.operations:
                oper.PT = random.randint(5, 50)              # 隨機生成操作的處理時間
                oper.machine = random.choice(allowed_values) # 從允許的機器列表中選擇一台機器
                total_processing_time += oper.PT             # 累計該作業的總處理時間
                oper.number = job.operations.index(oper)     # 設定操作的編號（索引值）
                allowed_values.remove(oper.machine)          # 移除已分配的機器，避免重複分配
                # 避免同一作業的操作重複使用相同的機器，模擬更靈活的工藝要求（例如每道工序需要不同的機器來完成）。
                # 如果作業允許操作分配到相同機器，則可以移除此行。
            
            # due_date_tightness 定義在這：用於控制截止時間的緊迫程度。值越小，截止時間越緊迫
            job.DD = job.release_time + (due_date_tightness * total_processing_time)
            job.RPT = total_processing_time
            job.RNO = len(job.operations)
            job.number = self.number
            jobs.append(job)
            jobs_var.append(job)

            # 分配第一道操作到機器隊列
            number_of_released_operation = job.operation_to_release
            machine_to_release = job.operations[number_of_released_operation].machine
            machines[machine_to_release].queue['Job'].append(job)
            machines[machine_to_release].queue['Operation'].append(job.operations[number_of_released_operation])
            machines[machine_to_release].queue['Priority'].append(0)
            job.operations[number_of_released_operation].release_time = self.clock

            # 設置到達間隔並更新時鐘
            # 使用指數分佈生成作業的到達間隔時間
            # 公式背景：指數分佈常用於模擬隨機到達的事件， 1/interarrival_time 是到達率。
            interarrival_time_current = random.expovariate(1/interarrival_time)  
            self.clock += interarrival_time_current
            self.number +=1         # 為下一個生成的作業分配新的編號

            return total_processing_time, number_operations
        
# ------------------------------------------------------------------------------------------------------------------------------------------- #

    # generate machines
    machines = [Machine(mechine_id) for mechine_id in range(number_machines)]

    # generate Job generator
    job_generator = JobGenerator()

    # execute Job generator the first time to generate a first job
    processing_time, number_operations = job_generator.execute()
    # update global parameters
    TRNO += number_operations
    SPT += processing_time

    # start simulation  |  loop until stopping criterion is met
    while len(jobs_finished) < number_jobs:

        # check if there are operations to be released on each job
        for j in jobs_var:
            if j.clock <= global_clock and j.release_status == 'yes':
                number_of_released_operation = j.operation_to_release
                if number_of_released_operation <= (len(j.operations)-1):
                    machine_to_release = j.operations[number_of_released_operation].machine
                    machines[machine_to_release].queue['Job'].append(j)
                    machines[machine_to_release].queue['Operation'].append(j.operations[number_of_released_operation])
                    machines[machine_to_release].queue['Priority'].append(0)
                    j.release_status = 'no'
                    j.operations[number_of_released_operation].release_time = j.end

        # check if there is a job to be released on the job generator
        if job_generator.clock <= global_clock:
            processing_time, number_operations = job_generator.execute()
            # update global parameters
            TRNO += number_operations
            SPT += processing_time

        # check if there are jobs waiting in the queue on each machine
        for i in machines:
            if i.clock <= global_clock and len(i.queue["Job"]) != 0:
                i.execute()
                # update global parameters
                TRNO -= 1
                SPT -= i.t_event

        # check for next event on the three classes (jobs, machines, jobgenerator)
        t_next_event_list = []
        for m in machines:
            if m.clock > global_clock:
                t_next_event_list.append(m.clock)
        for j in jobs_var:
            if j.clock > global_clock:
                t_next_event_list.append(j.clock)
        if job_generator.clock > global_clock:
            t_next_event_list.append(job_generator.clock)

        # 找到下一個事件的時間，並更新全局時鐘（global_clock）
        if t_next_event_list != []:
            t_next_event = min(t_next_event_list)
        else:
            t_next_event=0
        global_clock = t_next_event     # 將全局時鐘更新為下一個最早事件的時間

        # set the machine times to the global time for those that are less than the global time
        for i in machines:
            if i.clock <= global_clock:
                i.clock = global_clock
        for j in jobs_var:
            if j.clock <= global_clock:
                j.clock = global_clock
    # simulation terminate
    
    # calculate performance measures
    makespan = max(record["end"] for record in schedule)
    mean_flowtime = mean([(j.end - j.release_time) for j in jobs_finished[warm_up:]])
    max_flowtime = max([(j.end - j.release_time) for j in jobs_finished[warm_up:]])
    # max_tardiness = max([max((j.end - j.DD), 0) for j in jobs_finished[warm_up:]])
    # mean_tardiness = mean([max((j.end - j.DD), 0) for j in jobs_finished[warm_up:]])
    # waiting_time = np.sum((j.end-j.start) for j in jobs_finished[warm_up:])

# ------------------------------------ Plot Gantt Chart ------------------------------------ #

    if __name__ == "__main__":
        cmap = colormaps["tab20"]
        colors = [cmap(i / (len(jobs_finished) + 10)) for i in range(len(jobs_finished) + 10)]
        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, record in enumerate(schedule):
            machine = record["machine"]
            job = record["job"]
            operation = record["operation"]
            start = record["start"]
            end = record["end"]
            color = colors[job % len(colors) - 1]
            ax.add_patch(mpatches.Rectangle((start, machine), end - start, 0.8, color=color, label=f"Job {job}"))
        max_time = max(record["end"] for record in schedule)
        min_time = min(record["start"] for record in schedule)
        ax.set_xlim(min_time, max_time)
        ax.set_yticks(range(number_machines+1))
        ax.set_yticklabels([f"Machine {i}" for i in range(number_machines+1)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_title("Machine Job Allocation")
        job_labels = [f"Job {job}" for job in range(1, len(jobs_finished) + 11)]
        sorted_labels = sorted(job_labels, key=lambda x: int(x.split()[1]))  
        handles = [mpatches.Patch(color=colors[i % len(colors)], label=label) for i, label in enumerate(sorted_labels)]
        ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1, 1),title="Jobs")

        path = "./results/"
        try:
            os.makedirs(path, exist_ok=True)
            print("Successfully created the directory %s " % path)
        except OSError as e:
            print(f"Creation of the directory {path} failed due to {e}")
        plt.savefig(path+f"gantt_chat_rand={random_seed:02}.png", bbox_inches="tight", dpi=300)  # bbox_inches="tight" 確保圖例不被裁切
        plt.close()

# ------------------------------------------------------------------------------------------------------------------------------------- #
    
    return mean_flowtime, makespan, max_flowtime

def rule(PT, RT, RPT, RNO, DD, RTO, PTN, SL, WT, APTQ, NJQ, WINQ, CT):
    return RT

if __name__ == "__main__":
    #Test the algorithm
    start = time.time()
    max_tardiness = []
    mean_tardiness = []
    mean_flowtime = []
    
    random_seed = config.RANDOM_SEEDS_FOR_SIMULATION
    for i in random_seed:
        current_mean_flowtime, current_mean_tardiness, current_max_tardiness = \
            simulation(number_machines=config.NUMBER_MACHINES, number_jobs=config.NUMBER_JOBS, warm_up=config.WARM_UP, func=rule, random_seed=i, due_date_tightness=config.DUE_DATE_TIGHTNESS, utilization=config.UTILIZATION, missing_operation=config.MISSING_OPERATION)
        mean_flowtime.append(current_mean_flowtime)
        mean_tardiness.append(current_mean_tardiness)
        max_tardiness.append(current_max_tardiness)
    end = time.time()

    # schedule.to_excel('schedule.xlsx')
    print(mean_flowtime)
    print(mean_tardiness)
    print(max_tardiness)

    print(f'Execution time simulation per replication: {(end - start)}')
    print(f'Mean flowtime: {mean(mean_flowtime)}')
    print(f'Mean Tardiness: {mean(mean_tardiness)}')
    print(f'Max tardiness: {max(max_tardiness)}')