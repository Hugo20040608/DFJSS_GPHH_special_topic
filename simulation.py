import heapq
import random
import matplotlib.pyplot as plt

# ---------------------------
# Event 類別
# ---------------------------
class Event:
    def __init__(self, time, event_type, workpiece=None, process_index=None):
        """
        time: 事件發生時間
        event_type: 事件類型，如 'arrival', 'start_process', 'end_process'
        workpiece: 相關工件 (Workpiece 物件)
        process_index: 工件當前製程索引
        """
        self.time = time
        self.event_type = event_type
        self.workpiece = workpiece
        self.process_index = process_index

    def __lt__(self, other):
        # 用於 heapq，比較事件時間
        return self.time < other.time

# ---------------------------
# Workpiece 類別
# ---------------------------
class Workpiece:
    def __init__(self, wp_id, processes):
        """
        wp_id: 工件識別編號
        processes: 製程列表，每個製程為一個字典，例如 {'machine_options': [1,2,3]}
        """
        self.wp_id = wp_id
        self.processes = processes
        self.current_process = 0  # 當前製程索引
        self.arrival_time = None  # 工件抵達時間

    def __str__(self):
        return f"Workpiece {self.wp_id} (Process {self.current_process}/{len(self.processes)})"

# ---------------------------
# Factory 類別 (包含機台狀態管理)
# ---------------------------
class Factory:
    def __init__(self, machine_count, workpiece_count, utilization_rate, warmup_count):
        """
        machine_count: 機台數量
        workpiece_count: 工件數量
        utilization_rate: 機台利用率設定（影響工件進廠時間）
        warmup_count: 暖場工作數量
        """
        self.machine_count = machine_count
        self.workpiece_count = workpiece_count
        self.utilization_rate = utilization_rate
        self.warmup_count = warmup_count
        self.current_time = 0.0
        self.event_queue = []   # 儲存所有事件的 priority queue
        self.completed_workpieces = []
        self.schedule_records = []  # 記錄每個工件各製程的排程記錄
        self._ongoing_tasks = {}    # 暫存進行中的任務 (key: (wp_id, process_index))
        # 新增：機台狀態管理，記錄每台機台最早可用時間
        self.machine_status = {machine_id: 0.0 for machine_id in range(1, machine_count+1)}

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def run(self, simulation_end_time):
        """
        以中央時間管理方式執行模擬，直到 simulation_end_time 或事件耗盡
        """
        while self.event_queue and self.current_time < simulation_end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            self.process_event(event)

    def process_event(self, event):
        wp = event.workpiece
        if event.event_type == 'arrival':
            wp.arrival_time = event.time
            print(f"Time {event.time:.2f}: {wp} arrived")
            # 排程第一個製程開始
            new_event = Event(event.time, 'start_process', workpiece=wp, process_index=0)
            self.schedule_event(new_event)
        elif event.event_type == 'start_process':
            print(f"Time {event.time:.2f}: {wp} starts process {event.process_index}")
            process_options = wp.processes[event.process_index]['machine_options']
            
            # 從可選機台中挑選一台：選取當前空閒的機台，若都忙碌則選最早空閒的
            available_machine = None
            earliest_available = float('inf')
            for m in process_options:
                # 若此機台不在狀態字典中，則略過（理論上不會發生，因為 machine_status 已包含所有機台）
                if m not in self.machine_status:
                    continue
                if self.machine_status[m] <= event.time:
                    available_machine = m
                    break
                else:
                    if self.machine_status[m] < earliest_available:
                        earliest_available = self.machine_status[m]
                        available_machine = m
            if available_machine is None:
                # 理論上不應該發生，但以防萬一
                available_machine = process_options[0]
            chosen_machine = available_machine

            # 如果機台目前還在忙碌，延後排程
            if event.time < self.machine_status[chosen_machine]:
                new_time = self.machine_status[chosen_machine]
                print(f"Machine {chosen_machine} busy until {new_time:.2f}. Delaying {wp}'s process {event.process_index}.")
                new_event = Event(new_time, 'start_process', workpiece=wp, process_index=event.process_index)
                self.schedule_event(new_event)
                return

            # 模擬決策：這裡以隨機加工時間表示，實際上可依 GP 個體決策
            duration = random.uniform(1, 10)
            # 更新該機台狀態：設定忙碌至 (event.time + duration)
            self.machine_status[chosen_machine] = event.time + duration

            # 記錄排程資料
            record = {
                'wp_id': wp.wp_id,
                'process_index': event.process_index,
                'arrival_time': wp.arrival_time,
                'start_time': event.time,
                'machine_id': chosen_machine,
                'end_time': None
            }
            self.schedule_records.append(record)
            self._ongoing_tasks[(wp.wp_id, event.process_index)] = record

            new_event = Event(event.time + duration, 'end_process', workpiece=wp, process_index=event.process_index)
            self.schedule_event(new_event)
        elif event.event_type == 'end_process':
            print(f"Time {event.time:.2f}: {wp} ends process {event.process_index}")
            key = (wp.wp_id, event.process_index)
            if key in self._ongoing_tasks:
                self._ongoing_tasks[key]['end_time'] = event.time
                del self._ongoing_tasks[key]
            wp.current_process += 1
            if wp.current_process < len(wp.processes):
                new_event = Event(event.time, 'start_process', workpiece=wp, process_index=wp.current_process)
                self.schedule_event(new_event)
            else:
                print(f"Time {event.time:.2f}: {wp} completed all processes")
                self.completed_workpieces.append(wp)
        else:
            print(f"Time {event.time:.2f}: Unknown event type {event.event_type}")

# ---------------------------
# 隨機產生工件
# ---------------------------
def generate_random_workpieces(count, max_processes=5, min_processes=2, machine_count=5):
    workpieces = []
    for i in range(count):
        num_processes = random.randint(min_processes, max_processes)
        processes = []
        for _ in range(num_processes):
            # 確保隨機產生的機台選項在 1 ~ machine_count 之間
            option_count = random.randint(1, machine_count)
            options = random.sample(range(1, machine_count+1), option_count)
            processes.append({'machine_options': options})
        workpiece = Workpiece(i, processes)
        workpieces.append(workpiece)
    return workpieces


# ---------------------------
# 畫甘特圖 (以機台為 y 軸)
# ---------------------------
def plot_gantt_by_machine(schedule_records, machine_range=None):
    """
    根據 schedule_records 畫出以機台為 y 軸、時間為 x 軸的甘特圖，
    每個橫條代表一個工件在某個製程的排程，
    橫條上標示 "工件編號-製程編號"，並以不同顏色區分不同工件。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 取得所有機台編號（根據記錄）
    machine_ids = sorted({ rec['machine_id'] for rec in schedule_records if rec['end_time'] is not None })
    if machine_range is None:
        machine_range = machine_ids

    # 為不同工件分配顏色 (使用 tab20 colormap)
    unique_wp_ids = sorted({ rec['wp_id'] for rec in schedule_records })
    cmap = plt.get_cmap('tab20')
    wp_color = { wp: cmap(i % 20) for i, wp in enumerate(unique_wp_ids) }
    
    bar_height = 8
    gap = 2
    min_machine = min(machine_range)
    
    yticks = []
    yticklabels = []
    
    for rec in schedule_records:
        if rec['end_time'] is None:
            continue
        machine_id = rec['machine_id']
        start = rec['start_time']
        end = rec['end_time']
        duration = end - start
        
        # y 座標依機台編號決定
        y = (machine_id - min_machine) * (bar_height + gap)
        
        color = wp_color[rec['wp_id']]
        ax.broken_barh([(start, duration)], (y, bar_height), facecolors=color)
        ax.text(start + duration/2, y + bar_height/2, f"{rec['wp_id']}-{rec['process_index']}",
                color='white', weight='bold', ha='center', va='center', fontsize=8)
        
        if machine_id not in [int(lbl.split()[-1]) for lbl in yticklabels]:
            yticks.append(y + bar_height/2)
            yticklabels.append(f"Machine {machine_id}")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart by Machine")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True)
    plt.show()

# ---------------------------
# 主程式
# ---------------------------
if __name__ == "__main__":
    # 設定參數
    machine_count = 5
    workpiece_count = 10
    utilization_rate = 0.8   # 機台利用率（影響工件進廠時間）
    warmup_count = 2
    
    factory = Factory(machine_count, workpiece_count, utilization_rate, warmup_count)
    workpieces = generate_random_workpieces(workpiece_count)

    # 安排工件抵達事件 (這裡以隨機時間模擬)
    for wp in workpieces:
        arrival_time = random.uniform(0, 20)
        event = Event(arrival_time, 'arrival', workpiece=wp)
        factory.schedule_event(event)

    simulation_end_time = 100.0
    factory.run(simulation_end_time)

    # 列印每個工件的排程記錄 (抵達時間、開始/結束時間、機台編號)
    print("\nSchedule Records:")
    for rec in factory.schedule_records:
        if rec['end_time'] is not None:
            print(f"Workpiece {rec['wp_id']} - Process {rec['process_index']}: "
                  f"Arrival: {rec['arrival_time']:.2f}, Start: {rec['start_time']:.2f}, "
                  f"End: {rec['end_time']:.2f}, Machine: {rec['machine_id']}")

    # 畫出甘特圖（以機台為 y 軸）
    plot_gantt_by_machine(factory.schedule_records)
