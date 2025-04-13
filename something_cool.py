def border_my_word(*lines):
    """
    將傳入的文字行以框線包住，若只傳入一行也能正常處理。

    用法:
      border_my_word(line1, line2)   # 兩行文字
      border_my_word(line1)          # 一行文字
    """
    if not lines:
        return
    # 為每行加入一個前置空格
    formatted_lines = [f" {line} " for line in lines]
    max_width = max(len(l) for l in formatted_lines)
    border = "+" + "-" * max_width + "+"
    print(border)
    for line in formatted_lines:
        print("|" + line.ljust(max_width) + "|")
    print(border)

def double_border_my_word(*lines):
    if not lines:
        return
    # 為每行加入一個前置空格
    formatted_lines = [f" {line} " for line in lines]
    max_width = max(len(l) for l in formatted_lines)
    upper_border = "╔" + "═" * max_width + "╗"
    lower_border = "╚" + "═" * max_width + "╝"
    print(upper_border)
    for line in formatted_lines:
        print("║" + line.ljust(max_width) + "║")
    print(lower_border)

if __name__ == "__main__":
    # 範例使用：
    chosen_machine = 3
    machine_status = {3: 12.34}  # 假設狀態
    wp = "Workpiece_1"         # 假設工件
    event_process_index = 2
    priority = 0.6789

    line1 = f"> Machine {chosen_machine} busy until {machine_status[chosen_machine]:.2f}."
    line2 = f"> Adding {wp}'s process {event_process_index} to waiting queue with priority {priority:.4f}."

    # 呼叫兩行版本
    border_my_word(line1, line2)
    # 或只傳一行：
    double_border_my_word(line1)