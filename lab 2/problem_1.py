import sys

from queue import PriorityQueue

# 定义目标状态
goal_state = (1, 3, 5, 7, 0, 2, 6, 8, 4)

# 定义启发式函数：不在位数字的个数
def heuristic(state):
    sum = 0
    for i in range(9):
        if state[i] != 0:
            if state[i] != goal_state[i]:
                sum+=1
    return sum

# 定义移动操作
def move(state, direction):
    state_list = list(state)
    empty_index = state_list.index(0)
    if direction == 'left' and empty_index % 3 > 0:
        state_list[empty_index], state_list[empty_index - 1] = state_list[empty_index - 1], state_list[empty_index]
    elif direction == 'right' and empty_index % 3 < 2:
        state_list[empty_index], state_list[empty_index + 1] = state_list[empty_index + 1], state_list[empty_index]
    elif direction == 'up' and empty_index >= 3:
        state_list[empty_index], state_list[empty_index - 3] = state_list[empty_index - 3], state_list[empty_index]
    elif direction == 'down' and empty_index < 6:
        state_list[empty_index], state_list[empty_index + 3] = state_list[empty_index + 3], state_list[empty_index]
    return tuple(state_list)

# 定义A*算法
def astar(initial_state):
    open_list = PriorityQueue()
    open_list.put((heuristic(initial_state), initial_state, 0))
    closed_set = set()

    while not open_list.empty():
        _, current_state, g = open_list.get()
        if current_state == goal_state:
            return g

        if current_state in closed_set:
            continue

        closed_set.add(current_state)

        for direction in ['left', 'right', 'up', 'down']:
            new_state = move(current_state, direction)
            open_list.put((heuristic(new_state) + g + 1, new_state, g + 1))

    return -1  # 无解情况

# 解析命令行参数
if len(sys.argv) != 2:
    print("请提供初始状态作为命令行参数。")
    sys.exit(1)

initial_state = tuple(map(int, sys.argv[1]))

# 解决问题并输出步数
steps = astar(initial_state)
if steps >= 0:
    print(steps)
else:
    print("无解")
