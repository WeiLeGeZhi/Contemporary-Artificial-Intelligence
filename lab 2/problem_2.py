from queue import PriorityQueue

# 定义节点数量、节点间路径的数量和需要返回的行数
N, M, K = map(int, input().split())

# 构建图的邻接表
graph = [[] for _ in range(N)]
for _ in range(M):
    start, end, cost = map(int, input().split())
    graph[start - 1].append((end - 1, cost))

# 定义启发式函数
def heuristic(node):
    return (N - node)/N

# 执行A*算法
min_heap = PriorityQueue()
min_heap.put((0 + heuristic(0), 0, 0))
k_paths = []

while not min_heap.empty() and len(k_paths) < K:
    total_cost_plus_Heuristic, total_cost, current_node = min_heap.get()

    if current_node == N - 1:
        k_paths.append(total_cost)

    for neighbor, cost in graph[current_node]:
        new_cost = total_cost + cost
        new_cost_plus_heuristic = new_cost + heuristic(neighbor)
        min_heap.put((new_cost_plus_heuristic, new_cost, neighbor))


if len(k_paths) < K:
    k_paths = k_paths+([-1]*(K-len(k_paths)))
# 输出前K条路径的长度
for path_length in k_paths:
    print(path_length)