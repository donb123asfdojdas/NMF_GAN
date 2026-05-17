import numpy as np
import random
import matplotlib.pyplot as plt

# 参数设置
GRID_SIZE = 10  # 网格大小（10x10）
NUM_UAVS = 4    # 无人机数量
NUM_SIGNALS = 6 # 信号数量
ALPHA = 0.7     # 信号强度的权重
BETA = 0.3      # 覆盖信号数量的权重
MIN_REWARD = -100  # 未覆盖信号的最小奖励值
MAX_ITER = 50   # 最大迭代次数
POPULATION_SIZE = 20  # 种群大小
MUTATION_PROB = 0.1  # 变异概率
INITIAL_POSITIONS = [34, 37, 64, 67]  # 固定的无人机初始位置

# 随机生成信号位置（网格编号）
signal_positions = random.sample(range(1, GRID_SIZE**2 + 1), NUM_SIGNALS)

# 将网格编号转换为坐标
def grid_to_coord(grid_id):
    x = (grid_id - 1) % GRID_SIZE + 1
    y = (grid_id - 1) // GRID_SIZE + 1
    return np.array([x, y])

# 计算无人机到信号的距离
def compute_distance(uav_pos, signal_pos):
    return np.linalg.norm(grid_to_coord(uav_pos) - grid_to_coord(signal_pos))

# 适应度函数
def fitness_function(positions):
    signal_strength = 0
    coverage_count = 0
    uncovered_signals = set(signal_positions)

    for signal in signal_positions:
        covered = False
        for uav in positions:
            distance = compute_distance(uav, signal)
            if distance > 0:  # 防止除零
                signal_strength += 1 / (distance**2)
            if distance <= 3.5:  #如果某信号被该无人机被接收到
                covered = True                                                                         
                uncovered_signals.discard(signal)
        if covered:
            coverage_count += 1

    # 如果有信号未被覆盖，直接返回最小奖励值
    if uncovered_signals:
        return MIN_REWARD

    return ALPHA * signal_strength + BETA * coverage_count

# 绘制网格图
# 绘制网格图
def draw_grid(uav_positions, iteration, reward):
    plt.clf()
    plt.title(f"Iteration {iteration}, Reward: {reward:.2f}")
    plt.xlim(0.5, GRID_SIZE + 0.5)
    plt.ylim(0.5, GRID_SIZE + 0.5)

    # 绘制网格线
    for i in range(1, GRID_SIZE + 1):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0)  # 垂直网格线
        plt.axhline(y=i, color='gray', linestyle='--', linewidth=0)  # 水平网格线

    # 绘制信号位置
    for signal in signal_positions:
        coord = grid_to_coord(signal)
        plt.scatter(*coord, c='red', s=100, label='Signal' if signal == signal_positions[0] else "")

    # 绘制无人机位置
    for idx, uav in enumerate(uav_positions):
        coord = grid_to_coord(uav)
        plt.scatter(*coord, c='blue', marker='^', s=100, label=f'UAV {idx+1}' if idx == 0 else "")
        plt.text(coord[0], coord[1] + 0.3, f'UAV {idx+1}', fontsize=8, ha='center')

    # 设置网格和标签
    plt.xticks(range(1, GRID_SIZE + 1))
    plt.yticks(range(1, GRID_SIZE + 1))
    plt.grid(False)  # 禁用默认网格，使用自定义网格线
    plt.legend()
    plt.pause(0.5)


# 初始化种群（以固定位置为基础）
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = INITIAL_POSITIONS.copy()
        for i in range(NUM_UAVS):
            if random.random() < MUTATION_PROB:  # 初始变异
                individual[i] = random.randint(1, GRID_SIZE**2)
        population.append(individual)
    return population

# 变异操作
def mutate(individual):
    for i in range(NUM_UAVS):
        if random.random() < MUTATION_PROB:
            individual[i] = random.randint(1, GRID_SIZE**2)
    return individual

# 遗传算法主程序
def genetic_algorithm():
    population = initialize_population()
    best_solution = INITIAL_POSITIONS.copy()
    best_fitness = fitness_function(best_solution)

    plt.figure(figsize=(8, 8))
    for iteration in range(MAX_ITER):
        fitness_values = [fitness_function(individual) for individual in population]

        # 更新最优解
        max_fitness_idx = np.argmax(fitness_values)
        if fitness_values[max_fitness_idx] > best_fitness:
            best_fitness = fitness_values[max_fitness_idx]
            best_solution = population[max_fitness_idx]

        # 动态绘制
        draw_grid(best_solution, iteration + 1, best_fitness)

        # 选择和生成新种群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(population, weights=fitness_values, k=2)
            crossover_point = random.randint(1, NUM_UAVS - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            # 变异子代
            child1 = mutate(child1)
            child2 = mutate(child2)

            if len(set(child1)) == NUM_UAVS and all(1 <= x <= GRID_SIZE**2 for x in child1):
                new_population.append(child1)
            if len(set(child2)) == NUM_UAVS and all(1 <= x <= GRID_SIZE**2 for x in child2):
                new_population.append(child2)

        population = new_population

    plt.show()
    return best_solution, best_fitness

# 执行算法
best_positions, best_fitness = genetic_algorithm()
print("最佳无人机位置:", [grid_to_coord(pos) for pos in best_positions])
print("最高奖励值:", best_fitness)
