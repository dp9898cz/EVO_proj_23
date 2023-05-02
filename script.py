import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pygad
import numpy as np
import random

GRAPH_SIZE = 60
colors_number = GRAPH_SIZE

def generate_graph(min_nodes, max_nodes):
    num_nodes = random.randint(min_nodes, max_nodes)
    G = nx.Graph()
    nodes = range(num_nodes)
    G.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if i < j and random.random() < 0.33:
                G.add_edge(i, j)
    return G

def generate_unique_color_names(length):
    css_colors = list(mcolors.CSS4_COLORS.keys())
    color_names = []
    while len(color_names) < length:
        color_name = random.choice(css_colors)
        if color_name not in color_names:
            color_names.append(color_name)
    return color_names

# Define the graph and its colors
#graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3)])
graph = generate_graph(GRAPH_SIZE, GRAPH_SIZE)
colors = generate_unique_color_names(GRAPH_SIZE)


# Define the fitness function
def fitness_func(ga_instance, solution, solution_idx):
    conflicts = get_conflicts_number(solution)
    fitness = 1 / (conflicts + 1)
    return fitness

def get_conflicts_number(solution):
    conflicts = 0
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph.has_edge(i, j) and solution[i] == solution[j]:
                conflicts += 1
    return conflicts

def get_unique_colors_number(solution):
    return len(set(solution))

solution_fitness = 1
best_individual = None

while (solution_fitness == 1):
    print(f"Starting GA with {colors_number} colors to find solution.")

    # Create an instance of the GA class
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=len(graph),
        gene_space=list(range(colors_number)),
        mutation_percent_genes=10,
        crossover_probability=0.8,
        mutation_probability=0.2,
        parent_selection_type="rank",
        crossover_type="single_point",
        mutation_type="random",
        stop_criteria=["reach_1"]
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    if (solution_fitness == 1):
        best_individual = solution
        ga_instance.plot_fitness(save_dir="fitness.png")
        plt.clf()
    colors_number -= 1



best_solution_int = np.array(best_individual, dtype=int)
print("Best solution: " + str(best_solution_int) + " Fitness: " + str(solution_fitness))
print("Conflicts: " + str(get_conflicts_number(solution)) + " Unique colors: " + str(get_unique_colors_number(solution)))

# Visualize the best solution
pos = nx.spring_layout(graph)
nx.draw(
    graph, pos, node_color=[colors[best_solution_int[i]] for i in range(len(graph))]
)
plt.savefig("graph_coloring.png")


