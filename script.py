import networkx as nx
import matplotlib.pyplot as plt
import pygad
import numpy as np

# Define the graph and its colors
graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3)])
colors = ["red", "green", "blue", "yellow"]


# Define the fitness function
def fitness_func(ga_instance, solution, solution_idx):
    conflicts = 0
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph.has_edge(i, j) and solution[i] == solution[j]:
                conflicts += 1
    fitness = 1 / (conflicts + 1)
    return fitness


# Create an instance of the GA class
ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=len(graph),
    gene_space=list(range(len(colors))),
    mutation_percent_genes=10,
    crossover_probability=0.8,
    mutation_probability=0.2,
    parent_selection_type="rank",
    crossover_type="single_point",
    mutation_type="random",
)

# Run the genetic algorithm
ga_instance.run()

# Print the best solution
print("Best solution:", ga_instance.best_solution())

# Convert the best solution to integers
best_solution_int = np.array(ga_instance.best_solution()[0], dtype=int)

# Visualize the best solution
pos = nx.spring_layout(graph)
nx.draw(
    graph, pos, node_color=[colors[best_solution_int[i]] for i in range(len(graph))]
)
plt.savefig("graph_coloring.png")
plt.show()
