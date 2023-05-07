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
            if i < j and random.random() < 0.25:
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
graph = generate_graph(GRAPH_SIZE, GRAPH_SIZE)
colors = generate_unique_color_names(GRAPH_SIZE)


def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        random_split_point = np.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]
        offspring.append(parent1)
        idx += 1
    return np.array(offspring)


def crossover_with_neighbours(parents, offspring_size, ga_instance):
    parent1, parent2, parent3, parent4 = parents
    num_genes = len(parent1)
    offspring = np.empty(offspring_size)

    # Select two random parents for the single-point crossover
    parent_pair = random.sample(
        [
            (parent1, parent2),
            (parent1, parent3),
            (parent1, parent4),
            (parent2, parent3),
            (parent2, parent4),
            (parent3, parent4),
        ],
        1,
    )[0]
    parent1, parent2 = parent_pair
    parent1 = parent1.astype(int)
    parent2 = parent2.astype(int)

    # Select a random point in the chromosome
    crossover_point = random.randint(0, num_genes - 1)

    # Create a list of neighbouring nodes for each vertex
    neighbours = []
    for node in graph.nodes():
        neighbour_list = [int(n) for n in graph.neighbors(node)]
        neighbours.append(neighbour_list)

    # Iterate over the offspring and create each one
    for i in range(offspring_size[0]):
        new_chromosome = np.empty(num_genes)

        # Copy the first part from parent 1
        new_chromosome[:crossover_point] = parent1[:crossover_point]

        # Cross with neighbours
        node = parent1[crossover_point]
        neighbour_colors = set(parent2[j] for j in neighbours[node])
        possible_colors = set(range(len(ga_instance.population)))
        available_colors = possible_colors - neighbour_colors
        if len(available_colors) > 0:
            new_chromosome[crossover_point] = random.choice(list(available_colors))
        else:
            new_chromosome[crossover_point] = random.randint(
                0, len(range(len(ga_instance.population))) - 1
            )

        # Copy the second part from parent 2
        new_chromosome[crossover_point + 1 :] = parent2[crossover_point + 1 :]

        # Add the new chromosome to the offspring
        offspring[i, :] = new_chromosome

    return offspring


def mutate_no_conflict(offspring, ga_instance):
    # Iterate over each offspring in the population
    for i in range(len(offspring)):
        # Get the chromosome of the current offspring
        chromosome = offspring[i]

        # Iterate over each gene in the chromosome
        for j in range(len(chromosome)):
            # Get the current color of the gene
            current_color = int(chromosome[j])

            # Get the neighbors of the current gene
            neighbors = list(graph.neighbors(j))

            # Determine the set of colors used by the neighbors
            neighbor_colors = set(int(chromosome[int(n)]) for n in neighbors)

            # Determine the set of possible colors for the current gene
            possible_colors = set(range(len(ga_instance.gene_space)))

            # Determine the set of available colors that do not conflict with the neighbors
            available_colors = possible_colors - neighbor_colors

            # If there are no available colors, choose a random color
            if len(available_colors) == 0:
                new_color = random.randint(0, len(ga_instance.gene_space) - 1)
            # Otherwise, choose a color from the available colors
            else:
                new_color = random.choice(list(available_colors))

            # If the new color is different from the current color, update the chromosome
            if new_color != current_color:
                chromosome[j] = new_color

        # Update the offspring with the new chromosome
        offspring[i] = chromosome

    return offspring


def mutation_low_occurence(offspring, ga_instance):
    # Calculate the frequency of each color in the population
    color_counts = [0] * len(ga_instance.gene_space)
    for chromosome in offspring:
        for color in chromosome:
            color_counts[int(color)] += 1

    # Iterate over each offspring in the population
    for i in range(len(offspring)):
        # Get the chromosome of the current offspring
        chromosome = offspring[i]

        # Iterate over each gene in the chromosome
        for j in range(len(chromosome)):
            # Get the current color of the gene
            current_color = int(chromosome[j])

            # Get the frequency of the current color in the population
            current_color_count = color_counts[current_color]

            # Determine the set of possible colors for the current gene
            possible_colors = set(range(len(ga_instance.gene_space)))

            # Determine the color with the lowest frequency
            min_color = min(possible_colors, key=lambda c: color_counts[c])

            # If the current color has the lowest frequency, don't change it
            if current_color_count == color_counts[min_color]:
                continue

            # Update the chromosome with the new color
            chromosome[j] = min_color

            # Update the frequency count for the old and new colors
            color_counts[current_color] -= 1
            color_counts[min_color] += 1

        # Update the offspring with the new chromosome
        offspring[i] = chromosome

    return offspring


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
best_individual_fitness = 0
generations_sum = 0
generation_fitness = 0.0

while solution_fitness == 1:
    print(f"Starting GA with {colors_number} colors to find solution.")

    # Create an instance of the GA class
    ga_instance = pygad.GA(
        num_generations=2000,
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
        mutation_type=mutate_no_conflict,
        stop_criteria=["reach_1"],
    )

    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()

    if solution_fitness == 1:
        best_individual = solution
        best_individual_fitness = solution_fitness
        generations_sum += ga_instance.generations_completed
        generation_fitness = ga_instance.last_generation_fitness
        ga_instance.plot_fitness(save_dir="fitness.png")
        plt.close()

    colors_number -= 1


print("AVERAGE FITNESS: ", str(np.average(generation_fitness)))
print("GENERATIONS SUM: ", str(generations_sum))
print("UNIQUE COLORS: ", str(get_unique_colors_number(best_individual)))

# Visualize the best solution
best_solution_int = np.array(best_individual, dtype=int)
pos = nx.spring_layout(graph)
nx.draw(
    graph, pos, node_color=[colors[best_solution_int[i]] for i in range(len(graph))]
)
plt.savefig("graph_coloring.png")
