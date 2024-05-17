import random

# Exemple de données du problème VRP
# Chaque client est représenté par un tuple (id_client, demande)
clients = [
    (1, 10), (2, 5), (3, 8), (4, 3), (5, 7),
    (6, 2), (7, 6), (8, 9), (9, 4), (10, 12)
]

# Capacité maximale de chaque véhicule
CAPACITY = 20
# Nombre de véhicules disponibles
NUM_VEHICLES = 3
# Nombre de générations
GENERATIONS = 100
# Taille de la population
POPULATION_SIZE = 50

# Génère une solution initiale aléatoire
def generate_solution():
    # Séquence de clients à visiter pour chaque véhicule
    solution = []
    # Mélange les clients
    random.shuffle(clients)
    # Répartit les clients entre les véhicules
    for i in range(0, len(clients), len(clients)//NUM_VEHICLES):
        solution.append(clients[i:i+len(clients)//NUM_VEHICLES])
    return solution

# Calcule la distance totale d'un itinéraire
def calculate_distance(route):
    total_distance = 0
    for i in range(len(route)):
        for j in range(len(route[i]) - 1):
            total_distance += distance(route[i][j], route[i][j+1])
    return total_distance

# Calcule la distance euclidienne entre deux clients
def distance(client1, client2):
    # Ici, nous utilisons simplement l'identifiant des clients comme indicateur de distance
    return abs(client1[0] - client2[0])

# Vérifie si une solution est valide (capacité des véhicules respectée)
def is_valid(solution):
    for vehicle_route in solution:
        total_demand = sum(client[1] for client in vehicle_route)
        if total_demand > CAPACITY:
            return False
    return True

# Sélectionne des parents pour la reproduction (sélection de tournoi)
def select_parents(population):
    tournament_size = 5
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda x: fitness(x))
        parents.append(winner)
    return parents

# Croisement (reproduction) entre deux parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation d'une solution (échange de clients entre les véhicules)
def mutate(solution):
    vehicle1, vehicle2 = random.sample(solution, 2)
    client1 = random.choice(vehicle1)
    client2 = random.choice(vehicle2)
    vehicle1[vehicle1.index(client1)], vehicle2[vehicle2.index(client2)] = client2, client1
    return solution

# Fonction d'évaluation (à minimiser)
def fitness(solution):
    return calculate_distance(solution)

# Algorithme génétique principal
def genetic_algorithm():
    population = [generate_solution() for _ in range(POPULATION_SIZE)]

    for _ in range(GENERATIONS):
        parents = select_parents(population)
        next_generation = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])

        population = next_generation

    best_solution = min(population, key=fitness)
    return best_solution, fitness(best_solution)

# Exécution de l'algorithme génétique
best_solution, min_distance = genetic_algorithm()
print("Meilleure solution trouvée:", best_solution)
print("Distance minimale:", min_distance)
