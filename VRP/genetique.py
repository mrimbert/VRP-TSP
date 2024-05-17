import random
import numpy as np
import matplotlib.pyplot as plt

# Capacité maximale de chaque véhicule
CAPACITY = 20
# Nombre de véhicules disponibles
NUM_VEHICLES = 5
# Nombre de générations
GENERATIONS = 100
# Taille de la population
POPULATION_SIZE = 50

# Définition du dépôt
DEPOT = (0, 0, 0, 0)  # id = 0, x = 50, y = 50, demande = 0

# Génère une solution initiale aléatoire
def generate_solution(clients):
    all_clients = clients[:]
    random.shuffle(all_clients)
    solution = [[] for _ in range(NUM_VEHICLES)]
    for i, client in enumerate(all_clients):
        solution[i % NUM_VEHICLES].append(client)
    # Ajout du dépôt au début et à la fin de chaque route
    for route in solution:
        route.insert(0, DEPOT)
        route.append(DEPOT)
    return solution

# Calcule la distance euclidienne entre deux clients
def distance(client1, client2):
    return np.sqrt((client1[1] - client2[1])**2 + (client1[2] - client2[2])**2)

# Calcule la distance totale d'un itinéraire
def calculate_distance(route):
    total_distance = 0
    for vehicle_route in route:
        for i in range(len(vehicle_route) - 1):
            total_distance += distance(vehicle_route[i], vehicle_route[i + 1])
    return total_distance

# Vérifie si une solution est valide (capacité des véhicules respectée)
def is_valid(solution):
    for vehicle_route in solution:
        total_demand = sum(client[3] for client in vehicle_route)
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
    child1, child2 = [], []
    for i in range(NUM_VEHICLES):
        child1.append(parent1[i][:])
        child2.append(parent2[i][:])
    # Transfert des clients pour garantir que tous les clients sont présents
    all_clients = set(client for route in parent1 + parent2 for client in route)
    assigned_clients1 = set(client for route in child1 for client in route)
    assigned_clients2 = set(client for route in child2 for client in route)
    unassigned_clients1 = list(all_clients - assigned_clients1)
    unassigned_clients2 = list(all_clients - assigned_clients2)
    for client in unassigned_clients1:
        child1[random.randint(0, NUM_VEHICLES - 1)].insert(-1, client)
    for client in unassigned_clients2:
        child2[random.randint(0, NUM_VEHICLES - 1)].insert(-1, client)
    return child1, child2

# Mutation d'une solution (échange de clients entre les véhicules)
def mutate(solution):
    vehicle1, vehicle2 = random.sample(solution, 2)
    client1 = random.choice(vehicle1[1:-1])  # Ne pas muter le dépôt
    client2 = random.choice(vehicle2[1:-1])  # Ne pas muter le dépôt
    vehicle1[vehicle1.index(client1)], vehicle2[vehicle2.index(client2)] = client2, client1
    return solution

# Fonction d'évaluation (à minimiser)
def fitness(solution):
    return calculate_distance(solution)

# Algorithme génétique principal
def genetic_algorithm(clients):
    population = [generate_solution(clients) for _ in range(POPULATION_SIZE)]
    best_solution = None
    min_distance = float('inf')

    for _ in range(GENERATIONS):
        parents = select_parents(population)
        next_generation = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([mutate(child1), mutate(child2)])

        population = next_generation

        # Mettre à jour la meilleure solution et la distance minimale
        for solution in population:
            if is_valid(solution):
                route_distance = fitness(solution)
                if route_distance < min_distance:
                    best_solution = solution
                    min_distance = route_distance

    return best_solution, min_distance

# Exécution de l'algorithme génétique
#best_solution, min_distance = genetic_algorithm()
#print("Meilleure solution trouvée:", best_solution)
#print("Distance minimale:", min_distance)