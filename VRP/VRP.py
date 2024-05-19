import tkinter as tk
from tkinter import ttk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

def load_background_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

test = load_background_image("static/fond.png")

# Paramètres pour la génération des clients
MAP_SIZE = 100  # Taille de la carte (100x100)
CAPACITY = 20  # Capacité maximale de chaque véhicule
GENERATIONS = 100  # Nombre de générations
POPULATION_SIZE = 50  # Taille de la population

# Définition du dépôt (initial)
DEPOT = (0, MAP_SIZE // 2, MAP_SIZE // 2, 0)  # id = 0, x = 50, y = 50, demande = 0

# Génère des clients de manière aléatoire
def generate_clients(num_clients):
    clients = []
    for i in range(1, num_clients + 1):
        x = random.randint(0, MAP_SIZE)
        y = random.randint(0, MAP_SIZE)
        demand = random.randint(1, 10)  # Demande entre 1 et 10
        clients.append((i, x, y, demand))
    return clients

# Génère une solution initiale aléatoire
def generate_solution(clients, depot, num_vehicles):
    all_clients = clients[:]
    random.shuffle(all_clients)
    solution = [[] for _ in range(num_vehicles)]
    for i, client in enumerate(all_clients):
        solution[i % num_vehicles].append(client)
    # Ajout du dépôt au début et à la fin de chaque route
    for route in solution:
        route.insert(0, depot)
        route.append(depot)
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
def crossover(parent1, parent2, num_vehicles):
    child1, child2 = [], []
    for i in range(num_vehicles):
        child1.append(parent1[i][:])
        child2.append(parent2[i][:])
    # Transfert des clients pour garantir que tous les clients sont présents
    all_clients = set(client for route in parent1 + parent2 for client in route)
    assigned_clients1 = set(client for route in child1 for client in route)
    assigned_clients2 = set(client for route in child2 for client in route)
    unassigned_clients1 = list(all_clients - assigned_clients1)
    unassigned_clients2 = list(all_clients - assigned_clients2)
    for client in unassigned_clients1:
        child1[random.randint(0, num_vehicles - 1)].insert(-1, client)
    for client in unassigned_clients2:
        child2[random.randint(0, num_vehicles - 1)].insert(-1, client)
    return child1, child2

# Mutation d'une solution (échange de clients entre les véhicules)
def mutate(solution, num_vehicles):
    vehicle1, vehicle2 = random.sample(range(num_vehicles), 2)
    if len(solution[vehicle1]) > 2 and len(solution[vehicle2]) > 2:
        idx1 = random.randint(1, len(solution[vehicle1]) - 2)
        idx2 = random.randint(1, len(solution[vehicle2]) - 2)
        solution[vehicle1][idx1], solution[vehicle2][idx2] = solution[vehicle2][idx2], solution[vehicle1][idx1]
    return solution

# Fonction d'évaluation (à minimiser)
def fitness(solution):
    return calculate_distance(solution)

# Affichage du graphe et de la meilleure route
def plot_graph(solution, ax, clients, depot):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    ax.clear()
    for idx, vehicle_route in enumerate(solution):
        x = [client[1] for client in vehicle_route]
        y = [client[2] for client in vehicle_route]
        ax.plot(x, y, marker='o', color=colors[idx % len(colors)], label=f'Vehicle {idx+1}')
    for client in clients:
        ax.text(client[1], client[2], str(client[0]), fontsize=12)
    # Affichage du dépôt
    ax.scatter(depot[1], depot[2], color='black', marker='s', s=100, label='Dépôt')
    ax.legend()
    ax.set_title('Meilleure route trouvée')
    ax.imshow(test, extent=[-100, 100, -100, 100])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Algorithme génétique principal
def genetic_algorithm(clients, depot, num_vehicles):
    population = [generate_solution(clients, depot, num_vehicles) for _ in range(POPULATION_SIZE)]
    best_solution = None
    min_distance = float('inf')

    for _ in range(GENERATIONS):
        parents = select_parents(population)
        next_generation = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2, num_vehicles)
            next_generation.extend([mutate(child1, num_vehicles), mutate(child2, num_vehicles)])

        population = next_generation

        # Mettre à jour la meilleure solution et la distance minimale
        for solution in population:
            if is_valid(solution):
                route_distance = fitness(solution)
                if route_distance < min_distance:
                    best_solution = solution
                    min_distance = route_distance

    return best_solution, min_distance

class VRPApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Vehicle Routing Problem")
        self.geometry("800x600")

        self.clients = generate_clients(10)
        self.depot = DEPOT
        self.num_vehicles = 5  # Valeur par défaut pour le nombre de véhicules

        self.num_nodes_label = ttk.Label(self, text="Nombre de clients:")
        self.num_nodes_label.pack(pady=5)

        self.num_nodes_entry = ttk.Entry(self)
        self.num_nodes_entry.pack(pady=5)
        self.num_nodes_entry.insert(0, "10")  # Valeur par défaut

        self.num_vehicles_label = ttk.Label(self, text="Nombre de véhicules:")
        self.num_vehicles_label.pack(pady=5)

        self.num_vehicles_entry = ttk.Entry(self)
        self.num_vehicles_entry.pack(pady=5)
        self.num_vehicles_entry.insert(0, str(self.num_vehicles))  # Valeur par défaut

        self.depot_label = ttk.Label(self, text="Coordonnées du dépôt (x, y):")
        self.depot_label.pack(pady=5)

        self.depot_x_entry = ttk.Entry(self, width=10)
        self.depot_x_entry.pack(pady=5)
        self.depot_x_entry.insert(0, str(MAP_SIZE // 2))

        self.depot_y_entry = ttk.Entry(self, width=10)
        self.depot_y_entry.pack(pady=5)
        self.depot_y_entry.insert(0, str(MAP_SIZE // 2))

        self.generate_button = ttk.Button(self, text="Régénérer les points", command=self.regenerate_points)
        self.generate_button.pack(pady=10)

        self.figure, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_initial_graph()

    def regenerate_points(self):
        num_nodes = int(self.num_nodes_entry.get())
        self.clients = generate_clients(num_nodes)
        self.num_vehicles = int(self.num_vehicles_entry.get())
        depot_x = int(self.depot_x_entry.get())
        depot_y = int(self.depot_y_entry.get())
        self.depot = (0, depot_x, depot_y, 0)
        best_solution, min_distance = genetic_algorithm(self.clients, self.depot, self.num_vehicles)
        plot_graph(best_solution, self.ax, self.clients, self.depot)
        self.canvas.draw()

    def plot_initial_graph(self):
        best_solution, min_distance = genetic_algorithm(self.clients, self.depot, self.num_vehicles)
        plot_graph(best_solution, self.ax, self.clients, self.depot)
        self.canvas.draw()
