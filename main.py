import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, distance_matrix, start_city=0, num_ants=10, max_iter=100, alpha=1, beta=3, rho=0.1, q=100):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.start_city = start_city
        
    def run(self):
        pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        best_path = None
        best_path_length = np.inf
        
        for _ in range(self.max_iter):
            ant_paths = self.generate_ant_paths(pheromone_matrix)
            self.update_pheromones(pheromone_matrix, ant_paths)
            
            current_best_path = min(ant_paths, key=lambda x: self.calculate_path_length(x))
            current_best_path_length = self.calculate_path_length(current_best_path)
            
            if current_best_path_length < best_path_length:
                best_path = current_best_path
                best_path_length = current_best_path_length
        
        return best_path, best_path_length
    
    def generate_ant_paths(self, pheromone_matrix):
        ant_paths = []
        for _ in range(self.num_ants):
            visited = set()
            current_city = self.start_city
            visited.add(current_city)
            path = [current_city]
            
            while len(visited) < self.num_cities:
                probabilities = self.calculate_probabilities(pheromone_matrix, path[-1], visited)
                next_city = np.random.choice(range(self.num_cities), p=probabilities)
                path.append(next_city)
                visited.add(next_city)
                
            ant_paths.append(path)
        
        return ant_paths
    
    def calculate_probabilities(self, pheromone_matrix, current_city, visited):
        probabilities = []
        total = 0
        for city in range(self.num_cities):
            if city not in visited:
                pheromone = pheromone_matrix[current_city][city]
                distance = self.distance_matrix[current_city][city]
                total += (pheromone ** self.alpha) * ((1 / distance) ** self.beta)
                probabilities.append((pheromone ** self.alpha) * ((1 / distance) ** self.beta))
            else:
                probabilities.append(0)
        
        probabilities = [prob / total for prob in probabilities]
        return probabilities
    
    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distance_matrix[path[i]][path[i+1]]
        length += self.distance_matrix[path[-1]][path[0]]  # Return to the starting city
        return length
    
    def update_pheromones(self, pheromone_matrix, ant_paths):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    pheromone_matrix[i][j] *= (1 - self.rho)
        
        for path in ant_paths:
            path_length = self.calculate_path_length(path)
            for i in range(len(path) - 1):
                pheromone_matrix[path[i]][path[i+1]] += self.q / path_length
                pheromone_matrix[path[i+1]][path[i]] += self.q / path_length
            pheromone_matrix[path[-1]][path[0]] += self.q / path_length
            pheromone_matrix[path[0]][path[-1]] += self.q / path_length


def distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

def total_distance(solution, dist_matrix):
    distance = 0
    n = len(solution)
    for i in range(n - 1):
        distance += dist_matrix[solution[i]][solution[i+1]]
    distance += dist_matrix[solution[-1]][solution[0]]  # Return to the starting city
    return distance

def simulated_annealing(coords, initial_temp=1000, cooling_rate=0.99, num_iter=1000):
    n = len(coords)
    current_solution = np.random.permutation(n)
    best_solution = current_solution.copy()
    dist_matrix = distance_matrix(coords)
    current_cost = total_distance(current_solution, dist_matrix)
    best_cost = current_cost
    
    temp = initial_temp
    for i in range(num_iter):
        neighbor = np.random.permutation(n)
        neighbor_cost = total_distance(neighbor, dist_matrix)
        delta = neighbor_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
        temp *= cooling_rate
    
    return best_solution, best_cost


def generate_random_nodes(num_nodes, min_coord=0, max_coord=100):
    nodes = np.random.randint(min_coord, max_coord, size=(num_nodes, 2))
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            distances[i][j] = np.linalg.norm(nodes[i] - nodes[j])
            distances[j][i] = distances[i][j]  # Symétrie de la matrice
    return nodes, distances

def save_to_csv(nodes, best_path, best_path_length, distances, filename):
    df = pd.DataFrame(nodes, columns=['Node_X', 'Node_Y'])
    
    # Ajouter les colonnes pour le meilleur trajet et la distance minimale
    df['Best_Path'] = np.nan
    df.loc[best_path, 'Best_Path'] = best_path
    df['Best_Path_Length'] = best_path_length
    
    # Ajouter la matrice de distances au DataFrame
    for i in range(len(distances)):
        df[f'Distance_{i}'] = distances[i]

    # Enregistrer les données dans un fichier CSV
    df.to_csv(filename + ".csv", index=False)

    #print(f"Toutes les données ont été enregistrées dans '{filename}.csv'.")

def plot_graph(nodes):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
    for i, node in enumerate(nodes):
        plt.text(node[0], node[1], str(i), fontsize=12)
    plt.title('Graphe des sommets générés')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    return fig

def show_graph_in_gui():
    global nodes, distances
    nodes, distances = generate_random_nodes(10)
    fig = plot_graph(nodes)
    
    def regenerate_points():
        global nodes, distances
        num_nodes = int(num_nodes_entry.get())
        nodes, distances = generate_random_nodes(num_nodes)
        # Efface le graphique existant
        canvas.figure.clear()
        # Trace les nouveaux sommets
        plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
        plt.grid(True)
        for i, node in enumerate(nodes):
            plt.text(node[0], node[1], str(i), fontsize=12)
        canvas.draw()

    def solve_tsp_and_plot():
        start_city = int(start_city_combo.get())
        algorithm = algorithm_combo.get()
        if algorithm == 'Recuit simulé':
            best_path, best_path_length = simulated_annealing(nodes)
            print("Meilleure tournée:", best_path)
            print("Coût minimal:", best_path_length)
            # Efface le graphe existant
            canvas.figure.clear()
            # Trace les sommets
            plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
            for i, node in enumerate(nodes):
                plt.text(node[0], node[1], str(i), fontsize=12)
            # Trace la meilleure tournée
            for i in range(len(best_path) - 1):
                plt.plot([nodes[best_path[i]][0], nodes[best_path[i+1]][0]], [nodes[best_path[i]][1], nodes[best_path[i+1]][1]], color='red')
            plt.plot([nodes[best_path[-1]][0], nodes[best_path[0]][0]], [nodes[best_path[-1]][1], nodes[best_path[0]][1]], color='red')
            plt.title('Meilleure tournée trouvée')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            # Affiche la distance minimale
            plt.text(0.5, -0.1, f"Distance minimale: {best_path_length}", horizontalalignment='center', verticalalignment='center')
            canvas.draw()
            save_to_csv(nodes, best_path, best_path_length, distances, "recuit")

        elif algorithm == 'Colonie de fourmis':
            ant_colony = AntColony(distances, start_city)
            best_path, best_path_length = ant_colony.run()
            print("Meilleure tournée:", best_path)
            print("Coût minimal:", best_path_length)
            # Efface le graphe existant
            canvas.figure.clear()
            # Trace les sommets
            plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
            for i, node in enumerate(nodes):
                plt.text(node[0], node[1], str(i), fontsize=12)
            # Trace la meilleure tournée
            for i in range(len(best_path) - 1):
                plt.plot([nodes[best_path[i]][0], nodes[best_path[i+1]][0]], [nodes[best_path[i]][1], nodes[best_path[i+1]][1]], color='red')
            plt.plot([nodes[best_path[-1]][0], nodes[best_path[0]][0]], [nodes[best_path[-1]][1], nodes[best_path[0]][1]], color='red')
            plt.title('Meilleure tournée trouvée')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            # Affiche la distance minimale
            plt.text(0.5, -0.1, f"Distance minimale: {best_path_length}", horizontalalignment='center', verticalalignment='center')
            canvas.draw()
            save_to_csv(nodes, best_path, best_path_length, distances, "fourmis")

    root = tk.Tk()
    root.title("Résolution du Problème du Voyageur de Commerce")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = tk.Frame(root)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    start_city_label = tk.Label(toolbar, text="Ville de départ :")
    start_city_label.pack(side=tk.LEFT, padx=5)
    start_city_combo = ttk.Combobox(toolbar, values=[str(i) for i in range(len(nodes))])
    start_city_combo.pack(side=tk.LEFT, padx=5)
    start_city_combo.current(0)

    algorithm_label = tk.Label(toolbar, text="Algorithme :")
    algorithm_label.pack(side=tk.LEFT, padx=5)
    algorithm_combo = ttk.Combobox(toolbar, values=['Recuit simulé', 'Colonie de fourmis'])
    algorithm_combo.pack(side=tk.LEFT, padx=5)
    algorithm_combo.current(0)
    
    num_nodes_label = tk.Label(toolbar, text="Nombre de points :")
    num_nodes_label.pack(side=tk.LEFT, padx=5)
    num_nodes_entry = ttk.Entry(toolbar)
    num_nodes_entry.pack(side=tk.LEFT, padx=5)
    num_nodes_entry.insert(0, "10")  # Valeur par défaut
    regenerate_button = tk.Button(toolbar, text="Regénérer points", command=regenerate_points)
    regenerate_button.pack(side=tk.LEFT, padx=5)
    
    
    

    solve_button = tk.Button(toolbar, text="Résoudre", command=solve_tsp_and_plot)
    solve_button.pack(side=tk.RIGHT, padx=5)

    close_button = tk.Button(toolbar, text="Fermer", command=root.destroy)
    close_button.pack(side=tk.RIGHT)

    tk.mainloop()

show_graph_in_gui()
