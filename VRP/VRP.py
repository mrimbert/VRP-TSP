import matplotlib.pyplot as plt
from VRP.genetique import *
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import numpy as np


def load_background_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

test = load_background_image("static/fond.png")

def generate_clients(num_clients, map_size_min=-50, map_size_max=50):
    clients = []
    for i in range(1, num_clients + 1):
        x = np.random.randint(map_size_min, map_size_max)
        y = np.random.randint(map_size_min, map_size_max)
        demand = random.randint(1, 10)  # Demande entre 1 et 10
        clients.append((i, x, y, demand))
    return clients


def plot_graph(solution, min_distance):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(test, extent=[-100, 100, -100, 100])
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for idx, vehicle_route in enumerate(solution):
        x = [client[1] for client in vehicle_route]
        y = [client[2] for client in vehicle_route]
        plt.plot(x, y, marker='o', color=colors[idx % len(colors)], label=f'Vehicle {idx+1}')
    for client in clients:
        plt.text(client[1], client[2], str(client[0]), fontsize=12)
    # Affichage du dépôt
    plt.scatter(DEPOT[1], DEPOT[2], color='black', marker='s', s=100, label='Dépôt')
    plt.legend()
    plt.title('Meilleure route trouvée')
    plt.text(0.5, -0.1, f"Distance minimale: {min_distance}", horizontalalignment='center', verticalalignment='center')
    plt.xlabel('X')
    plt.ylabel('Y')
    return fig


def VRP():
    global clients 
    clients = generate_clients(10)
    best_solution, min_distance = genetic_algorithm(clients)
    print(clients)
    print(best_solution)
    fig = plot_graph(best_solution, min_distance)

    root = tk.Tk()
    root.title("Résolution du Vehicule Routing Problem")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = tk.Frame(root)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    num_nodes_label = tk.Label(toolbar, text="Nombre de clients :")
    num_nodes_label.pack(side=tk.LEFT, padx=5)
    num_nodes_entry = ttk.Entry(toolbar)
    num_nodes_entry.pack(side=tk.LEFT, padx=5)
    num_nodes_entry.insert(0, "10")  # Valeur par défaut

    veh_nodes_label = tk.Label(toolbar, text="Nombre de véhicules :")
    veh_nodes_label.pack(side=tk.LEFT, padx=5)
    veh_nodes_entry = ttk.Entry(toolbar)
    veh_nodes_entry.pack(side=tk.LEFT, padx=5)
    veh_nodes_entry.insert(0, "10")  # Valeur par défaut
    regenerate_button = tk.Button(toolbar, text="Regénérer points", command=print("A dev"))
    regenerate_button.pack(side=tk.LEFT, padx=5)

    close_button = tk.Button(toolbar, text="Fermer", command=root.destroy)
    close_button.pack(side=tk.RIGHT)
