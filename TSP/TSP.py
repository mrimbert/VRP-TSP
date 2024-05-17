import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TSP.ant import *
from TSP.recuit import *
from PIL import Image

def load_background_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

test = load_background_image("static/fond.png")

def generate_random_nodes(num_nodes, min_coord=-100, max_coord=100):
    nodes = np.random.randint(min_coord, max_coord, size=(num_nodes, 2))
    #nodes = np.insert(nodes, 1, np.array([-3,45]), axis=0)
    print(nodes)
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
    plt.imshow(test, extent=[-100, 100, -100, 100])
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
    for i, node in enumerate(nodes):
        plt.text(node[0], node[1], str(i), fontsize=12)
    plt.title('Graphe des sommets générés')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    return fig

def TSP():
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
        plt.imshow(test, extent=[-100, 100, -100, 100])
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
            plt.imshow(test, extent=[-100, 100, -100, 100])
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
            plt.imshow(test, extent=[-100, 100, -100, 100])
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