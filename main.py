import tkinter as tk
from TSP.TSP import TSP

def run_TSP():
    root.destroy()
    TSP()

root = tk.Tk()
root.title("Programme de résolution VRP-TSP")

tsp_button = tk.Button(root,text="Résolution du problème voyageur de commerce", command=run_TSP)
tsp_button.pack(pady=10)

tsp_button = tk.Button(root,text="Résolution du problème VRP", command=print("Pas encore créé"))
tsp_button.pack(pady=10)

root.mainloop()