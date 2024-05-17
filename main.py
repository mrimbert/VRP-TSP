import tkinter as tk
from TSP.TSP import TSP
from VRP.VRP import VRP

def run_TSP():
    root.destroy()
    TSP()

def run_VRP():
    root.destroy()
    VRP()

root = tk.Tk()
root.title("Programme de résolution VRP-TSP")

tsp_button = tk.Button(root,text="Résolution du problème voyageur de commerce", command=run_TSP)
tsp_button.pack(pady=10)

tsp_button = tk.Button(root,text="Résolution du problème VRP", command=run_VRP)
tsp_button.pack(pady=10)

root.mainloop()