import numpy as np
from TSP.TSP import *

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