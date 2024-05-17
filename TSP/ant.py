import numpy as np


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