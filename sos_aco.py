import numpy as np
import random

# A classe SOS_ACO foi refatorada para maior integração com a tsplib95
class SOS_ACO:
    """
    Implementation of the hybrid Symbiotic Organisms Search (SOS) and 
    Ant Colony Optimization (ACO) algorithm, now integrated with tsplib95.
    """
    def __init__(self, problem, n_ants, n_organisms, max_iter, rho=0.1, q=100,
                 alpha_range=(1, 5), beta_range=(1, 5)):
        """
        Initializes the SOS-ACO solver.

        Args:
            problem (tsplib95.Problem): A problem instance loaded from tsplib95.
            n_ants (int): Number of ants in the colony (m).
            n_organisms (int): Number of organisms in the SOS ecosystem (N).
            ... (other params)
        """
        # --- General Parameters ---
        self.problem = problem
        # Convert node coords from dict to numpy array
        self.cities = np.array([problem.node_coords[i] for i in sorted(problem.node_coords.keys())])
        self.n_cities = len(self.cities)
        self.max_iter = max_iter
        
        # --- ACO Parameters ---
        self.n_ants = n_ants
        self.rho = rho
        self.q = q
        
        # --- SOS Parameters ---
        self.n_organisms = n_organisms
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        
        # --- Internal State ---
        self.distance_matrix = self._calculate_distance_matrix()
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        self.heuristic_info = 1 / (self.distance_matrix + 1e-10)
        
        self.ecosystem = np.random.uniform(
            low=[self.alpha_range[0], self.beta_range[0]],
            high=[self.alpha_range[1], self.beta_range[1]],
            size=(self.n_organisms, 2)
        )
        self.fitness = np.zeros(self.n_organisms)

        self.best_tour = None
        self.best_tour_length = float('inf')
        self.history = [] # To store the best length at each iteration

    def _calculate_distance_matrix(self):
        """
        Calculates the distance matrix using the problem's edge weight function.
        tsplib95 uses 1-based indexing, so we adjust our 0-based loops.
        """
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i, self.n_cities):
                # Using the library's get_weight is more robust
                dist = self.problem.get_weight(i + 1, j + 1)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _calculate_tour_length(self, tour):
        """Calculates the total length of a given tour."""
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distance_matrix[tour[i], tour[i+1]]
        length += self.distance_matrix[tour[-1], tour[0]]
        return length
    
    # ... (Os métodos _construct_tour, _run_aco_for_fitness, _apply_local_search, _update_pheromone
    # permanecem os mesmos da versão anterior, pois já são genéricos)
    def _construct_tour(self, start_city, alpha, beta):
        tour = [start_city]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_city)
        current_city = start_city
        while unvisited:
            probs = []
            for next_city in unvisited:
                pheromone_val = self.pheromone[current_city, next_city] ** alpha
                heuristic_val = self.heuristic_info[current_city, next_city] ** beta
                probs.append(pheromone_val * heuristic_val)
            probs = np.array(probs)
            probs_sum = probs.sum()
            if probs_sum == 0:
                next_city = random.choice(unvisited)
            else:
                probs /= probs_sum
                next_city = np.random.choice(unvisited, p=probs)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        return tour
    def _run_aco_for_fitness(self, organism):
        alpha, beta = organism
        tours = []
        for i in range(self.n_ants):
            start_city = random.randint(0, self.n_cities - 1)
            tours.append(self._construct_tour(start_city, alpha, beta))
        lengths = [self._calculate_tour_length(tour) for tour in tours]
        best_length = min(lengths)
        return 1 / (best_length + 1e-10)
    def _apply_local_search(self, tour):
        n = self.n_cities
        current_tour = list(tour)
        current_length = self._calculate_tour_length(current_tour)
        for i in range(n):
            k = current_tour[i]
            tour_indices = {city_id: index for index, city_id in enumerate(current_tour)}
            best_neighbor_dist = float('inf')
            best_neighbor_idx = -1
            for j in range(i + 1, n):
                neighbor_city = current_tour[j]
                dist = self.distance_matrix[k, neighbor_city]
                if dist < best_neighbor_dist:
                    best_neighbor_dist = dist
                    best_neighbor_idx = j
            if best_neighbor_idx != -1:
                new_tour = current_tour[:]
                segment_to_reverse = new_tour[i+1 : best_neighbor_idx+1]
                segment_to_reverse.reverse()
                new_tour[i+1 : best_neighbor_idx+1] = segment_to_reverse
                new_length = self._calculate_tour_length(new_tour)
                if new_length < current_length:
                    current_tour = new_tour
                    current_length = new_length
        return current_tour
    def _update_pheromone(self, tours, best_tour_length):
        self.pheromone *= (1 - self.rho)
        for tour in tours:
            tour_len = self._calculate_tour_length(tour)
            pheromone_to_add = self.q / (tour_len + 1e-10)
            for i in range(self.n_cities - 1):
                self.pheromone[tour[i], tour[i+1]] += pheromone_to_add
                self.pheromone[tour[i+1], tour[i]] += pheromone_to_add
            self.pheromone[tour[-1], tour[0]] += pheromone_to_add
            self.pheromone[tour[0], tour[-1]] += pheromone_to_add
            
    def solve(self, verbose=False):
        """
        The main method to run the SOS-ACO algorithm. Now returns history.
        """
        print("Starting SOS-ACO optimization...")
        
        for i in range(self.max_iter):
            # --- SOS Phase ---
            for j in range(self.n_organisms): self.fitness[j] = self._run_aco_for_fitness(self.ecosystem[j])
            best_organism_idx = np.argmax(self.fitness)
            best_organism = self.ecosystem[best_organism_idx]
            # Mutualism Phase
            for j in range(self.n_organisms):
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                mutual_vector = (self.ecosystem[j] + self.ecosystem[k]) / 2
                bf1, bf2 = random.choice([1, 2]), random.choice([1, 2])
                new_j = self.ecosystem[j] + random.random() * (best_organism - mutual_vector * bf1)
                new_k = self.ecosystem[k] + random.random() * (best_organism - mutual_vector * bf2)
                new_j = np.clip(new_j, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])
                new_k = np.clip(new_k, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])
                new_fitness_j = self._run_aco_for_fitness(new_j); new_fitness_k = self._run_aco_for_fitness(new_k)
                if new_fitness_j > self.fitness[j]: self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j
                if new_fitness_k > self.fitness[k]: self.ecosystem[k], self.fitness[k] = new_k, new_fitness_k
            # Commensalism Phase
            for j in range(self.n_organisms):
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                new_j = self.ecosystem[j] + (random.uniform(-1, 1)) * (best_organism - self.ecosystem[k])
                new_j = np.clip(new_j, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])
                new_fitness_j = self._run_aco_for_fitness(new_j)
                if new_fitness_j > self.fitness[j]: self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j
            # Parasitism Phase
            for j in range(self.n_organisms):
                parasite_vector = self.ecosystem[j].copy()
                dim_to_modify = random.randint(0, 1)
                parasite_vector[dim_to_modify] = np.random.uniform(self.alpha_range[0] if dim_to_modify == 0 else self.beta_range[0], self.alpha_range[1] if dim_to_modify == 0 else self.beta_range[1])
                host_idx = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                parasite_fitness = self._run_aco_for_fitness(parasite_vector)
                if parasite_fitness > self.fitness[host_idx]: self.ecosystem[host_idx], self.fitness[host_idx] = parasite_vector, parasite_fitness

            # --- ACO Phase ---
            best_alpha, best_beta = self.ecosystem[np.argmax(self.fitness)]
            current_tours = [self._construct_tour(random.randint(0, self.n_cities - 1), best_alpha, best_beta) for _ in range(self.n_ants)]
            current_lengths = [self._calculate_tour_length(tour) for tour in current_tours]
            iteration_best_length = min(current_lengths)
            iteration_best_tour = current_tours[np.argmin(current_lengths)]
            iteration_best_tour = self._apply_local_search(iteration_best_tour)
            iteration_best_length = self._calculate_tour_length(iteration_best_tour)
            if iteration_best_length < self.best_tour_length:
                self.best_tour = iteration_best_tour
                self.best_tour_length = iteration_best_length
            
            # Store history and print progress
            self.history.append(self.best_tour_length)
            if verbose:
                print(f"Iteration {i+1}/{self.max_iter} | Best Length: {self.best_tour_length:.2f} | Best [α, β]: [{best_alpha:.2f}, {best_beta:.2f}]")
            
        print("\nOptimization finished.")
        # Return history for plotting
        return self.best_tour, self.best_tour_length, self.history