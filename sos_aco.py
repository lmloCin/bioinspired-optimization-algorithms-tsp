import numpy as np
import random

class SOS_ACO:
    """
    Implementation of the hybrid Symbiotic Organisms Search (SOS) and 
    Ant Colony Optimization (ACO) algorithm for the Traveling Salesman Problem (TSP)
    as described in the paper (Wang & Han, 2021).
    
    Reference: https://doi.org/10.1016/j.asoc.2021.107439
    """
    def __init__(self, cities, n_ants, n_organisms, max_iter, rho=0.1, q=100,
                 alpha_range=(1, 5), beta_range=(1, 5)):
        """
        Initializes the SOS-ACO solver.

        Args:
            cities (np.array): A numpy array of shape (n_cities, 2) with city coordinates.
            n_ants (int): Number of ants in the colony (m).
            n_organisms (int): Number of organisms in the SOS ecosystem (N).
            max_iter (int): Maximum number of iterations.
            rho (float): Pheromone evaporation rate.
            q (float): Pheromone intensity coefficient (Q).
            alpha_range (tuple): The search range for the alpha parameter.
            beta_range (tuple): The search range for the beta parameter.
        """
        # --- General Parameters ---
        self.cities = cities
        self.n_cities = len(cities)
        self.max_iter = max_iter
        
        # --- ACO Parameters ---
        self.n_ants = n_ants
        self.rho = rho # Pheromone evaporation rate
        self.q = q     # Pheromone enhancement coefficient
        
        # --- SOS Parameters ---
        self.n_organisms = n_organisms
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        
        # --- Internal State ---
        self.distance_matrix = self._calculate_distance_matrix()
        self.pheromone = np.ones((self.n_cities, self.n_cities)) # Initial pheromone
        self.heuristic_info = 1 / (self.distance_matrix + 1e-10) # Heuristic info (eta)
        
        # Initialize the ecosystem for SOS
        # Each organism is a candidate solution [alpha, beta]
        self.ecosystem = np.random.uniform(
            low=[self.alpha_range[0], self.beta_range[0]],
            high=[self.alpha_range[1], self.beta_range[1]],
            size=(self.n_organisms, 2)
        )
        self.fitness = np.zeros(self.n_organisms)

        # To store the best solution found so far
        self.best_tour = None
        self.best_tour_length = float('inf')

    def _calculate_distance_matrix(self):
        """Pre-calculates the Euclidean distance between all pairs of cities."""
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i, self.n_cities):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _calculate_tour_length(self, tour):
        """Calculates the total length of a given tour."""
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distance_matrix[tour[i], tour[i+1]]
        length += self.distance_matrix[tour[-1], tour[0]] # Return to start
        return length
    
    def _construct_tour(self, start_city, alpha, beta):
        """Constructs a single tour for one ant."""
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
            
            # Select next city based on probability
            probs = np.array(probs)
            probs_sum = probs.sum()
            if probs_sum == 0: # Handle cases where all probabilities are zero
                next_city = random.choice(unvisited)
            else:
                probs /= probs_sum
                next_city = np.random.choice(unvisited, p=probs)

            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        return tour

    def _run_aco_for_fitness(self, organism):
        """
        Runs a simplified ACO iteration to evaluate the fitness of an organism [alpha, beta].
        This is the fitness function for the SOS algorithm.
        """
        alpha, beta = organism
        tours = []
        for i in range(self.n_ants):
            start_city = random.randint(0, self.n_cities - 1)
            tours.append(self._construct_tour(start_city, alpha, beta))
        
        lengths = [self._calculate_tour_length(tour) for tour in tours]
        best_length = min(lengths)
        
        # Fitness is the inverse of the best length found 
        return 1 / (best_length + 1e-10)

    def _apply_local_search(self, tour):
        """
        Applies the path reversal local optimization strategy to a tour. 
        """
        n = self.n_cities
        current_tour = list(tour)
        current_length = self._calculate_tour_length(current_tour)
        
        for i in range(n):
            # Find the city k' nearest to city k among the following cities [cite: 231]
            k = current_tour[i]
            # Create a map from city_id to its index in the current tour
            tour_indices = {city_id: index for index, city_id in enumerate(current_tour)}

            best_neighbor_dist = float('inf')
            best_neighbor_idx = -1
            
            # Search for the nearest neighbor in the rest of the tour
            for j in range(i + 1, n):
                neighbor_city = current_tour[j]
                dist = self.distance_matrix[k, neighbor_city]
                if dist < best_neighbor_dist:
                    best_neighbor_dist = dist
                    best_neighbor_idx = j
            
            if best_neighbor_idx != -1:
                # Reverse the path between v_k+1 and v_k' [cite: 236]
                new_tour = current_tour[:]
                segment_to_reverse = new_tour[i+1 : best_neighbor_idx+1]
                segment_to_reverse.reverse()
                new_tour[i+1 : best_neighbor_idx+1] = segment_to_reverse
                
                new_length = self._calculate_tour_length(new_tour)
                # Keep the change if the new path is shorter [cite: 237]
                if new_length < current_length:
                    current_tour = new_tour
                    current_length = new_length
        
        return current_tour

    def _update_pheromone(self, tours, best_tour_length):
        """Updates the pheromone matrix."""
        # Pheromone evaporation 
        self.pheromone *= (1 - self.rho)
        
        # Pheromone deposition [cite: 156, 157]
        # In the paper, all ants deposit pheromone based on their tour length.
        # A common variation is to let only the best ant deposit. Here we follow the paper.
        for tour in tours:
            tour_len = self._calculate_tour_length(tour)
            # The paper defines delta_tau as Q/L_k [cite: 164]
            pheromone_to_add = self.q / tour_len
            for i in range(self.n_cities - 1):
                self.pheromone[tour[i], tour[i+1]] += pheromone_to_add
                self.pheromone[tour[i+1], tour[i]] += pheromone_to_add
            self.pheromone[tour[-1], tour[0]] += pheromone_to_add
            self.pheromone[tour[0], tour[-1]] += pheromone_to_add
            
    def solve(self):
        """
        The main method to run the SOS-ACO algorithm.
        """
        print("Starting SOS-ACO optimization...")
        
        for i in range(self.max_iter):
            # --- SOS Phase: Optimize ACO parameters ---
            
            # 1. Evaluate fitness of each organism
            for j in range(self.n_organisms):
                self.fitness[j] = self._run_aco_for_fitness(self.ecosystem[j])
            
            best_organism_idx = np.argmax(self.fitness)
            best_organism = self.ecosystem[best_organism_idx]

            # 2. Mutualism Phase [cite: 177]
            for j in range(self.n_organisms):
                # Select a random organism X_k different from X_j
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                
                mutual_vector = (self.ecosystem[j] + self.ecosystem[k]) / 2
                bf1, bf2 = random.choice([1, 2]), random.choice([1, 2])
                
                # Create new candidates [cite: 180]
                new_j = self.ecosystem[j] + random.random() * (best_organism - mutual_vector * bf1)
                new_k = self.ecosystem[k] + random.random() * (best_organism - mutual_vector * bf2)
                
                # Clip to stay within bounds
                new_j = np.clip(new_j, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])
                new_k = np.clip(new_k, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])

                new_fitness_j = self._run_aco_for_fitness(new_j)
                if new_fitness_j > self.fitness[j]:
                    self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j

                new_fitness_k = self._run_aco_for_fitness(new_k)
                if new_fitness_k > self.fitness[k]:
                    self.ecosystem[k], self.fitness[k] = new_k, new_fitness_k
            
            # 3. Commensalism Phase [cite: 206]
            for j in range(self.n_organisms):
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                
                # Create new candidate [cite: 210]
                new_j = self.ecosystem[j] + (random.uniform(-1, 1)) * (best_organism - self.ecosystem[k])
                new_j = np.clip(new_j, [self.alpha_range[0], self.beta_range[0]], [self.alpha_range[1], self.beta_range[1]])

                new_fitness_j = self._run_aco_for_fitness(new_j)
                if new_fitness_j > self.fitness[j]:
                    self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j

            # 4. Parasitism Phase [cite: 217]
            for j in range(self.n_organisms):
                # Create a parasite vector by copying and modifying an organism
                parasite_vector = self.ecosystem[j].copy()
                dim_to_modify = random.randint(0, 1)
                parasite_vector[dim_to_modify] = np.random.uniform(
                    low=self.alpha_range[0] if dim_to_modify == 0 else self.beta_range[0],
                    high=self.alpha_range[1] if dim_to_modify == 0 else self.beta_range[1]
                )
                
                # Select a random host to compete against
                host_idx = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                
                parasite_fitness = self._run_aco_for_fitness(parasite_vector)
                if parasite_fitness > self.fitness[host_idx]:
                    self.ecosystem[host_idx], self.fitness[host_idx] = parasite_vector, parasite_fitness

            # --- ACO Phase: Solve TSP with optimized parameters ---
            
            # Get the best parameters found by SOS in this iteration
            best_alpha, best_beta = self.ecosystem[np.argmax(self.fitness)]

            # Construct tours using the best parameters
            current_tours = []
            for _ in range(self.n_ants):
                start_city = random.randint(0, self.n_cities - 1)
                current_tours.append(self._construct_tour(start_city, best_alpha, best_beta))
                
            # Find the best tour in this iteration
            current_lengths = [self._calculate_tour_length(tour) for tour in current_tours]
            iteration_best_length = min(current_lengths)
            iteration_best_tour = current_tours[np.argmin(current_lengths)]

            # Apply local search to the iteration's best tour
            iteration_best_tour = self._apply_local_search(iteration_best_tour)
            iteration_best_length = self._calculate_tour_length(iteration_best_tour)

            # Update the overall best solution found
            if iteration_best_length < self.best_tour_length:
                self.best_tour = iteration_best_tour
                self.best_tour_length = iteration_best_length
            
            # Update pheromone trails
            self._update_pheromone(current_tours, self.best_tour_length)

            print(f"Iteration {i+1}/{self.max_iter} | Best Length: {self.best_tour_length:.2f} | Best [α, β]: [{best_alpha:.2f}, {best_beta:.2f}]")
            
        print("\nOptimization finished.")
        return self.best_tour, self.best_tour_length