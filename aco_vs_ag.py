import numpy as np
import tsplib95
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import requests
import os

# --- Função Auxiliar para Baixar Arquivo ---
def download_tsp_file(filename):
    if os.path.exists(filename):
        return
    url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{filename}"
    print(f"Baixando arquivo {filename}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'w') as f:
            f.write(r.text)
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o arquivo: {e}")
        exit()

# --------------------------------------------------------------------------
# PARTE 1: CÓDIGO DO ALGORITMO SOS-ACO (FORNECIDO POR VOCÊ)
# --------------------------------------------------------------------------
class SOS_ACO:
    """
    Implementation of the hybrid Symbiotic Organisms Search (SOS) and 
    Ant Colony Optimization (ACO) algorithm, now integrated with tsplib95.
    """
    def __init__(self, problem, n_ants, n_organisms, max_iter, rho=0.1, q=100,
                 alpha_range=(1, 5), beta_range=(1, 5)):
        self.problem = problem
        self.cities = np.array([problem.node_coords[i] for i in sorted(problem.node_coords.keys())])
        self.n_cities = len(self.cities)
        self.max_iter = max_iter
        self.n_ants = n_ants
        self.rho = rho
        self.q = q
        self.n_organisms = n_organisms
        self.alpha_range = alpha_range
        self.beta_range = beta_range
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
        self.history = []

    def _calculate_distance_matrix(self):
        dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i, self.n_cities):
                dist = self.problem.get_weight(i + 1, j + 1)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _calculate_tour_length(self, tour):
        length = 0
        for i in range(self.n_cities - 1):
            length += self.distance_matrix[tour[i], tour[i+1]]
        length += self.distance_matrix[tour[-1], tour[0]]
        return length
    
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
        tours = [self._construct_tour(random.randint(0, self.n_cities - 1), alpha, beta) for _ in range(self.n_ants)]
        lengths = [self._calculate_tour_length(tour) for tour in tours]
        return 1 / (min(lengths) + 1e-10)

    def _update_pheromone(self, tours):
        self.pheromone *= (1 - self.rho)
        for tour in tours:
            tour_len = self._calculate_tour_length(tour)
            pheromone_to_add = self.q / (tour_len + 1e-10)
            for i in range(self.n_cities - 1):
                self.pheromone[tour[i], tour[i+1]] += pheromone_to_add
                self.pheromone[tour[i+1], tour[i]] += pheromone_to_add
            self.pheromone[tour[-1], tour[0]] += pheromone_to_add
            self.pheromone[tour[0], tour[-1]] += pheromone_to_add
            
    def solve(self):
        for i in range(self.max_iter):
            for j in range(self.n_organisms): self.fitness[j] = self._run_aco_for_fitness(self.ecosystem[j])
            best_organism_idx = np.argmax(self.fitness)
            best_organism = self.ecosystem[best_organism_idx]
            for j in range(self.n_organisms):
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                mutual_vector = (self.ecosystem[j] + self.ecosystem[k]) / 2
                bf1, bf2 = random.choice([1, 2]), random.choice([1, 2])
                new_j = self.ecosystem[j] + random.random() * (best_organism - mutual_vector * bf1)
                new_k = self.ecosystem[k] + random.random() * (best_organism - mutual_vector * bf2)
                new_j = np.clip(new_j, self.alpha_range[0], self.alpha_range[1])
                new_k = np.clip(new_k, self.beta_range[0], self.beta_range[1])
                new_fitness_j = self._run_aco_for_fitness(new_j); new_fitness_k = self._run_aco_for_fitness(new_k)
                if new_fitness_j > self.fitness[j]: self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j
                if new_fitness_k > self.fitness[k]: self.ecosystem[k], self.fitness[k] = new_k, new_fitness_k
            for j in range(self.n_organisms):
                k = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                new_j = self.ecosystem[j] + (random.uniform(-1, 1)) * (best_organism - self.ecosystem[k])
                new_j = np.clip(new_j, self.alpha_range[0], self.alpha_range[1])
                new_fitness_j = self._run_aco_for_fitness(new_j)
                if new_fitness_j > self.fitness[j]: self.ecosystem[j], self.fitness[j] = new_j, new_fitness_j
            for j in range(self.n_organisms):
                parasite_vector = self.ecosystem[j].copy()
                dim_to_modify = random.randint(0, 1)
                parasite_vector[dim_to_modify] = np.random.uniform(self.alpha_range[0] if dim_to_modify == 0 else self.beta_range[0], self.alpha_range[1] if dim_to_modify == 0 else self.beta_range[1])
                host_idx = random.choice([idx for idx in range(self.n_organisms) if idx != j])
                parasite_fitness = self._run_aco_for_fitness(parasite_vector)
                if parasite_fitness > self.fitness[host_idx]: self.ecosystem[host_idx], self.fitness[host_idx] = parasite_vector, parasite_fitness

            best_alpha, best_beta = self.ecosystem[np.argmax(self.fitness)]
            current_tours = [self._construct_tour(random.randint(0, self.n_cities - 1), best_alpha, best_beta) for _ in range(self.n_ants)]
            current_lengths = [self._calculate_tour_length(tour) for tour in current_tours]
            
            if min(current_lengths) < self.best_tour_length:
                self.best_tour_length = min(current_lengths)
                self.best_tour = current_tours[np.argmin(current_lengths)]
            
            self._update_pheromone(current_tours)
            self.history.append(self.best_tour_length)
            
        return self.best_tour, self.best_tour_length, self.history

# --------------------------------------------------------------------------
# PARTE 2: CÓDIGO DO ALGORITMO GENÉTICO (GA)
# --------------------------------------------------------------------------
def get_geo_distance_matrix(problem):
    coords = problem.node_coords
    num_cities = len(coords)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = problem.get_weight(i + 1, j + 1)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])
    child[start:end+1] = parent1[start:end+1]
    pointer = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[pointer] = gene
            pointer = (pointer + 1) % size
    return child

def inversion_mutation(tour, mutation_rate):
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(tour)), 2))
        tour[start:end+1] = tour[start:end+1][::-1]
    return tour

def tournament_selection(population, fitnesses, k=2):
    selection_ix = np.random.randint(len(population), size=k)
    best_ix = -1
    best_fitness = -1
    for ix in selection_ix:
        if fitnesses[ix] > best_fitness:
            best_fitness = fitnesses[ix]
            best_ix = ix
    return population[best_ix]

class GeneticAlgorithm:
    def __init__(self, problem, pop_size, mutation_rate, crossover_rate, generations):
        self.problem = problem
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.distance_matrix = get_geo_distance_matrix(self.problem)
        self.num_cities = len(self.problem.node_coords)
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        base_tour = list(range(self.num_cities))
        return [random.sample(base_tour, len(base_tour)) for _ in range(self.pop_size)]

    def _calculate_total_distance(self, tour):
        length = 0
        for i in range(self.num_cities):
            length += self.distance_matrix[tour[i], tour[(i + 1) % self.num_cities]]
        return length

    def run(self):
        best_overall_distance = float('inf')
        history = []
        for gen in range(self.generations):
            distances = [self._calculate_total_distance(tour) for tour in self.population]
            fitnesses = [1.0 / (d + 1e-10) for d in distances]
            best_current_distance = min(distances)
            if best_current_distance < best_overall_distance:
                best_overall_distance = best_current_distance
            history.append(best_overall_distance)
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = tournament_selection(self.population, fitnesses)
                parent2 = tournament_selection(self.population, fitnesses)
                if random.random() < self.crossover_rate:
                    child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
                else:
                    child1, child2 = parent1[:], parent2[:]
                new_population.extend([
                    inversion_mutation(child1, self.mutation_rate),
                    inversion_mutation(child2, self.mutation_rate)
                ])
            self.population = new_population
        return None, best_overall_distance, history

# --------------------------------------------------------------------------
# PARTE 3: BLOCO DE EXECUÇÃO E ANÁLISE COMPARATIVA
# --------------------------------------------------------------------------
if __name__ == '__main__':
    N_RUNS = 30
    PROBLEM_NAME = 'att48'  # Problema usado no artigo
    
    download_tsp_file(f"{PROBLEM_NAME}.tsp")
    problem = tsplib95.load(f"{PROBLEM_NAME}.tsp")
    
    # Parâmetros para os algoritmos
    ga_params = {
        'problem': problem,
        'pop_size': len(problem.node_coords) * 10,
        'generations': 100,
        'crossover_rate': 0.9541,
        'mutation_rate': 0.0017
    }
    sos_aco_params = {
        'problem': problem,
        'n_ants': 10,
        'n_organisms': 10,
        'max_iter': 100, # Igualado ao GA para comparação justa de convergência
        'rho': 0.1,
        'q': 1000,
        'alpha_range': (1, 5),
        'beta_range': (1, 5)
    }
    
    algorithms = {
        "GA (Tournament)": {"class": GeneticAlgorithm, "params": ga_params, "method": "run"},
        "SOS-ACO": {"class": SOS_ACO, "params": sos_aco_params, "method": "solve"}
    }

    results = []
    
    print(f"Iniciando simulações para o problema: {PROBLEM_NAME}...")
    for algo_name, config in algorithms.items():
        print(f"\nProcessando algoritmo: {algo_name} ({N_RUNS} execuções)")
        
        AlgoClass = config["class"]
        params = config["params"]
        method_name = config["method"]

        for i in range(N_RUNS):
            start_time = time.time()
            
            algorithm_instance = AlgoClass(**params)
            run_method = getattr(algorithm_instance, method_name)
            _, best_cost, history = run_method()

            execution_time = time.time() - start_time
            
            results.append({
                'Algorithm': algo_name,
                'Run': i + 1,
                'Best Cost': best_cost,
                'Time': execution_time,
                'History': history
            })
            print(f"  Execução {i+1}/{N_RUNS} -> Custo Final: {best_cost:.2f}")

    print("\nSimulações concluídas. Gerando análise...")
    df_results = pd.DataFrame(results)

    # 1. Tabela Principal Comparativa
    summary_table = df_results.groupby('Algorithm').agg(
        Melhor_Custo=('Best Cost', 'min'),
        Custo_Medio=('Best Cost', 'mean'),
        Desvio_Padrao=('Best Cost', 'std'),
        Tempo_Medio=('Time', 'mean')
    ).round(2)
    
    print("\n--- Tabela Comparativa Principal ---")
    print(summary_table)

    # 2. Gráficos de Caixa (Boxplots)
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df_results, x='Algorithm', y='Best Cost', palette="viridis")
    plt.title(f'Distribuição dos Resultados para "{PROBLEM_NAME}" ({N_RUNS} Execuções)', fontsize=16)
    plt.ylabel('Melhor Custo Encontrado')
    plt.xlabel('Algoritmo')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. Gráficos de Convergência
    plt.figure(figsize=(12, 8))
    for algo_name in algorithms.keys():
        subset = df_results[df_results['Algorithm'] == algo_name]
        histories = np.array(subset['History'].tolist())
        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        
        # Plota a curva média
        plt.plot(mean_history, label=algo_name)
        # Plota a área de desvio padrão para mostrar a variabilidade
        plt.fill_between(range(len(mean_history)), mean_history - std_history, mean_history + std_history, alpha=0.2)
    
    # Adiciona a linha da solução ótima
    best_known_cost = 3323 # Para burma14
    plt.axhline(y=best_known_cost, color='r', linestyle='--', label=f'Ótimo Conhecido ({best_known_cost})')
    
    plt.title(f'Convergência Média dos Algoritmos para "{PROBLEM_NAME}"', fontsize=16)
    plt.xlabel('Geração / Iteração')
    plt.ylabel('Custo Médio da Melhor Solução')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()