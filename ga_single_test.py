import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt
import math

# --------------------------------------------------------------------------
# 1. FUNÇÃO PARA PLOTAR A ROTA (NOVO)
# --------------------------------------------------------------------------

def plot_tour(problem, tour, title):
    """
    Plota o gráfico de uma rota do TSP.

    Args:
        problem (tsplib95.Problem): O objeto do problema para obter as coordenadas.
        tour (list): A lista de índices de cidades que formam a rota.
        title (str): O título do gráfico.
    """
    # Extrai as coordenadas do problema
    # O dicionário é 1-based, então criamos uma lista 0-based para facilitar
    coords = [problem.node_coords[i] for i in range(1, len(problem.node_coords) + 1)]
    
    plt.figure(figsize=(10, 8))
    
    # Plota as cidades como pontos
    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, color='blue', s=50)
        plt.text(x + 0.5, y + 0.5, str(i + 1), fontsize=9) # Adiciona o número da cidade

    # Plota as linhas da rota
    for i in range(len(tour)):
        from_city_idx = tour[i]
        to_city_idx = tour[(i + 1) % len(tour)] # O % garante o retorno ao início
        
        # Pega as coordenadas
        from_coord = coords[from_city_idx]
        to_coord = coords[to_city_idx]
        
        # Desenha a linha
        plt.plot([from_coord[0], to_coord[0]], [from_coord[1], to_coord[1]], 'r-')

    plt.title(title, fontsize=16)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Garante que a escala dos eixos seja a mesma
    plt.show()


# --------------------------------------------------------------------------
# 2. FUNÇÕES AUXILIARES E DE CÁLCULO
# --------------------------------------------------------------------------

def calculate_total_distance(tour, distance_matrix):
    """Calcula a distância total de uma rota (tour)."""
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        from_city = tour[i]
        to_city = tour[(i + 1) % num_cities]
        total_distance += distance_matrix[from_city, to_city]
    return total_distance

def get_geo_distance_matrix(coords):
    """Cria uma matriz de distâncias usando a fórmula para coordenadas geográficas (GEO)."""
    num_cities = len(coords)
    distance_matrix = np.zeros((num_cities, num_cities))
    
    def get_geo_distance(coord1, coord2):
        PI = 3.141592
        RRR = 6378.388
        lat1_rad = PI * coord1[0] / 180.0
        lon1_rad = PI * coord1[1] / 180.0
        lat2_rad = PI * coord2[0] / 180.0
        lon2_rad = PI * coord2[1] / 180.0
        q1 = math.cos(lon1_rad - lon2_rad)
        q2 = math.cos(lat1_rad - lat2_rad)
        q3 = math.cos(lat1_rad + lat2_rad)
        distance = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        return distance

    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = get_geo_distance(coords[i+1], coords[j+1])
            distance_matrix[i, j] = distance_matrix[j, i] = dist
            
    return distance_matrix


# --------------------------------------------------------------------------
# 3. OPERADORES GENÉTICOS E MÉTODOS DE SELEÇÃO
# --------------------------------------------------------------------------
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

def swap_mutation(tour, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(tour)), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
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

def proportional_roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

def rank_based_roulette_wheel_selection(population, fitnesses):
    sorted_indices = np.argsort(fitnesses)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(population) + 1)
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


# --------------------------------------------------------------------------
# 4. CLASSE PRINCIPAL DO ALGORITMO GENÉTICO
# --------------------------------------------------------------------------
class GeneticAlgorithm:
    def __init__(self, problem, pop_size=100, mutation_rate=0.01, crossover_rate=0.9, generations=500):
        self.problem = problem
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        
        if self.problem.edge_weight_type == "GEO":
            self.distance_matrix = get_geo_distance_matrix(self.problem.node_coords)
        else:
            def get_euc_2d_distance_matrix(coords):
                num_cities = len(coords)
                matrix = np.zeros((num_cities, num_cities))
                for i in range(num_cities):
                    for j in range(i, num_cities):
                        dist = np.linalg.norm(np.array(coords[i+1]) - np.array(coords[j+1]))
                        matrix[i, j] = matrix[j, i] = dist
                return matrix
            self.distance_matrix = get_euc_2d_distance_matrix(self.problem.node_coords)

        self.num_cities = len(self.problem.node_coords)
        self.population = self._create_initial_population()
        
    def _create_initial_population(self):
        population = []
        base_tour = list(range(self.num_cities))
        for _ in range(self.pop_size):
            tour = random.sample(base_tour, len(base_tour))
            population.append(tour)
        return population

    def run(self, selection_strategy):
        best_overall_tour = None
        best_overall_distance = float('inf')
        history = []
        for gen in range(self.generations):
            distances = [calculate_total_distance(tour, self.distance_matrix) for tour in self.population]
            fitnesses = [1 / (d + 1e-10) for d in distances]
            best_current_distance_idx = np.argmin(distances)
            if distances[best_current_distance_idx] < best_overall_distance:
                best_overall_distance = distances[best_current_distance_idx]
                best_overall_tour = self.population[best_current_distance_idx]
            history.append(best_overall_distance)
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = selection_strategy(self.population, fitnesses)
                parent2 = selection_strategy(self.population, fitnesses)
                if random.random() < self.crossover_rate:
                    child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
                else:
                    child1, child2 = parent1[:], parent2[:]
                new_population.extend([
                    swap_mutation(child1, self.mutation_rate),
                    swap_mutation(child2, self.mutation_rate)
                ])
            self.population = new_population
        return best_overall_tour, best_overall_distance, history

# --------------------------------------------------------------------------
# 5. BLOCO DE EXECUÇÃO DO EXPERIMENTO (MODIFICADO)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    problem = tsplib95.load('burma14.tsp')
    best_known = 3323
    
    print(f"Problema: {problem.name}, Cidades: {len(problem.node_coords)}")
    print(f"Tipo de Distância: {problem.edge_weight_type}")
    print(f"Solução ótima conhecida: {best_known}")
    print("-" * 30)

    POP_SIZE = len(problem.node_coords) * 10
    GENERATIONS = 500
    MUTATION_RATE = 0.02 # Ajustado para um valor mais comum
    CROSSOVER_RATE = 0.9 # Ajustado para um valor mais comum
    
    # <--- MODIFICADO: Roda apenas com a melhor estratégia (Torneio) para simplificar
    selection_strategy = tournament_selection
    
    # <--- ADICIONADO: Variáveis para guardar a melhor rota de todas
    overall_best_tour = None
    overall_best_distance = float('inf')
    
    print(f"Executando GA com a estratégia: {selection_strategy.__name__}")
    
    # Executa o AG uma vez para obter um resultado para plotar
    ga = GeneticAlgorithm(problem, pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE)
    best_tour, best_dist, history = ga.run(selection_strategy=selection_strategy)
    
    print(f"\n--- Resultados Finais do AG ---")
    print(f"Melhor distância encontrada: {best_dist:.2f}")
    error = ((best_dist - best_known) / best_known) * 100
    print(f"Erro percentual: {error:.2f}%")
    print("-" * 30)
        
    # --- Plota o gráfico de convergência ---
    plt.figure(figsize=(12, 7))
    plt.plot(history, label='GA (Tournament)')
    plt.title(f'Convergência do AG para o Problema "{problem.name}"')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Distância Encontrada')
    
    if isinstance(best_known, (int, float)):
        plt.axhline(y=best_known, color='r', linestyle='--', label=f'Ótimo Conhecido ({best_known})')
        
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- ADICIONADO: Plota a melhor rota encontrada ---
    plot_tour(problem, best_tour, f"Melhor Rota Encontrada pelo AG (Distância: {best_dist:.2f})")