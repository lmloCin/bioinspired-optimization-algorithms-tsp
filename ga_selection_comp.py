import numpy as np
import tsplib95
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------
# 1. FUNÇÕES AUXILIARES E DE CÁLCULO (sem alterações)
# --------------------------------------------------------------------------
def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        from_city = tour[i]
        to_city = tour[(i + 1) % num_cities]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance

def get_distance_matrix(coords):
    num_cities = len(coords)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = np.linalg.norm(np.array(coords[i+1]) - np.array(coords[j+1]))
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    return distance_matrix

# --------------------------------------------------------------------------
# 2. OPERADORES GENÉTICOS E MÉTODOS DE SELEÇÃO
# --------------------------------------------------------------------------
def ordered_crossover(parent1, parent2):
    # Este operador é mantido
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

# <--- MUDANÇA 1: Mutação por Inversão, como no artigo
def inversion_mutation(tour, mutation_rate):
    """
    Implementa a mutação por inversão.
    Seleciona um trecho da rota e inverte a ordem das cidades.
    """
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(len(tour)), 2))
        sub_tour = tour[start:end+1]
        tour[start:end+1] = sub_tour[::-1] # Inverte a subsequência
    return tour

# <--- MUDANÇA 2: Tamanho do torneio (k) ajustado para 2
def tournament_selection(population, fitnesses, k=2):
    """
    Seleção por Torneio com k=2 (Torneio Binário), como no artigo.
    """
    selection_ix = np.random.randint(len(population), size=k)
    best_ix = -1
    best_fitness = -1
    for ix in selection_ix:
        if fitnesses[ix] > best_fitness:
            best_fitness = fitnesses[ix]
            best_ix = ix
    return population[best_ix]

def proportional_roulette_wheel_selection(population, fitnesses):
    # Sem alterações na lógica fundamental
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    probabilities = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

# <--- MUDANÇA 3: Seleção por Rank reescrita com a fórmula de Pressão Seletiva do artigo
def rank_based_roulette_wheel_selection(population, fitnesses, selective_pressure=1.1):
    """
    Seleção por Roleta Baseada em Rank, usando a fórmula de Pressão Seletiva (SP) do artigo.
    """
    pop_size = len(population)
    # Ordena os índices da população pela aptidão (do pior para o melhor)
    sorted_indices = np.argsort(fitnesses)
    
    # Calcula um novo valor de "fitness" (scaled rank) para cada indivíduo usando a fórmula do artigo
    scaled_ranks = np.zeros(pop_size)
    for i in range(pop_size):
        # Posição (rank) do indivíduo (1 para o pior, pop_size para o melhor)
        pos = i + 1
        # Fórmula (2) do artigo
        scaled_rank = (2 - selective_pressure) + (2 * (selective_pressure - 1) * (pos - 1) / (pop_size - 1))
        # O indivíduo no índice `sorted_indices[i]` tem o rank `i+1`
        scaled_ranks[sorted_indices[i]] = scaled_rank

    # Normaliza os scaled ranks para que somem 1, tornando-os probabilidades
    total_scaled_rank = np.sum(scaled_ranks)
    probabilities = scaled_ranks / total_scaled_rank
    
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


# --------------------------------------------------------------------------
# 3. CLASSE PRINCIPAL DO ALGORITMO GENÉTICO (GA)
# --------------------------------------------------------------------------
class GeneticAlgorithm:
    def __init__(self, cities_coords, pop_size, mutation_rate, crossover_rate, generations):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.distance_matrix = get_distance_matrix(cities_coords)
        self.num_cities = len(cities_coords)
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        population = []
        base_tour = list(range(self.num_cities))
        for _ in range(self.pop_size):
            tour = random.sample(base_tour, len(base_tour))
            population.append(tour)
        return population

    def run(self, selection_strategy):
        best_overall_distance = float('inf')
        history = []
        for gen in range(self.generations):
            distances = [calculate_total_distance(tour, self.distance_matrix) for tour in self.population]
            fitnesses = [1.0 / d for d in distances]
            
            current_best_idx = np.argmin(distances)
            if distances[current_best_idx] < best_overall_distance:
                best_overall_distance = distances[current_best_idx]
            
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
                    # <--- MUDANÇA 4: Usando a mutação por inversão
                    inversion_mutation(child1, self.mutation_rate),
                    inversion_mutation(child2, self.mutation_rate)
                ])
            self.population = new_population
            
        return best_overall_distance, history

# --------------------------------------------------------------------------
# 4. BLOCO DE EXECUÇÃO E ANÁLISE
# --------------------------------------------------------------------------
if __name__ == '__main__':
    N_RUNS = 30
    PROBLEM_INSTANCES = ['att48']

    selection_strategies = {
        "Tournament": tournament_selection,
        "Roulette Wheel": proportional_roulette_wheel_selection,
        "Rank Based": rank_based_roulette_wheel_selection
    }
    
    results = []
    
    print("Iniciando simulações com a metodologia do artigo...")
    for instance_name in PROBLEM_INSTANCES:
        problem = tsplib95.load(f'{instance_name}.tsp')
        cities_coords = problem.node_coords
        
        # <--- MUDANÇA 5: Parâmetros do AG agora são dinâmicos por instância
        pop_size = len(cities_coords) * 10
        GA_PARAMS = {
            'pop_size': pop_size,
            'generations': 500, 
            'crossover_rate': 0.9541, 
            'mutation_rate': 0.0017
        }

        print(f"\nProcessando instância: {instance_name} (População: {pop_size})")
        
        for strategy_name, strategy_function in selection_strategies.items():
            print(f"  Testando método: {strategy_name}...")
            for i in range(N_RUNS):
                start_time = time.time()
                
                algorithm_instance = GeneticAlgorithm(cities_coords=cities_coords, **GA_PARAMS)
                best_cost, history = algorithm_instance.run(selection_strategy=strategy_function)
                
                execution_time = time.time() - start_time
                
                results.append({
                    'Selection Method': strategy_name,
                    'Instance': instance_name,
                    'Run': i + 1,
                    'Best Cost': best_cost,
                    'Time': execution_time,
                    'History': history
                })

    print("\nSimulações concluídas. Gerando análise...")
    df_results = pd.DataFrame(results)

    summary_table = df_results.groupby(['Instance', 'Selection Method']).agg(
        Melhor_Custo=('Best Cost', 'min'),
        Custo_Medio=('Best Cost', 'mean'),
        Desvio_Padrao=('Best Cost', 'std'),
        Tempo_Medio=('Time', 'mean')
    ).round(2)
    
    print("\n--- Tabela Comparativa de Métodos de Seleção (Metodologia do Artigo) ---")
    print(summary_table)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_results, x='Instance', y='Best Cost', hue='Selection Method')
    plt.title('Distribuição dos Resultados por Método de Seleção (30 Execuções)', fontsize=16)
    plt.ylabel('Melhor Custo Encontrado')
    plt.xlabel('Instância do Problema')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    for instance_name in PROBLEM_INSTANCES:
        plt.figure(figsize=(12, 8))
        for strategy_name in selection_strategies.keys():
            subset = df_results[(df_results['Instance'] == instance_name) & (df_results['Selection Method'] == strategy_name)]
            histories = np.array(subset['History'].tolist())
            mean_history = np.mean(histories, axis=0)
            plt.plot(mean_history, label=strategy_name)
        
        plt.title(f'Convergência Média por Método de Seleção para "{instance_name}"', fontsize=16)
        plt.xlabel('Geração')
        plt.ylabel('Custo Médio da Melhor Solução')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()