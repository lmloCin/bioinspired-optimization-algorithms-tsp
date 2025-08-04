import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt

# --- Adicionar imports para o Ray Tune ---
import ray
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch

# Todo o seu código anterior (Funções Auxiliares, Operadores Genéticos, Estratégias de Seleção)
# permanece exatamente o mesmo. Vou omiti-lo aqui para ser breve, mas ele deve estar presente no seu arquivo.
# ... (cole aqui as funções calculate_total_distance, get_distance_matrix, ordered_crossover, 
# swap_mutation, e as 3 funções de seleção) ...

# --------------------------------------------------------------------------
# 1. FUNÇÕES AUXILIARES E DE CÁLCULO (COLE SEU CÓDIGO AQUI)
# --------------------------------------------------------------------------
def calculate_total_distance(tour, distance_matrix):
    """Calcula a distância total de uma rota (tour)."""
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        from_city = tour[i]
        to_city = tour[(i + 1) % num_cities]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance

def get_distance_matrix(coords):
    """Cria uma matriz de distâncias a partir das coordenadas das cidades."""
    num_cities = len(coords)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i, num_cities):
            dist = np.linalg.norm(np.array(coords[i+1]) - np.array(coords[j+1]))
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    return distance_matrix

# --------------------------------------------------------------------------
# 2. OPERADORES GENÉTICOS (COLE SEU CÓDIGO AQUI)
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

# --------------------------------------------------------------------------
# 3. ESTRATÉGIAS DE SELEÇÃO DE PAIS (COLE SEU CÓDIGO AQUI)
# --------------------------------------------------------------------------
def tournament_selection(population, fitnesses, k=5):
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
# 4. CLASSE PRINCIPAL DO ALGORITMO GENÉTICO (COLE SEU CÓDIGO AQUI)
# --------------------------------------------------------------------------
class GeneticAlgorithm:
    # A classe permanece a mesma, mas agora os parâmetros virão do Ray Tune
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
        for gen in range(self.generations):
            distances = [calculate_total_distance(tour, self.distance_matrix) for tour in self.population]
            fitnesses = [1 / d for d in distances]
            best_current_distance = min(distances)
            if best_current_distance < best_overall_distance:
                best_overall_distance = best_current_distance
            
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = selection_strategy(self.population, fitnesses)
                parent2 = selection_strategy(self.population, fitnesses)
                if random.random() < self.crossover_rate:
                    child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
                else:
                    child1, child2 = parent1, parent2
                new_population.extend([
                    swap_mutation(child1, self.mutation_rate),
                    swap_mutation(child2, self.mutation_rate)
                ])
            self.population = new_population
        return best_overall_distance

# --------------------------------------------------------------------------
# 5. FUNÇÃO "TREINÁVEL" PARA O RAY TUNE (CORRIGIDA)
# --------------------------------------------------------------------------

def train_ga(config, cities_coords, selection_strategy):
    """
    Esta função encapsula uma execução do AG e a torna "treinável" pelo Ray Tune.
    O dicionário 'config' contém os hiperparâmetros para esta execução específica.
    """
    # Parâmetros fixos
    POP_SIZE = 150
    GENERATIONS = 200 # Reduzido para o tuning ser mais rápido

    # Instancia o AG com os parâmetros da 'config' do Tune
    ga = GeneticAlgorithm(
        cities_coords=cities_coords,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_rate=config["mutation_rate"],
        crossover_rate=config["crossover_rate"]
    )

    # Executa o algoritmo
    best_distance = ga.run(selection_strategy=selection_strategy)
    
    # Reporta o resultado (a métrica que queremos minimizar) para o Tune
    # <--- CORREÇÃO AQUI
    train.report({"distance": best_distance})


# --------------------------------------------------------------------------
# 6. BLOCO DE EXECUÇÃO COM O RAY TUNE  <--- MODIFICAÇÃO PRINCIPAL
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Inicializa o Ray
    ray.init(num_cpus=4, ignore_reinit_error=True)

    # Carrega o problema do TSPLIB
    problem = tsplib95.load('att48.tsp')
    cities_coords = problem.node_coords
    
    # Vamos tunar os parâmetros para UMA estratégia de seleção por vez.
    # Escolha a que você quer otimizar. A de Torneio costuma ser uma boa escolha.
    chosen_selection_strategy = tournament_selection

    print(f"Iniciando o ajuste de hiperparâmetros para o problema: {problem.name}")
    print(f"Estratégia de Seleção: {chosen_selection_strategy.__name__}")
    print("-" * 30)
    
    # 1. Defina o espaço de busca dos hiperparâmetros
    search_space = {
        "mutation_rate": tune.loguniform(0.001, 0.2), # Taxa de mutação entre 0.1% e 20% em escala log
        "crossover_rate": tune.uniform(0.5, 1.0)     # Taxa de crossover entre 50% e 100%
    }
    
    # 2. Crie uma função "parcial" que fixa os argumentos que não serão tunados
    trainable_with_params = tune.with_parameters(
        train_ga,
        cities_coords=cities_coords,
        selection_strategy=chosen_selection_strategy
    )

    # 3. Configure o processo de tuning
    tuner = tune.Tuner(
        trainable_with_params,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="distance",  # A métrica que reportamos em train_ga
            mode="min",         # Queremos MINIMIZAR a distância
            num_samples=50,     # Número de combinações diferentes a serem testadas
            search_alg=HyperOptSearch(), # Algoritmo de busca (opcional, mas recomendado)
        ),
    )

    # 4. Execute o tuning
    results = tuner.fit()

    # 5. Obtenha e imprima os melhores resultados
    best_result = results.get_best_result(metric="distance", mode="min")

    print("\n" + "="*30)
    print("AJUSTE DE HIPERPARÂMETROS CONCLUÍDO!")
    print(f"Melhor distância encontrada: {best_result.metrics['distance']:.2f}")
    print("Melhor configuração de parâmetros encontrada:")
    print(f"  - Crossover Rate: {best_result.config['crossover_rate']:.4f}")
    print(f"  - Mutation Rate: {best_result.config['mutation_rate']:.4f}")
    print("="*30)