import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. FUNÇÕES AUXILIARES E DE CÁLCULO
# --------------------------------------------------------------------------

def calculate_total_distance(tour, distance_matrix):
    """Calcula a distância total de uma rota (tour)."""
    total_distance = 0
    num_cities = len(tour)
    for i in range(num_cities):
        # Pega a distância da cidade atual para a próxima
        # O operador % garante que a última cidade se conecte à primeira
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
            # Coordenadas são 1-based no tsplib, ajustamos para 0-based
            dist = np.linalg.norm(np.array(coords[i+1]) - np.array(coords[j+1]))
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    return distance_matrix

# --------------------------------------------------------------------------
# 2. OPERADORES GENÉTICOS
# --------------------------------------------------------------------------

def ordered_crossover(parent1, parent2):
    """
    Implementa o Ordered Crossover (OX1), eficaz para o PCV.
    Garante que a rota filha seja sempre válida (sem cidades repetidas).
    """
    size = len(parent1)
    child = [-1] * size
    
    # Seleciona um trecho aleatório do primeiro pai
    start, end = sorted([random.randint(0, size - 1) for _ in range(2)])
    child[start:end+1] = parent1[start:end+1]
    
    # Preenche o restante com os genes do segundo pai, na ordem que aparecem
    pointer = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[pointer] = gene
            pointer = (pointer + 1) % size
            
    return child

def swap_mutation(tour, mutation_rate):
    """
    Implementa a mutação por troca (swap).
    Troca a posição de duas cidades aleatórias na rota.
    """
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(tour)), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# --------------------------------------------------------------------------
# 3. ESTRATÉGIAS DE SELEÇÃO DE PAIS
# --------------------------------------------------------------------------

def tournament_selection(population, fitnesses, k=5):
    """
    Seleção por Torneio: seleciona k indivíduos e retorna o melhor deles.
    """
    # Seleciona k indivíduos aleatórios da população
    selection_ix = np.random.randint(len(population), size=k)
    
    # Encontra o melhor entre os selecionados
    best_ix = -1
    best_fitness = -1
    for ix in selection_ix:
        if fitnesses[ix] > best_fitness:
            best_fitness = fitnesses[ix]
            best_ix = ix
            
    return population[best_ix]

def proportional_roulette_wheel_selection(population, fitnesses):
    """
    Seleção por Roleta Proporcional: a chance de seleção é proporcional à aptidão.
    """
    total_fitness = sum(fitnesses)
    if total_fitness == 0: # Evita divisão por zero se todas as fitness forem 0
        return random.choice(population)
        
    # Calcula as probabilidades de seleção
    probabilities = [f / total_fitness for f in fitnesses]
    
    # Usa a função `choice` do numpy que já implementa a lógica da roleta
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

def rank_based_roulette_wheel_selection(population, fitnesses):
    """
    Seleção por Roleta Baseada em Rank: a chance de seleção é baseada no ranking.
    """
    # Ordena os índices da população pela aptidão (do pior para o melhor)
    sorted_indices = np.argsort(fitnesses)
    
    # Cria os ranks (pior=1, melhor=N)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(population) + 1)
    
    # Calcula a probabilidade baseada no rank
    total_rank = sum(ranks)
    probabilities = [r / total_rank for r in ranks]
    
    # Usa a função `choice` com as probabilidades do rank
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


# --------------------------------------------------------------------------
# 4. CLASSE PRINCIPAL DO ALGORITMO GENÉTICO
# --------------------------------------------------------------------------

class GeneticAlgorithm:
    def __init__(self, cities_coords, pop_size=100, mutation_rate=0.01, crossover_rate=0.9, generations=500):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.distance_matrix = get_distance_matrix(cities_coords)
        self.num_cities = len(cities_coords)
        
        # Inicializa a população com rotas aleatórias
        self.population = self._create_initial_population()
        
    def _create_initial_population(self):
        population = []
        base_tour = list(range(self.num_cities))
        for _ in range(self.pop_size):
            tour = random.sample(base_tour, len(base_tour))
            population.append(tour)
        return population

    def run(self, selection_strategy):
        """Executa o ciclo do algoritmo genético."""
        best_overall_tour = None
        best_overall_distance = float('inf')
        history = []

        print(f"Executando com a estratégia: {selection_strategy.__name__}")

        for gen in range(self.generations):
            # 1. Avaliação (Fitness)
            # Fitness é o inverso da distância (maior é melhor)
            distances = [calculate_total_distance(tour, self.distance_matrix) for tour in self.population]
            fitnesses = [1 / d for d in distances]
            
            # 2. Guarda o melhor resultado da geração
            best_current_distance = min(distances)
            if best_current_distance < best_overall_distance:
                best_overall_distance = best_current_distance
                best_overall_tour = self.population[np.argmin(distances)]
            
            history.append(best_overall_distance)

            # Imprime o progresso
            if (gen + 1) % 50 == 0:
                print(f"> Geração {gen+1}/{self.generations}, Melhor Distância: {best_overall_distance:.2f}")

            # 3. Seleção e Reprodução
            new_population = []
            for _ in range(self.pop_size // 2):
                # Seleciona os pais usando a ESTRATÉGIA FORNECIDA
                parent1 = selection_strategy(self.population, fitnesses)
                parent2 = selection_strategy(self.population, fitnesses)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1 = ordered_crossover(parent1, parent2)
                    child2 = ordered_crossover(parent2, parent1)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutação
                child1 = swap_mutation(child1, self.mutation_rate)
                child2 = swap_mutation(child2, self.mutation_rate)
                
                new_population.extend([child1, child2])
            
            self.population = new_population
            
        return best_overall_tour, best_overall_distance, history

# --------------------------------------------------------------------------
# 5. BLOCO DE EXECUÇÃO DO EXPERIMENTO
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # Carrega um problema da TSPLIB (usado no artigo)
    # 🧠 Você pode trocar 'eil51' por 'berlin52', 'att48', etc.
    problem = tsplib95.load('att48.tsp')
    cities_coords = problem.node_coords
    best_solution = {"att48": 10628, "berlin52": 7542, "eil51": 426, "pr76": 108159}
    print(f"Problema: {problem.name}, Cidades: {len(cities_coords)}")
    
    # Tenta obter a solução ótima do problema, se não, usa nosso dicionário
    try:
        best_known = problem.get_best_known()
    except Exception:
        best_known = best_solution.get(problem.name, 'Não disponível')
        
    print(f"Solução ótima conhecida: {best_known}")
    print("-" * 30)

    # Parâmetros do AG
    POP_SIZE = 150
    GENERATIONS = 500
    MUTATION_RATE = 0.0017
    CROSSOVER_RATE = 0.9541
    
    # Lista das estratégias de seleção a serem testadas
    selection_strategies = [
        tournament_selection,
        proportional_roulette_wheel_selection,
        rank_based_roulette_wheel_selection
    ]
    
    results_history = {}

    # Executa o AG para cada estratégia
    for strategy in selection_strategies:
        ga = GeneticAlgorithm(cities_coords, pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE)
        best_tour, best_dist, history = ga.run(selection_strategy=strategy)
        
        strategy_name = strategy.__name__.replace("_selection", "").replace("_", " ").title()
        results_history[strategy_name] = history
        
        print(f"\nMelhor resultado para '{strategy_name}':")
        print(f"  Distância: {best_dist:.2f}")
        # print(f"  Rota: {best_tour}") # Descomente para ver a rota
        print("-" * 30)
        
    # Plotando os resultados
    plt.figure(figsize=(12, 7))
    for name, history in results_history.items():
        plt.plot(history, label=name)
        
    plt.title(f'Comparação das Estratégias de Seleção para o Problema "{problem.name}"')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Distância Encontrada')
    
    # Adiciona a linha da solução ótima, se disponível
    if isinstance(best_known, (int, float)):
        plt.axhline(y=best_known, color='r', linestyle='--', label=f'Ótimo Conhecido ({best_known})')
        
    plt.legend()
    plt.grid(True)
    plt.show()