from sos_aco_single_run import download_tsp_file
from sos_aco import SOS_ACO
import tsplib95
import time
import numpy as np
import matplotlib.pyplot as plt


# --- Bloco Principal: Geração e Análise dos Experimentos ---
if __name__ == '__main__':
    PROBLEM_NAME = 'burma14' # Problema usado no artigo
    TSP_FILE = f"{PROBLEM_NAME}.tsp"
    N_RUNS = 30 # Número de execuções para robustez estatística
    
    download_tsp_file(TSP_FILE)
    problem = tsplib95.load(TSP_FILE)
    
    print(f"Iniciando análise de desempenho do SOS-ACO para o problema: {problem.name}")
    print(f"Número de execuções independentes: {N_RUNS}")
    print("-" * 50)
    
    # --- Configuração do solver SOS-ACO com os parâmetros do artigo ---
    # Parâmetros de acordo com a Tabela 2 do artigo
    solver = SOS_ACO(
        problem=problem, 
        n_ants=10, 
        n_organisms=10, 
        max_iter=100, 
        rho=0.1, 
        q=1000, 
        alpha_range=(1,5), 
        beta_range=(1,5)
    )
    
    # Lista para armazenar o melhor fitness de cada uma das 30 execuções
    results_sos_aco = []
    
    # --- Loop de Execução dos Experimentos ---
    start_time = time.time()
    for i in range(N_RUNS):
        # Cada chamada a .solve() é uma execução completa e independente
        _, final_fitness, _ = solver.solve()
        results_sos_aco.append(final_fitness)
        print(f"  Execução {i+1}/{N_RUNS} -> Melhor Fitness Final: {final_fitness:.2f}")
    end_time = time.time()
    print(f"\nAnálise concluída em {end_time - start_time:.2f} segundos.")
        
    print("\n--- Análise Estatística dos Resultados do SOS-ACO ---")
    
    # --- Cálculo e Impressão das Métricas ---
    best_overall_fitness = np.min(results_sos_aco)
    mean_final_fitness = np.mean(results_sos_aco)
    std_dev_final_fitness = np.std(results_sos_aco)
    
    print(f"  - Melhor Fitness Absoluto (Overall Best): {best_overall_fitness:.2f}")
    print(f"  - Média do Fitness Final:               {mean_final_fitness:.2f}")
    print(f"  - Desvio Padrão do Fitness Final:       {std_dev_final_fitness:.2f}")
    
    # --- Geração do Histograma para Análise de Distribuição ---
    plt.figure(figsize=(12, 7))
    plt.hist(results_sos_aco, bins='auto', alpha=0.7, rwidth=0.85, label='Frequência dos Resultados')
    
    # Adiciona linhas verticais para Média e Melhor Resultado
    plt.axvline(mean_final_fitness, color='red', linestyle='dashed', linewidth=2, label=f'Média: {mean_final_fitness:.2f}')
    plt.axvline(best_overall_fitness, color='green', linestyle='dashed', linewidth=2, label=f'Melhor: {best_overall_fitness:.2f}')
    
    plt.title(f'Distribuição dos Resultados do SOS-ACO para "{problem.name}" ({N_RUNS} Execuções)')
    plt.xlabel('Melhor Distância Final (Fitness)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plt.show()