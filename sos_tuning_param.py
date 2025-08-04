from sos_aco_single_run import download_tsp_file
from sos_aco import SOS_ACO
import tsplib95
import time

# Import para Ray Tune
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment.trial import Trial

def short_name_creator(trial: Trial):
    return f"{trial.experiment_tag}_{trial.trial_id}"

# --- Função Objetivo para o Ray Tune ---
def funcao_objetivo(config):
    """
    Esta função é chamada pelo Ray Tune para cada 'trial' (tentativa).
    """
    # Carrega o problema uma vez por 'trial'
    problem = tsplib95.load(config["problem_file"])
    
    # Cria o solver com os hiperparâmetros da 'config' atual
    solver = SOS_ACO(
        problem=problem,
        n_ants=config["n_ants"],
        n_organisms=config["n_organisms"],
        max_iter=config["max_iter"],
        rho=config["rho"],
        q=config["q"],
        alpha_range=(config["alpha_start"], config["alpha_end"]),
        beta_range=(config["beta_start"], config["beta_end"])
    )
    
    # Executa o solver
    final_distance = solver.solve(verbose=False) # verbose=False para não poluir o output
    
    # Reporta o resultado para o Ray Tune
    tune.report(final_distance=final_distance)

# --- Bloco Principal: Configuração e Execução do Tuning ---
if __name__ == '__main__':
    PROBLEM_NAME = 'att48'
    TSP_FILE = f"{PROBLEM_NAME}.tsp"
    download_tsp_file(TSP_FILE)

    # 1. Definição do Espaço de Busca de Hiperparâmetros
    search_space = {
        "problem_file": TSP_FILE,
        "max_iter": 150, # Fixo para o tuning, para ser mais rápido
        "n_ants": tune.qrandint(10, 40, 5),
        "n_organisms": tune.qrandint(10, 30, 5),
        "rho": tune.uniform(0.05, 0.5),
        "q": tune.choice([10, 100, 500, 1000, 2000]),
        "alpha_start": tune.uniform(0.1, 1.0),
        "alpha_end": tune.uniform(2.0, 8.0),
        "beta_start": tune.uniform(0.5, 2.0),
        "beta_end": tune.uniform(5.0, 15.0),
    }

    # 2. Algoritmo de Busca e Agendador
    search_alg = HyperOptSearch(metric="final_distance", mode="min")
    scheduler = ASHAScheduler(metric="final_distance", mode="min")
    
    # 3. Execução do Experimento de Tuning
    print("🚀 Iniciando o fine-tuning dos hiperparâmetros com Ray Tune...")
    start_time = time.time()
    
    analysis = tune.run(
        funcao_objetivo,
        storage_path="C:/ray_tmp",
        trial_dirname_creator=short_name_creator,
        config=search_space,
        num_samples=50,  # Número de combinações diferentes a serem testadas
        resources_per_trial={"cpu": 1},
        search_alg=search_alg,
        scheduler=scheduler,
        verbose=1 # Mostra uma tabela de progresso
    )
    
    end_time = time.time()
    print(f"\nFine-tuning concluído em {end_time - start_time:.2f} segundos.")
    
    # 4. Análise dos Resultados
    best_config = analysis.get_best_config(metric="final_distance", mode="min")
    best_result = analysis.get_best_trial(metric="final_distance", mode="min").last_result
    
    print("\n--- ✅ Melhor Configuração Encontrada ---")
    print(f"Melhor distância obtida durante o tuning: {best_result['final_distance']:.2f}")
    print("Hiperparâmetros ótimos:")
    for key, value in best_config.items():
        if key != 'problem_file' and key != 'max_iter':
             # Arredonda valores float para facilitar a leitura
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")