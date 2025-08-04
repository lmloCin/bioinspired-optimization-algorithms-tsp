from sos_aco import SOS_ACO
import numpy as np
import matplotlib.pyplot as plt
import tsplib95
import os
import urllib.request

def plot_tour(cities, tour, title=""):
    """Helper function to plot the TSP tour."""
    ordered_cities = cities[tour]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
    plt.figure(figsize=(10, 10))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

def download_tsp_file(filename):
    """Downloads a TSP file if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{filename}"
        urllib.request.urlretrieve(url, filename)

# --- Main execution block ---
if __name__ == '__main__':
    # Define the problem to solve
    PROBLEM_NAME = 'att48' # You can change this to 'eil51', 'berlin52', etc.
    TSP_FILE = f"{PROBLEM_NAME}.tsp"
    
    # Download the file if not present
    download_tsp_file(TSP_FILE)
    
    # --- 1. Load problem from tsplib95 ---
    problem = tsplib95.load(TSP_FILE)
    
    print(f"Problema: {problem.name}")
    print(f"Número de cidades: {problem.dimension}")
    
    try:
        best_known = problem.get_best_known()
    except Exception:
        # Fallback dictionary for common problems if .get_best_known() fails
        best_solutions = {"att48": 10628, "berlin52": 7542, "eil51": 426}
        best_known = best_solutions.get(problem.name, None)
        
    if best_known:
        print(f"Solução ótima conhecida: {best_known}")
    else:
        print("Solução ótima não encontrada nos dados.")
    print("-" * 30)

    # --- 2. Set up and run the SOS-ACO solver ---
    solver = SOS_ACO(
        problem=problem,
        n_ants=10,
        n_organisms=10,
        max_iter=100,
        rho=0.1,
        q=100,
        alpha_range=(1, 5.0),
        beta_range=(1, 5.0)
    )
    
    best_tour_found, best_length_found, history = solver.solve(verbose=True)
    
    # --- 3. Print final results and comparison ---
    print("\n--- Resultados Finais do SOS-ACO ---")
    print(f"Melhor distância encontrada: {best_length_found:.2f}")
    if best_known:
        error = ((best_length_found - best_known) / best_known) * 100
        print(f"Solução ótima conhecida:    {best_known}")
        print(f"Erro percentual: {error:.2f}%")
    
    # --- 4. Plot convergence history ---
    plt.figure(figsize=(12, 7))
    plt.plot(history, label='SOS-ACO')
    plt.title(f'Convergência do SOS-ACO para o Problema "{problem.name}"')
    plt.xlabel('Iteração')
    plt.ylabel('Melhor Distância Encontrada')
    
    if best_known:
        plt.axhline(y=best_known, color='r', linestyle='--', label=f'Ótimo Conhecido ({best_known})')
        
    plt.legend()
    plt.grid(True)
    
    # --- 5. Plot the best tour found ---
    plot_tour(solver.cities, best_tour_found, f"Melhor Rota Encontrada (Distância: {best_length_found:.2f})")
    
    plt.show()