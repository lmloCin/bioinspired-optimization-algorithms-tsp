import matplotlib.pyplot as plt
import numpy as np
from sos_aco import SOS_ACO

def plot_tour(cities, tour, title=""):
    """Helper function to plot the TSP tour."""
    ordered_cities = cities[tour]
    # Add the starting city to the end to close the loop
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
    
    plt.figure(figsize=(8, 8))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- 1. Generate a set of random cities ---
    NUM_CITIES = 50
    random_cities = np.random.rand(NUM_CITIES, 2) * 100

    # --- 2. Set up and run the SOS-ACO solver ---
    solver = SOS_ACO(
        cities=random_cities,
        n_ants=10,          # m
        n_organisms=10,     # N
        max_iter=100,
        rho=0.1,
        q=100,
        alpha_range=(1, 5.0), # Search space for alpha
        beta_range=(1, 5.0)  # Search space for beta
    )
    
    best_tour_found, best_length_found = solver.solve()

    # --- 3. Print and plot the results ---
    print(f"\nBest tour found: {best_tour_found}")
    print(f"Length of best tour: {best_length_found:.2f}")
    
    # Plot the resulting tour
    plot_tour(random_cities, best_tour_found, f"SOS-ACO Best Tour (Length: {best_length_found:.2f})")