import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

class DynamicFireflyAlgorithm:
    def __init__(self, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.initial_pop_size = pop_size  # Başlangıç popülasyon boyutu
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)
    
    def run(self, function, dim, lb, ub, max_evals):
        pop_size = self.initial_pop_size
        fireflies = self.rng.uniform(lb, ub, (pop_size, dim))
        intensity = np.apply_along_axis(function, 1, fireflies)
        best = np.min(intensity)

        evaluations = 1
        new_alpha = self.alpha
        search_range = ub - lb

        best_intensities = []

        iteration_counter = 0

        while evaluations <= max_evals:
            new_alpha *= 0.97
            iteration_counter += 1

            # Her 5 iterasyonda bir popülasyon artırılıyor
            if iteration_counter % 5 == 0:
                # list_em=[]
                # for i in fireflies:
                #     list_em.append(function(i[0]))
                # min_value=min(list_em)
                # min_index=list_em.index(min_value)
                # new_firefly=fireflies[min_index]
                pop_size += 1
                new_firefly = self.rng.uniform(lb, ub, (1, dim))  # Rasgele
                fireflies = np.vstack((fireflies, new_firefly))
                intensity = np.append(intensity, function(new_firefly[0]))

            # Tüm popülasyonu güncelle
            for i in range(pop_size):
                for j in range(pop_size):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = function(fireflies[i])
                        
            evaluations += 1
            best = np.min(intensity)
            best_intensities.append(best)

        return best_intensities, pop_size

# Sphere fonksiyonu
def sphere(x):
    return np.sum(x**2)

# Rastrigin function
def rastrigin(X):
    A = 10
    n = len(X)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# Modified multifunc to run FA multiple times and return a dictionary
def multifunc(function, x=1):
    results = {}
    for i in range(x):
        result = function()  # Call the passed function
        # Store result as a dictionary
        results[f"Run {i+1}"] = {
            "best_intensities": result[0],
            "pop_size": result[1]
        }

    return results

# Tanımlamaları yapıyoruz. 
FA = DynamicFireflyAlgorithm(pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None)

# Wrap FA.run in a function to be passed to multifunc with rastrigin
def run_FA_sphere():
    return FA.run(function=sphere, dim=10, lb=-5, ub=5, max_evals=200)

def run_FA_rastrigin():
    return FA.run(function=rastrigin, dim=10, lb=-5, ub=5, max_evals=200)

# Run the FA algorithm 5 times using multifunc
results = multifunc(run_FA_sphere, x=4)#•formülün kaç kere çalışacağı-fuction belirle


# Print the final results
print(results)


def visualize_results(results):
    num_runs = len(results)
    fig, axs = plt.subplots(num_runs, 1, figsize=(8, 10))

    # Create subplots for each run
    for i, (key, value) in enumerate(results.items()):
        axs[i].plot(value['best_intensities'], label=f'{key} - Best Intensities')
        axs[i].set_title(f'{key} - Best Intensities Over Time')
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('Best Intensity')
        axs[i].grid(True)
        axs[i].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
    

# Visualize the result using subplots
visualize_results(results)



