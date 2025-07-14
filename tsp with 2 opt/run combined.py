from combination import FGOA
from untitled3 import MYFGOA
from swapFGOA import sFGOA
import numpy as np
import matplotlib.pyplot as plt
import os 
from datasets import TSPDataset

def total_distance(path, cities):
    if path is None or len(path) == 0:
        return float('inf')
    dist = 0.0
    for i in range(len(path) - 1):
        dist += np.linalg.norm(np.array(cities[int(path[i])]) - np.array(cities[int(path[i + 1])]))
    dist += np.linalg.norm(np.array(cities[int(path[-1])]) - np.array(cities[int(path[0])]))
    return dist

def visualize(cities, gbest, gbest_score, current_iter, label_suffix=''):
    plt.figure(figsize=(10, 8))
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red', label='Cities')
    if len(gbest) > 0:
        x = [cities[i][0] for i in gbest]
        y = [cities[i][1] for i in gbest]
        x.append(cities[gbest[0]][0])
        y.append(cities[gbest[0]][1])
        plt.plot(x, y, 'o-', color='blue', label=f'Best Path {label_suffix} - Iter {current_iter}')
    plt.title(f'Best Path Visualization {label_suffix}: Iteration {current_iter} | Best Score: {gbest_score}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i + 1), fontsize=9, ha='right')
    plt.legend()
    plt.grid(True)
    plt.show()

def cities():
    file_path = 'D:/tsp/github/tsp with 2 opt/dataset/wi29.tsp'
    if not os.path.exists(file_path):
        print(f"ERROR: File tidak ditemukan di {file_path}")
        return []
    dataset = TSPDataset()
    dataset.read(file_path)
    return dataset.positions

# Load cities
cities_data = cities()
dim = len(cities_data)
size = 100
minx = 0
maxx = dim - 1
fatigue = 2

# --- Run FGOA ---
iter_fgoa = 1000
incentive_threshold_fgoa = int(0.2 * dim) if iter_fgoa % 2 == 1 else int(0.1 * dim)
fgoa = FGOA(dim, size, minx, maxx, iter_fgoa, incentive_threshold_fgoa, fatigue, cities_data)
best_path_fgoa, best_score_fgoa, history_scores_fgoa = fgoa.optimize(total_distance, cities_data)

# --- Run MYFGOA ---
iter_myfgoa = iter_fgoa
incentive_threshold_myfgoa = int(0.2 * dim) if iter_myfgoa % 2 == 1 else int(0.1 * dim)
myfgoa = MYFGOA(dim, size, minx, maxx, iter_myfgoa, incentive_threshold_myfgoa, fatigue, cities_data)
best_path_myfgoa, best_score_myfgoa, history_scores_myfgoa = myfgoa.optimize(total_distance, cities_data)

#-- Run sFGOA --
iter_sfgoa = 1000
incentive_threshold_sfgoa = int(0.2 * dim) if iter_sfgoa % 2 == 1 else int(0.1 * dim)
sfgoa = sFGOA(dim, size, minx, maxx, iter_sfgoa, incentive_threshold_sfgoa, fatigue, cities_data)
best_path_sfgoa, best_score_sfgoa, history_scores_sfgoa = sfgoa.optimize(total_distance, cities_data)
print(f'[FGOA] Best path: {best_path_fgoa}, Best score: {best_score_fgoa}')
print(f'[FGOA] Incentive Threshold: {incentive_threshold_fgoa}')
print(f'[sFGOA] Best path: {best_path_sfgoa}, Best score: {best_score_sfgoa}')
print(f'[sFGOA] Incentive Threshold: {incentive_threshold_sfgoa}')
print(f'[MYFGOA] Best path: {best_path_myfgoa}, Best score: {best_score_myfgoa}')
print(f'[MYFGOA] Incentive Threshold: {incentive_threshold_myfgoa}')

# Optional: visualize best path
visualize(cities_data, best_path_fgoa, best_score_fgoa, iter_fgoa, label_suffix='FGOA')
visualize(cities_data, best_path_sfgoa, best_score_sfgoa, iter_sfgoa, label_suffix='sFGOA')
visualize(cities_data, best_path_myfgoa, best_score_myfgoa, iter_myfgoa, label_suffix='MYFGOA')
# --- Combined Plot ---
plt.figure(figsize=(12, 6))
if len(history_scores_fgoa) > 0:
    plt.plot(history_scores_fgoa, label='CombinedFGOA')
if len(history_scores_sfgoa) > 0:
    plt.plot(history_scores_sfgoa, label='CombinedFGOA')
if len(history_scores_myfgoa) > 0:
    plt.plot(history_scores_myfgoa, label='MYFGOA')
plt.title("Comparison of Best Scores over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Best Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
A = 2*(r1*az) - az
  = 2 * ([0.4*1.3, 0.7*1.3, 0.2*1.3, 0.9*1.3, 0.5*1.3]) - 1.3
  = 2 * [0.52, 0.91, 0.26, 1.17, 0.65] - 1.3
  = [1.04, 1.82, 0.52, 2.34, 1.3] - 1.3
  = [-0.26, 0.52, -0.78, 1.04, -0.0]

C = 2 * r2 = [1.2, 0.2, 1.6, 0.6, 0.8]
'''
