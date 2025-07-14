import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import TSPDataset
import os
from typing import List, Optional
save=open('fgoa.txt', 'w')
class Goose:
    def __init__(self, dim, minx, maxx, best_position=None, updrafts=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.updrafts = updrafts
        if best_position is None:
            self.position = np.random.permutation(dim).astype(int)
        else:
            self.position = np.copy(best_position).astype(int)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf
        self.score = np.inf

    def use_updraft(self, updraft_center_list, strength_list, gbest, gbest2,gbest3,gbest4, inertia,k):
        self.position = np.array(self.position)  # pastikan numpy array
        for i, updraft in enumerate(updraft_center_list):
            updraft = np.array(updraft)  # pastikan numpy array
            distance = np.linalg.norm(self.position - updraft)
            if distance < 2:
                influence = strength_list * np.exp(-distance**2 / (2 * 1.0**2))
                velocity_change = influence * (updraft_center_list - gbest) / (distance + 1e-10)
                self.velocity += np.clip(velocity_change, float("-inf"), float("inf"))

            else:
                   
                   r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                   az= np.random.uniform(0, 2)
                   A = 2*(r1*az)-az
                   C= 2*r2
                   inertia = inertia
                   cognitive = 0.8
                   social = 0.7
                   if az <1 :
                       self.velocity = self.velocity + A * abs(C * (self.best_position - gbest))
                   else:
                       cognitive_value = np.random.uniform(0, cognitive)
                       social = 4-social
    
                       #print(cognitive, social)
                       # 若不在氣流範圍內，按標準的 PSO 更新規則更新速度
                       self.velocity =k*( inertia * self.velocity + 2 * r1 * ( gbest3 - self.best_position ) + 2 * r2 * (gbest2 - gbest))

class FGOA:
    def __init__(self, dim, size, minx, maxx, iter, incentive_threshold, fatigue, cities):
        self.dim = dim
        self.size = size
        self.minx = minx
        self.maxx = maxx
        self.iter = iter
        self.incentive_threshold = incentive_threshold
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.permutation(dim).astype(int)
        self.gbest_score = np.inf
        self.fatigue = fatigue
        self.best_scores = []
        if not isinstance(cities, list):
            raise TypeError("Expected a list of cities, but got a function or invalid type.")
        self.cities = cities
        self.num_cities = len(cities)
        self.initial_route = list(range(self.num_cities))
    
    def print_path_details(self, gbest_path, dist_matrix):
        total_distance = 0
        n = len(gbest_path)
        
        print("\n=== Best Path Segment Details ===")
        for i in range(n):
            current_city = gbest_path[i]
            next_city = gbest_path[(i + 1) % n]  # Kembali ke kota awal untuk membentuk siklus
            segment_distance = dist_matrix[current_city, next_city]
            
            print(f"Segment {i + 1}: City {current_city + 1} → City {next_city + 1} | Distance = {segment_distance:.2f}")
            total_distance += segment_distance
        
        print(f"\nTotal Best Score (Sum of Segments) = {total_distance:.2f}")
        return total_distance
    def is_hamiltonian_cycle(path, num_cities):
        return len(set(path)) == num_cities and len(path) == num_cities
    
    def compute_distance_matrix(self, cities):
        num_cities = len(cities)
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                dist = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        self.distance_matrix = dist_matrix
        return self.distance_matrix
    

    def optimized_total_distance(self, path, dist_matrix):
        dist = sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
        dist += dist_matrix[path[-1], path[0]]
        return dist
    
    def reverse(self, path, start, end, n):
        while end - start > 0:
            path[start % n], path[end % n] = path[end % n], path[start % n]
            start += 1
            end -= 1
    
    def is_path_shorter(self, graph, v1, v2, v3, v4, total_dist):
        if (graph[v1, v3] + graph[v2, v4]) < (graph[v1, v2] + graph[v3, v4]):
            total_dist -= (graph[v1, v2] + graph[v3, v4] - graph[v1, v3] - graph[v2, v4])
            return True, total_dist
        return False, total_dist
    
    def optimized_2opt(self, path, dist_matrix, improvement_threshold=0.001):
        best_path = path.copy()
        n = len(best_path)
        best_distance = self.optimized_total_distance(best_path, dist_matrix)
        improvement_factor = 3
        
        while improvement_factor > improvement_threshold:
            previous_best = best_distance
            improved = False
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if j == n - 1:
                        v1, v2, v3, v4 = best_path[i-1], best_path[i], best_path[j], best_path[0]
                    else:
                        v1, v2, v3, v4 = best_path[i-1], best_path[i], best_path[j], best_path[j+1]
                    
                    is_shorter, new_dist = self.is_path_shorter(dist_matrix, v1, v2, v3, v4, best_distance)
                    
                    if is_shorter:
                        self.reverse(best_path, i, j, n)
                        best_distance = new_dist
                        improved = True
            
            improvement_factor = best_distance / previous_best if improved else 0
        
        return best_path, best_distance
    
    def local_search(self, position, dist_matrix):
        best_position, best_distance = self.optimized_2opt(position, dist_matrix)
        return best_position
    
    def adaptive_inertia(self, iteration):
        si = 0.9
        send = iteration / self.iter
        if 0.3 <= send <= 0.9:
            return 0.8 + 0.2 * (send - 0.4) / 0.4 
        
        inertia = send + (si - send)*(1-iteration/self.iter)
        return inertia
        
        
    def update_geese(self, iteration,exploitation_prob=0.3, neighborhood_radius=2, cities=None):
        num_updrafts = np.random.randint(1, 10)  # 每次隨機生成 1 到 3 個上升氣流
        updraft_center_list = []
        strength_list = []
        # 為每個上升氣流隨機生成中心位置和強度
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=2.0)
            updraft_center_list.append(updraft_center)
            strength_list.append(strength)
        def scaled_sigmoid(h, delta):
            return 3 / (1 + np.exp(h * delta))
        h = np.linspace(-10, 10, 400)
        delta = 1  # 您可以根据需要调整 delta 的值
        for goose in self.geese:
            j = 40
            phi = 22/7
            k = (np.cos(phi / self.iter * j) * iteration + 2.5) / 4
            inertia = self.adaptive_inertia(iteration)
            goose_position = np.array(goose.position)
            goose.use_updraft(updraft_center_list, strength_list, self.gbest, self.gbest2,self.gbest3,self.gbest4,inertia,k)
            #goose.position += (goose.velocity + 0.1 * k.astype(int))
            
            goose.position = np.clip(goose.position, self.minx, self.maxx)
            goose.position = np.argsort(goose.position)

            
            score = self.cost_func(goose.position, cities)
            if score < goose.best_score:
                goose.best_score = score
                goose.best_position = np.copy(goose.position)
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = np.copy(goose.position)
               
    def evaluate(self, func, cities):
        evaluations = []
        for goose in self.geese:
            goose.score = func(goose.position, cities)
            if goose.score < goose.best_score:
                goose.best_score = goose.score
                goose.best_position = np.copy(goose.position)
    
           
            evaluations.append({
                'position': goose.position,
                'score': goose.score
            })
    
        return evaluations  
    def incentive(self):
        incentives_log = []
    
        num_incentives = int(0.5 * len(self.geese))  # Pilih 50% angsa untuk diberi insentif
        incentive_indices = np.random.choice(range(len(self.geese)), size=num_incentives, replace=False)
    
        for idx in incentive_indices:
            old_position = np.copy(self.geese[idx].position)
            r = np.random.rand(self.dim)  # Vektor acak antara 0-1
    
            # Perbarui posisi dengan dorongan menuju gbest
            self.geese[idx].position = r * self.gbest + (1 - r) * self.geese[idx].position
            new_position = self.geese[idx].position
    
            incentives_log.append({
                'idx': idx,
                'old_position': old_position,
                'new_position': new_position,
                'gbest': np.copy(self.gbest),
                'incentives_vector': r
            })
    
        return incentives_log

            
    def whiffling_exploitation(self, goose, cities=None, neighbors=None):
        proximity = (goose.best_score - self.gbest_score) / max(abs(self.gbest_score),1)
        exploration_prob = np.clip(1 - proximity, 0.1, 0.7)
        #distance_matrix = self.compute_distance_matrix(cities)
        if np.random.rand() < exploration_prob:
            goose.position = self.local_search(goose.position, self.distance_matrix)
        else:
            goose.position = self.order_crossover(goose.position, self.gbest)
        score = self.cost_func(goose.position, cities)
        if score < goose.best_score:
            goose.best_score = score
            goose.best_position = np.copy(goose.position)

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted([random.randint(0, size) for _ in range(2)])
        child = [-1] * size
        child[start:end] = parent1[start:end]
        p2_index = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[i] = parent2[p2_index]
                p2_index += 1
        return child

    

    def whiffling_search(self, particle, func, cities):
        new_position = np.copy(particle.position)
        idx1, idx2 = np.random.choice(len(new_position), size=2, replace=True)
        new_position[idx1], new_position[idx2] = new_position[idx2], new_position[idx1]
        new_score = func(new_position, cities)
        if new_score < particle.score:
            particle.position = new_position
            particle.score = new_score
            if new_score < particle.best_score:
                particle.best_score = new_score
                particle.best_position = np.copy(new_position)

    def assist_lagging_geese(self, func, cities):
        avg_score = np.mean([p.score for p in self.geese])
        threshold = avg_score * 1.3
        for particle in self.geese:
            if particle.score > threshold:
                particle.position = self.order_crossover(particle.position, self.gbest)
                particle.score = func(particle.position, cities)
                if particle.score < particle.best_score:
                    particle.best_score = particle.score
                    particle.best_position = np.round(particle.best_position + 0.01 * (self.gbest - particle.position)).astype(int)

    def set_leader(self):
        scores = [p.best_score for p in self.geese]
        leader_index = np.argmin(scores)
        self.leader = self.geese[leader_index]
        if self.leader.best_score < self.gbest_score:
            self.gbest_score = self.leader.best_score
            self.gbest = np.copy(self.leader.best_position)
    
    def change_leader(self):
        # 獲取當前的 Gbest 位置
        Gbest_position = self.leader.position
        scores = [p.best_score for p in self.geese]
        leader_index = scores.index(min(scores))
        # 隨機選擇一個粒子，將當前領導者的位置分配給它
        random_index = np.random.randint(0, len(self.geese))
        particle_position = self.geese[random_index].position
        self.geese[random_index].position = Gbest_position
        self.geese[leader_index].position = particle_position
        
    def optimize(self, func, cities):
        if not callable(func):
            raise ValueError("func must be callable, but got type {}".format(type(func)))
    
        dist_matrix = self.compute_distance_matrix(cities)
        self.cost_func = func
        fatigue_counter = 0
        stagnant_counter = 0
        history_scores = []
    
        for iter_num in range(self.iter):
            evaluations = self.evaluate(func, cities)
            scores = [p.best_score for p in self.geese]
            positions = [p.best_position for p in self.geese]
            
            sorted_indices = np.argsort(scores)
            
            gbest2 = positions[np.argsort(scores)[1]]
            gbest3 = positions[np.argsort(scores)[2]]
            
            random_idx = np.random.choice(sorted_indices[4:])  # Mulai dari rank 5
            gbest_rand_low = positions[random_idx]
            
            self.gbest2 = gbest2
            self.gbest3 = gbest3
            self.gbest4 = gbest_rand_low
            
            self.set_leader()
            incentives_log = self.incentive()
            self.update_geese(iter_num,cities=cities)
            self.best_scores.append(self.gbest_score)
            history_scores.append(self.gbest_score)
    
            # Evaluasi tiap angsa
            for idx, eval_data in enumerate(evaluations):
                print(f"Goose {idx}: Position = {eval_data['position']}, Score = {eval_data['score']}")
    
            # Logging insentif
            for log_i, log in enumerate(incentives_log):
                if isinstance(log, dict) and 'idx' in log:
                    print(f"Incentive Log {log_i}:")
                    print(f"  Goose Index     : {log['idx']}")
                    print(f"  Old Position    : {log['old_position']}")
                    print(f"  New Position    : {log['new_position']}")
                    print(f"  Global Best     : {log['gbest']}")
                    print(f"  r-Incentive Vec : {log['incentives_vector']}")
                else:
                    print(f"Warning: log[{log_i}] is invalid")
    
            # Update tiap 10/100 iterasi
            if iter_num % 10 == 0:
                self.assist_lagging_geese(func, cities)
                self.change_leader()
            if iter_num % 100 == 0:
                visualize(cities, self.gbest, self.gbest_score, iter_num)
    
            # Hitung fatigue
            if self.leader.score == self.leader.best_score:
                fatigue_counter += 5
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0
    
            if fatigue_counter >= self.fatigue:
                self.set_leader()
    
            # Cetak best path
            city_names = [str(i + 1) for i in range(len(cities))]
            best_path_cities = [city_names[idx] for idx in self.gbest]
            print(f'{self.gbest}')
            print(f'Iteration {iter_num}: Best Path: {best_path_cities}, Best Score = {self.gbest_score}')
            self.print_path_details(self.gbest, dist_matrix)
            # Simpan ke file (pastikan objek 'save' sudah didefinisikan sebelumnya)
            save.write(f"Iteration {iter_num}: Best Path: {best_path_cities}, Best Score = {self.gbest_score}\n")
    
        return self.gbest, self.gbest_score, history_scores


def visualize(cities, gbest, gbest_score, current_iter):
    plt.figure(figsize=(10, 8))
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red', label='Cities')
    if len(gbest) > 0:
        x = [cities[i][0] for i in gbest]
        y = [cities[i][1] for i in gbest]
        x.append(cities[gbest[0]][0])
        y.append(cities[gbest[0]][1])
        plt.plot(x, y, 'o-', color='blue', label=f'Best Path - Iter {current_iter}')
    plt.title(f'Best Path Visualization: Iteration {current_iter} | Best Score: {gbest_score}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i + 1), fontsize=9, ha='right')
    plt.legend()
    plt.grid(True)
    plt.show()


    
