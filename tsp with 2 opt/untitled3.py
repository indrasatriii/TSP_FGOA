import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import TSPDataset
import os
from typing import List, Optional

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

    def use_updraft(self):
        num_updrafts = np.random.randint(1, 4)
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=2.0)
            distance = np.linalg.norm(self.position - updraft_center)
            if distance < 1.0:
                max_velocity_change = 1.0
                velocity_change = strength * (updraft_center - self.position) / distance
                self.velocity += np.clip(velocity_change, -max_velocity_change, max_velocity_change)

class MYFGOA:
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
    
    def optimized_2opt(self, path, dist_matrix, improvement_threshold=0.1):
        best_path = path.copy()
        n = len(best_path)
        best_distance = self.optimized_total_distance(best_path, dist_matrix)
        improvement_factor = 1
        
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

    def update_geese(self, exploitation_prob=0.3, neighborhood_radius=2, cities=None):
        for goose in self.geese:
            goose_position = np.array(goose.position)
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            az= np.random.uniform(0, 2)
            A = 2*(r1*az)-az
            C= 2*r2
            inertia = 0.6
            cognitive = 0.8
            social = 0.7
            goose.velocity = goose.velocity + A * abs(C * (goose.best_position - goose.position))
            goose.velocity = (inertia * goose.velocity + cognitive * r1 * (np.array(goose.best_position) - goose_position) + social * r2 * (np.array(self.gbest) - goose_position))
            #goose.position += goose.velocity + 0.1 * np.random.randn(self.dim) 
           #goose.position = np.argsort(goose.velocity)
            goose.position = np.clip(goose.position, self.minx, self.maxx).astype(int)
            goose.use_updraft()
            score = self.cost_func(goose.position, cities)
            if score < goose.best_score:
                goose.best_score = score
                goose.best_position = np.copy(goose.position)
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest = np.copy(goose.position)

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

    def incentive(self):
        avg_score = np.mean([np.sum(p.score) for p in self.geese])
        self.incentive_threshold = np.abs(avg_score - np.sum(self.gbest_score))
    
        for particle in self.geese:
            if np.sum(particle.score) < self.incentive_threshold * np.sum(self.gbest_score):
       
                r = np.random.rand(self.dim)
                interpolated = r * self.gbest + (1 - r) * particle.position
              
                particle.position = np.argsort(interpolated)
                particle.position = np.clip(particle.position, self.minx, self.maxx)

    def scramble_mutation(self, position: List[int]) -> List[int]:
        if len(position) <= 1:
            raise ValueError(f"Position must have more than one element for scramble mutation, but found {len(position)} elements.")
        idx1, idx2 = sorted(np.random.choice(len(position), size=2, replace=False))
        subseq = position[idx1:idx2]
        np.random.shuffle(subseq)
        mutated_position = position.copy()
        mutated_position[idx1:idx2] = subseq
        return mutated_position

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

    def evaluate(self, func, cities):
        for goose in self.geese:
            goose.score = func(goose.position, cities)
            if goose.score < goose.best_score:
                goose.best_score = goose.score
                goose.best_position = np.copy(goose.position)

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

    def optimize(self, func, cities):
        if not callable(func):
            raise ValueError("func must be callable, but got type {}".format(type(func)))
        dist_matrix = self.compute_distance_matrix(cities)
        self.cost_func = func
        fatigue_counter = 0
        stagnant_counter = 0
        history_scores = []
        for i in range(self.iter):
            for goose in self.geese:
                
                self.whiffling_exploitation(goose, cities)
            self.evaluate(func, cities)
            self.set_leader()
            self.incentive()
            self.update_geese(cities=cities)
            self.best_scores.append(self.gbest_score)
            history_scores.append(self.gbest_score)
            if i % 10 == 0:
                self.assist_lagging_geese(func, cities)
            if i % 1 == 0:
                visualize(cities, self.gbest, self.gbest_score, i)
            if stagnant_counter >= 5:
                for goose in self.geese:
                    self.whiffling_search(goose, func, cities)
            if self.leader.score == self.leader.best_score:
                fatigue_counter += 5
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0
            if fatigue_counter >= self.fatigue:
                self.set_leader()
            city_names = [str(i + 1) for i in range(len(cities))]
            best_path_cities = [city_names[idx] for idx in self.gbest]
            print(f'Iteration {i}: Best Path: {best_path_cities}, Best Score = {self.gbest_score}')
        return self.gbest, self.gbest_score, history_scores

def total_distance(path, cities):
    if path is None or len(path) == 0:
        return float('inf')
    dist = 0.0
    for i in range(len(path) - 1):
        dist += np.linalg.norm(np.array(cities[int(path[i])]) - np.array(cities[int(path[i + 1])]))
    dist += np.linalg.norm(np.array(cities[int(path[-1])]) - np.array(cities[int(path[0])]))
    return dist

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


    
