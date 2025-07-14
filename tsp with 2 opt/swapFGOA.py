# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:20:54 2023

@author: user
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import TSPDataset
import os
# Importing required libraries
from sklearn.manifold import TSNE 
# Re-defining the Particle class to be used in the AdjustedIPSO algorithm
class Goose:
    def __init__(self, dim, minx, maxx, best_position=None, updrafts=None):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.updrafts = updrafts

        if best_position is None:
            self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        else:
            self.position = self.initialize_previous_best(dim, best_position)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_score = np.inf
        self.score = np.inf

    def initialize_previous_best(self, dim, best_position):
        return np.copy(best_position)

    # def use_updraft(self):
    #     if self.updrafts:
    #         for updraft_center, strength in self.updrafts:
    #             distance = np.linalg.norm(self.position - updraft_center)
    #             if distance < 2.0:
    #                 self.velocity += strength * (updraft_center - self.position) / distance
                    
    def use_updraft(self):
        # 隨機產生上升氣流的數量
        num_updrafts = np.random.randint(1, 4)  # 假設每次有 1 到 3 個上升氣流

        # 為每個上升氣流隨機產生中心位置和強度
        for _ in range(num_updrafts):
            updraft_center = np.random.uniform(low=self.minx, high=self.maxx, size=self.dim)
            strength = np.random.uniform(low=0.1, high=2.0)  # 強度範圍可根據需要調整

            # 計算粒子與上升氣流中心的距離
            distance = np.linalg.norm(self.position - updraft_center)

            # 如果粒子在上升氣流的有效範圍內，則根據氣流強度調整速度
            if distance < 1.0:
                self.velocity += strength * (updraft_center - self.position) / distance


# Adjusted Goose Theory for High-dimensional Problems
class sFGOA:
    def __init__(self, dim, size, minx, maxx, iter, incentive_threshold, fatigue):
        self.dim = dim
        self.size = size
        self.minx = minx
        self.maxx = maxx
        self.iter = iter
        self.incentive_threshold = incentive_threshold # 激勵閾值
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)
        self.gbest_score = np.inf
        self.fatigue = fatigue
        self.best_scores = []
        self.experience = 0

    def set_leader(self):
        scores = [p.best_score for p in self.geese]
        leader_index = scores.index(min(scores))
        self.leader = self.geese[leader_index]
        self.gbest_score = self.leader.best_score
        self.gbest = np.copy(self.leader.best_position)
        
                
    def whiffling_exploitation(self, particle, experience_level):      
        direction_to_gbest = self.gbest - particle.position
   
       # Faktor acak untuk variasi (mirip whiffling)
        random_factor = np.random.rand() * np.abs(direction_to_gbest)
       
       # Semakin berpengalaman, semakin fokus pada gbest (eksploitasi)
        exploitation_component = direction_to_gbest + random_factor * (1 - experience_level)
       
       # Eksplorasi tambahan: sedikit variasi besar untuk partikel muda
        if np.random.rand() < experience_level:  # Semakin tua, semakin jarang eksplorasi
           exploration_component = np.random.uniform(-1, 1) * (1 - experience_level)
        else:
           exploration_component = 0
       
       # Gabungkan eksploitasi dan eksplorasi
        new_position = particle.position + exploitation_component + exploration_component
       
       # Batasi posisi agar tetap dalam batas pencarian
        particle.position = np.clip(new_position, self.minx, self.maxx)
        
      
    def update_geese(self, current_iter, max_iter):
        """
        Memperbarui posisi angsa (partikel) dengan keseimbangan eksplorasi-eksploitasi.
        
        Parameters:
            current_iter: iterasi saat ini
            max_iter: total maksimum iterasi
        """
        # Dynamic neighborhood radius: besar di awal, kecil di akhir
        neighborhood_radius = 1.5 * (1 - current_iter / max_iter)
    
        for particle in self.geese:
            # Hitung tingkat pengalaman berdasarkan umur atau fitness
            experience_level = particle.experience / max_iter if hasattr(particle, 'experience') else 0.5
    
            # === EKSPLORASI AKTIF UNTUK ANGSA MUDA ===
            if experience_level < 0.5:
                # Gunakan random walk atau variasi besar
                exploration_step = np.random.uniform(-1, 1, size=self.dim) * (1 - experience_level)
                particle.position += exploration_step
    
            # === WHIFFLING EKSPLOITASI UNTUK ANGSA DEWASA ===
            elif np.random.rand() < 0.2 + 0.3 * experience_level:
                self.whiffling_exploitation(particle,experience_level)
    
            # === INTERAKSI SOSIAL (Cohesion, Alignment, Separation) ===
            neighbors = [p for p in self.geese if np.linalg.norm(p.position - particle.position) < neighborhood_radius]
            if len(neighbors) > 1:
                avg_position = np.mean([p.position for p in neighbors], axis=0)
                avg_velocity = np.mean([p.velocity for p in neighbors], axis=0)
    
                # Cohesion
                particle.velocity += 0.05 * (avg_position - particle.position)
                # Alignment
                particle.velocity += 0.05 * (avg_velocity - particle.velocity)
    
                # Separation
                for neighbor in neighbors:
                    dist = np.linalg.norm(neighbor.position - particle.position)
                    if dist < neighborhood_radius / 2:
                        repulsion = (particle.position - neighbor.position) / (dist + 1e-6)
                        particle.velocity += 0.5 * repulsion
    
            # === PSO VELOCITY UPDATE ===
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia = 0.4 - 0.2 * (current_iter / max_iter)  # Inersia menurun sedikit
            cognitive = 1.5
            social = 1.5
            particle.velocity = inertia * particle.velocity \
                                + cognitive * r1 * (particle.best_position - particle.position) \
                                + social * r2 * (self.gbest - particle.position)
    
            # Tambahkan noise adaptif
            noise_factor = 0.1 * (1 - current_iter / max_iter)
            particle.position += particle.velocity + noise_factor * np.random.randn(self.dim)
    
            # Batasi posisi
            particle.position = np.clip(particle.position, self.minx, self.maxx)
    
            # Update pengalaman (jika ada atribut ini)
            if hasattr(particle, 'experience'):
                particle.experience += 1
        
            # Use Updraft
            particle.use_updraft()
    
            
    def evaluate(self, func): #要偏移就變成 (self, func, o)
        for particle in self.geese:
            particle.score = func(particle.position)
            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_position = np.copy(particle.position)
                
    def whiffling_search(self, particle, func):
        new_position = particle.position + np.random.uniform(low=-0.01, high=0.01, size=self.dim)
        new_position = np.clip(new_position, self.minx, self.maxx)
        new_score = func(new_position)
        if new_score < particle.score:
            particle.position = new_position
            particle.score = new_score
            if new_score < particle.best_score:
                particle.best_score = new_score
                particle.best_position = np.copy(new_position)
                
    def incentive(self):
        self.incentive_threshold = np.abs(np.mean([p.score for p in self.geese]) - self.gbest_score)
        for particle in self.geese:
            if particle.score < self.incentive_threshold * self.gbest_score:
                r = np.random.rand(self.dim)
                self.gbest = r * self.gbest + (1 - r) * particle.position
    
    def assist_lagging_geese(self):
        # Calculate average score of the flock
        avg_score = np.mean([p.score for p in self.geese])

        # Define a threshold to identify underperforming geese
        threshold = avg_score * 1.9  # Example: 20% worse than average

        # Loop through geese and assist the ones lagging behind
        for particle in self.geese:
            if particle.score > threshold:
                # Adjust position towards the global best or average position of better performers
                # This is a simple example, you might want to use a more sophisticated method
                particle.position += 0.1 * (self.gbest - particle.position)
                
    def visualize(self, func):
        if self.dim > 2:
            print("Visualization is only available for 2D.")
            return

        # Create a mesh grid for contour plot
        x = np.linspace(self.minx, self.maxx, 100)
        y = np.linspace(self.minx, self.maxx, 100)
        X, Y = np.meshgrid(x, y)
        # Z = func(np.array([X, Y]))
        Z = np.array([[func(np.array([x_i, y_i])) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, 20)
        plt.scatter([p.position[0] for p in self.geese], [p.position[1] for p in self.geese], color='red')

        plt.title('Fly Geese in the search space')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
    def visualize_3d(self, func):
        if self.dim != 3:
            print("Visualization is only available for 3D.")
            return
    
        # Create a 3D mesh grid for contour plot
        x = np.linspace(self.minx, self.maxx, 100)
        y = np.linspace(self.minx, self.maxx, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        
        Z = Z.reshape(X.shape)
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis', alpha=0.5)
        ax.scatter([p.position[0] for p in self.geese], 
                   [p.position[1] for p in self.geese], 
                   [func(p.position) for p in self.geese], 
                   color='r', s=100, depthshade=True)
    
        ax.set_title('Fly Geese in the search space')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        plt.show()
        
    def visualize_high_dim(self, positions, title="geese in the 2D visualized space"):
        """
        Visualize high-dimensional particle positions using t-SNE.
        """
        tsne = TSNE(n_components=2)
        reduced_data = tsne.fit_transform(positions)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='red')
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        
    def optimize(self, func):
        fatigue_counter = 0
        stagnant_counter = 0  # Initialize stagnation counter for WLS
        for i in range(self.iter):
            self.evaluate(func) #要偏移就變成 (func, o)
            self.set_leader()
            self.incentive()
            self.update_geese()
            self.best_scores.append(self.gbest_score)
            
            # Assist lagging geese at certain intervals, e.g., every 100 iterations
            if i % 100 == 0:
                self.assist_lagging_geese()
            
            if i % 100 == 0:  # Update the visualization every 100 iterations
                self.visualize(func)
                # self.visualize_high_dim([p.position for p in self.geese], title=f"Iteration {i}")
            
            # Whiffling Local Search
            if stagnant_counter >= 100:  # If no improvement in last N iterations
                for particle in self.geese:
                    self.whiffling_search(particle, func)
            
            # Fatigue Check and Reset
            if self.leader.score == self.leader.best_score:
                fatigue_counter += 1
                stagnant_counter += 1
            else:
                fatigue_counter = 0
                stagnant_counter = 0 # Reset the stagnation counter if any improvement
            
            if fatigue_counter >= self.fatigue:
                self.set_leader()
                
        return self.gbest, self.gbest_score
    



