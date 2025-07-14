import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 

# Goose 類，代表每個粒子
class Goose:
    def __init__(self, dim, minx, maxx, best_position=None, updrafts=None):
        self.dim = dim  # 維度
        self.minx = minx  # 最小邊界
        self.maxx = maxx  # 最大邊界
        self.updrafts = updrafts  # 上升氣流

        # 初始化位置
        if best_position is None:
            self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        else:
            self.position = np.copy(best_position)
        # 初始化速度
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)  # 最佳位置
        self.best_score = np.inf  # 最佳分數初始化為無限大
        self.score = np.inf  # 當前分數初始化為無限大

    # 根據上升氣流更新粒子的速度
    def use_updraft(self, updraft_center_list, strength_list, inertia, cognitive, social, gbest):
        for i, updraft in enumerate(updraft_center_list):
            # 計算粒子與上升氣流中心的距離
            distance = np.linalg.norm(self.position - updraft)
            # 如果粒子在上升氣流範圍內，根據氣流強度調整速度
            if distance < 1.5:
                self.velocity += strength_list[i] * (updraft - self.position) / distance
            else:
                
                
                
                #print(cognitive, social)
                index = np.random.randint(0, len(cognitive))  # Generate a random integer index
                cognitive_value = cognitive[index]
                social = 4-social

                #print(cognitive, social)
                # 若不在氣流範圍內，按標準的 PSO 更新規則更新速度
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity = inertia * self.velocity + cognitive_value * r1 * (self.best_position - self.position) + social * r2 * (gbest - self.position)


# FGOA10 類，代表粒子群最佳化算法
class FGOA9:
    """
    激勵50%雁子，隨機更換lader位置，更新速度:如果有在上升氣流就只用上升氣流更新速度，反之按標準更新速度
    """
    def __init__(self, dim, size, minx, maxx, iter, incentive_threshold, fatigue, inertia, cognitive, social):
        self.dim = dim  # 維度
        self.size = size  # 粒子數量
        self.minx = minx  # 最小邊界
        self.maxx = maxx  # 最大邊界
        self.iter = iter  # 最大迭代次數
        #self.incentive_threshold = incentive_threshold  # 激勵閾值
        self.geese = [Goose(dim, minx, maxx) for _ in range(size)]  # 初始化粒子群
        self.gbest = np.random.uniform(low=minx, high=maxx, size=dim)  # 全局最佳位置
        self.gbest_score = np.inf  # 全局最佳分數初始化為無限大
        self.fatigue = fatigue  # 疲勞參數
        self.best_scores = []  # 用來儲存歷史最佳分數
        self.inertia = inertia  # 慣性因子
        self.cognitive = cognitive  # 認知因子
        self.social = social  # 社會因子

    # 評估每個粒子的目標函數值
    def evaluate(self, func):
        for particle in self.geese:
            particle.score = func(particle.position)  # 計算當前分數
            # 更新粒子的最佳分數和最佳位置
            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_position = np.copy(particle.position)
    
    # 設定領導者，根據最佳分數選擇最優的粒子
    def set_leader(self):
        scores = [p.best_score for p in self.geese]
        leader_index = scores.index(min(scores))
        self.leader = self.geese[leader_index]
        self.gbest_score = self.leader.best_score  # 更新全局最佳分數
        self.gbest = np.copy(self.leader.best_position)  # 更新全局最佳位置

    # 根據激勵閾值激勵部分粒子
    def incentive(self):
        # 計算激勵閾值
        #incentive_threshold = np.abs(np.mean([p.score for p in self.geese]) - self.gbest_score)
        
        # 隨機選擇50%的粒子進行激勵
        num_incentives = int(0.5 * len(self.geese))  # 選擇50%的粒子進行激勵
        incentive_indices = np.random.choice(range(len(self.geese)), size=num_incentives, replace=False)
        
        # 對選中的粒子進行激勵
        for idx in incentive_indices:
            r = np.random.rand(self.dim)  # 隨機數
            # 更新激勵後的粒子位置
            self.geese[idx].position = r * self.gbest + (1 - r) * self.geese[idx].position

    # 更新粒子的速度和位置，並考慮上升氣流
    def update_geese(self):
        num_updrafts = np.random.randint(1, 10)  # 每次隨機生成 1 到 3 個上升氣流
        updraft_center_list = []
        strength_list = []
        # 為每個上升氣流隨機生成中心位置和強度
        for _ in range(num_updrafts):
            updraft_center = 5
            strength = np.random.uniform(low=0.1, high=2.0)
            updraft_center_list.append(updraft_center)
            strength_list.append(strength)
        def scaled_sigmoid(h, delta):
            return 3 / (1 + np.exp(h * delta))
        h = np.linspace(-10, 10, 400)
        delta = 1  # 您可以根据需要调整 delta 的值

        for particle in self.geese:
                # 使用上升氣流更新粒子速度
                particle.use_updraft(updraft_center_list, strength_list, self.inertia, scaled_sigmoid(h, delta), self.cognitive,self.gbest)
                # 更新粒子的位置
                particle.position += particle.velocity + 0.1 * np.random.randn(self.dim)  # 增加隨機性
                particle.position = np.clip(particle.position, self.minx, self.maxx)  # 限制位置在邊界內

    # 幫助落後的粒子，將其位置朝向全局最佳位置調整
    def assist_lagging_geese(self):
        avg_score = np.mean([p.score for p in self.geese])  # 計算平均分數
        threshold = avg_score * 1.9  # 設定分數閾值
        for particle in self.geese:
            if particle.score > threshold:
                particle.position += 0.1 * (self.gbest - particle.position)  # 調整位置朝向全局最佳

    # 改變領導者粒子
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

    # 優化過程
    def optimize(self, func):
        for i in range(self.iter):
            self.evaluate(func)  # 評估每個粒子
            self.set_leader()  # 設定領導者
            self.incentive()  # 激勵粒子
            self.update_geese()  # 更新粒子位置和速度
            self.best_scores.append(self.gbest_score)  # 儲存歷史最佳分數
            # 每 100 次迭代幫助落後的粒子
            if i % 10 == 0:
                self.assist_lagging_geese()
            # 每 50 次迭代改變領導者
            if i % 50 == 0:
                self.change_leader()
        return self.gbest, self.gbest_score  # 返回全局最佳位置和分數
