import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SituationModel:
    def __init__(self, n, T, f):
        self.n = n
        self.T = T
        self.f = f
        self.g = nx.complete_graph(n)
    def step(self, p):
        chosen = np.random.choice(self.g.nodes)
        if np.random.rand()<p:
            if np.random.rand()<self.f:
                self.current_states[chosen] *= -1
        else:
            neighbors = np.random.choice(list(self.g.neighbors(chosen)),4,False)
            neighbors_states = self.current_states[neighbors]
            if np.abs(np.sum(neighbors_states))==4:
                self.current_states[chosen] = neighbors_states[0]
    def simulate(self, p):
        self.current_states = np.ones(self.n)
        for t in range(self.T):
            self.step(p)
        return np.sum(self.current_states==1)/self.n
    def mc(self, nmc, p):
        c = np.zeros(nmc)
        for mc in range(nmc):
            c[mc] = self.simulate(p)
        return np.mean(c)
    def plot1(self, p_vector, nmc):
        c_vector = np.zeros(p_vector.shape)
        for i, p in enumerate(p_vector):
            print(f'doing for p={p}')
            c_vector[i] = self.mc(nmc, p)
            
        plt.figure(figsize=(12,8))    
        plt.plot(p_vector, c_vector, 'o-')
        plt.xlabel('p')
        plt.ylabel('c')
        plt.show()
        return c_vector

if __name__ == "__main__":
    
    model = SituationModel(100, 10000, 0.5)
    ps = np.arange(0,1.05,0.05)
    cs = model.plot1(ps,50)
