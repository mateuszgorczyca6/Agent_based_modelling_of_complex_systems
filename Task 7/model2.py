import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import multiprocessing as mp


class SituationModel:
    def __init__(self, n, T):
        self.n = n
        self.T = T
        self.g = nx.complete_graph(n)

    def step(self, p, f):
        chosen = np.random.choice(self.g.nodes)
        if np.random.rand() < p:
            if np.random.rand() < f:
                self.current_states[chosen] *= -1
        else:
            neighbors = np.random.choice(list(self.g.neighbors(chosen)), 4, False)
            neighbors_states = self.current_states[neighbors]
            if np.abs(np.sum(neighbors_states)) == 4:
                self.current_states[chosen] = neighbors_states[0]

    def simulate(self, p, f):
        self.current_states = np.ones(self.n)
        for t in range(self.T):
            self.step(p, f)
        return np.sum(self.current_states == 1) / self.n

    def mc(self, arguments):
        nmc, p, f = arguments
        c = np.zeros(nmc)
        for mc in range(nmc):
            c[mc] = self.simulate(p, f)
        print(f'\tDone for p={p}.')
        return np.mean(c)

    def plot1(self, p_vector, f_vector, nmc, fname='plot.pdf'):
        c_vector = np.zeros((len(f_vector), len(p_vector)))
        plt.figure(figsize=(12, 8))
        for i, f in enumerate(f_vector):
            print(f'===> Working on f={f} <===')
            # for j, p in enumerate(p_vector):
            #     c_vector[i, j] = self.mc(nmc, p, f)
            arguments = list(map(lambda x: (nmc, x, f), p_vector))
            with mp.Pool() as pool:
                c_vector[i] = pool.map(self.mc, arguments)
            plt.plot(p_vector, c_vector[i, :], 'o-', label=f'f={f:.2}')
        plt.xlabel('p', fontsize=18)
        plt.ylabel('c', fontsize=18)
        title = 'Concentration of adopted $c$ in the stationary state for the situation model'
        plt.title(title, fontsize=24)
        plt.legend(fontsize=14)
        plt.savefig(fname)
        plt.show()
        return c_vector


if __name__ == "__main__":
    model = SituationModel(100, 10000)
    ps = np.arange(0, 1.05, 0.05)
    fs = np.arange(0.2, 0.6, 0.1)
    cs = model.plot1(ps, fs, 50)
