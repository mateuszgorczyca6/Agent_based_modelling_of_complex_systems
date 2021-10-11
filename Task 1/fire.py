import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os


class Fire:
    def __init__(self, L, p):
        self.L = L
        self.p = p

    def iteration(self):
        neighborhood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        self.lattice_copy = self.lattice.copy()
        for i in range(self.L):
            for j in range(self.L):
                if self.lattice[i, j] == 2:
                    for dx, dy in neighborhood:
                        if i + dx >= 0 and i + dx < self.L and j + dy >= 0 and j + dy < self.L:
                            if self.lattice[i + dx, j + dy] == 1:
                                self.lattice_copy[i + dx, j + dy] = 2
                    self.lattice_copy[i, j] = 3

    def simulation(self):
        self.lattice = np.random.choice([0, 1], self.L ** 2, True, [1 - self.p, self.p]).reshape(self.L, self.L)
        self.num_of_trees = self.lattice.sum()
        self.lattice[
            self.lattice[:, 0] == 1, 0] = 2  # fire starts on the left edge of the forest and spreading to the rigth
        self.arrays = []
        self.iteration()
        self.arrays.append(self.lattice_copy)
        while not np.all(self.lattice == self.lattice_copy):
            self.lattice = self.lattice_copy.copy()
            self.iteration()
            self.arrays.append(self.lattice)

    def make_gif(self):
        cmap = colors.ListedColormap(['yellow', 'green', 'red', 'black'])
        filenames = []
        images = []
        for i in range(len(self.arrays)):
            plt.figure(figsize=(6, 6))
            plt.imshow(self.arrays[i], cmap=cmap, vmin=0, vmax=3)
            plt.title('p={}, L={}\n t={}'.format(self.p, self.L, i))
            image_name = 'graph{}.png'.format(i)
            plt.axis('off')
            plt.savefig(image_name)
            filenames.append(image_name)
            images.append(imageio.imread(image_name))
            plt.close()
        imageio.mimsave('fire.gif', images, fps=7)
        for i in filenames:
            os.remove(i)

    def do_MC(self, N_probes):
        self.fire_to_the_other_end_count = 0
        self.biggest_clusters_sum = 0

        for n in range(N_probes):
            self.simulation()
            if 3 in self.lattice[:, -1]:
                self.fire_to_the_other_end_count += 1
        self.fire_to_the_other_end_count /= N_probes

        return self.fire_to_the_other_end_count