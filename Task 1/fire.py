import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os
from itertools import product


class Fire:
    """Class of burning forest on grid Simulator."""

    def __init__(self, L: int, p: float):
        """Init function of burning forest Simulator.

        :argument
        L: int
        > the length of the side of the grid
        p: float from 0 to 1
        > probability that the tree will appear in particular cell of the grid"""
        self.L = L
        self.p = p

    def iteration(self):
        """One step of simulation."""
        neighborhood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        self.lattice_copy = self.lattice.copy()
        for i in range(self.L):
            for j in range(self.L):
                if self.lattice[i, j] == 2:
                    for dx, dy in neighborhood:
                        if 0 <= i + dx < self.L and 0 <= j + dy < self.L:
                            if self.lattice[i + dx, j + dy] == 1:
                                self.lattice_copy[i + dx, j + dy] = 2
                    self.lattice_copy[i, j] = 3

    def simulation(self):
        """Reset whole simulation and simulate it."""
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

    def make_gif(self, fname: str = "fire.gif"):
        """Create gif from simulation and saves it as fname.

        :argument
        fname: str
        > name of created gif"""
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
        imageio.mimsave(fname, images, fps=7)
        for i in filenames:
            os.remove(i)

    def find_biggest_cluster(self, plot: bool = False) -> int:
        """Function is labeling and finding the biggest cluster is self.lattice

        :argument
        plot: bool
        > if True then the the final, labeled matrix and biggest cluster size will be printed
        > default: False

        :return
        biggest_cluster: int
        > size of biggest cluster in self.lattice"""

        # labeling
        largest_label = 0
        label = np.zeros((self.L, self.L))
        for P in product(range(self.L), repeat=2):
            P_label = self.lattice[P[0]][P[1]]
            L = (P[0] - 1, P[1])
            U = (P[0], P[1] - 1)
            if P_label == 3:
                if L[0] >= 0 and self.lattice[L[0]][L[1]] == 3:
                    L_label = label[L[0]][L[1]]
                    if U[1] >= 0 and self.lattice[U[0]][U[1]] == 3:
                        U_label = label[U[0]][U[1]]
                        if not L_label == U_label:
                            label[label == U_label] = L_label
                        label[P[0]][P[1]] = L_label
                    else:
                        label[P[0]][P[1]] = L_label
                elif U[1] >= 0 and self.lattice[U[0]][U[1]] == 3:
                    U_label = label[U[0]][U[1]]
                    label[P[0]][P[1]] = U_label
                else:
                    largest_label += 1
                    label[P[0]][P[1]] = largest_label

        # counting the biggest cluster size
        biggest_cluster = 0
        for num in range(1, largest_label + 1):
            cluster_size = np.sum(label == num)
            if cluster_size > biggest_cluster:
                biggest_cluster = cluster_size

        # plotting if desired
        if plot:
            print("The forest at the end:")
            print(self.lattice)
            print("The labeled clusters")
            print(label)
            print("Biggest cluster is ", biggest_cluster)

        return biggest_cluster

    def do_MC(self, N_probes: int):
        """Creates N_probes of monte carlo simulation and returns probability that the fire will go to the other
        side of grid.

        :argument
        N_probes: int
        > Number of Monte Carlo probes to be made.

        :return
        fire_to_the_other_end_count: float
        > probability that the fire will go to the other side of a grid.
        biggest_clusters: float
        > the average size of biggest cluster of burned trees after the simulation is finished.
        """

        fire_to_the_other_end_count = 0
        biggest_clusters = 0
        for n in range(N_probes):
            self.simulation()
            if 3 in self.lattice[:, -1]:
                fire_to_the_other_end_count += 1
                biggest_clusters += self.find_biggest_cluster()
        fire_to_the_other_end_count /= N_probes
        biggest_clusters /= N_probes

        return fire_to_the_other_end_count, biggest_clusters
