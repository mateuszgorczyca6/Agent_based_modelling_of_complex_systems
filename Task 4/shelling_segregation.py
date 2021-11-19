import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os
import multiprocessing as mp


class ShellingSegregation:
    def __init__(self):
        self.lattice_list = []
        self.iteration = 0
        self.average_happiness = 0

    def simulate(self, L, R, B, j_r, j_b, k):
        lattice = np.zeros(L * L)
        self.Js = []
        lattice[0:R] = 1
        lattice[R:B + R] = 2
        np.random.shuffle(lattice)
        lattice = lattice.reshape((L, L))
        self.lattice_list.append(lattice.copy())
        J = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if lattice[i, j] != 0:
                    neighborhood = self.neighbors(lattice, i, j, k)
                    num_neighbors = np.sum(neighborhood > 0) - 1
                    if num_neighbors == 0:
                        J[i, j] = 1  # isolated agent is happy
                    else:
                        same_type = np.sum(neighborhood == lattice[i, j]) - 1
                        J[i, j] = same_type / num_neighbors
        self.Js.append(J)
        iteration = 0

        nochange_count = 0
        min_n_of_u_blue = L ** 2
        min_n_of_u_red = L ** 2

        while np.logical_and(iteration < 1000, np.any(np.logical_or(np.logical_and(J < j_r, lattice == 1),
                                                                    np.logical_and(J < j_b, lattice == 2)))):
            iteration += 1
            unhappy_red_map = np.logical_and(lattice == 1, J < j_r)
            unhappy_blue_map = np.logical_and(lattice == 2, J < j_b)
            empty_map = lattice == 0
            possible_map = np.logical_or(np.logical_or(unhappy_blue_map, unhappy_red_map), empty_map)
            values = lattice[possible_map]
            np.random.shuffle(values)
            lattice[possible_map] = values
            self.lattice_list.append(lattice.copy())
            average_happiness = np.mean(J[lattice > 0])
            self.Js.append(J)

            J = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    if lattice[i, j] != 0:
                        neighborhood = self.neighbors(lattice, i, j, k)
                        num_neighbors = np.sum(neighborhood > 0) - 1
                        if num_neighbors == 0:
                            J[i, j] = 1  # isolated agent is happy
                        else:
                            same_type = np.sum(neighborhood == lattice[i, j]) - 1
                            J[i, j] = same_type / num_neighbors

            n_of_u_blue = np.sum(unhappy_red_map)
            n_of_u_red = np.sum(unhappy_blue_map)

            if n_of_u_red >= min_n_of_u_red and n_of_u_blue >= min_n_of_u_blue:
                nochange_count += 1
            else:
                nochange_count = 0
                min_n_of_u_red = n_of_u_red
                min_n_of_u_blue = n_of_u_blue

            print(f"step: {iteration} min red: {np.min(J[lattice == 1])} ({n_of_u_red}), min blue:"
                  f"{np.min(J[lattice == 2])} ({n_of_u_blue}), avg: {average_happiness}"
                  f"{' no improve: ' + str(nochange_count) if nochange_count>0 else ''}")

            self.iteration = iteration
            self.average_happiness = average_happiness
            if nochange_count == 10:
                return iteration, average_happiness, False

        return iteration, average_happiness, True

    def monte_carlo(self, N, L, R, B, j_r, j_b, k):
        avg_iteration, avg_average_happiness = 0, 0
        for _ in range(N):
            iteration, average_happiness, not_stopped = self.simulate(L, R, B, j_r, j_b, k)
            avg_iteration = iteration / N
            avg_average_happiness = average_happiness / N
            if not not_stopped:
                return avg_iteration, avg_average_happiness, False
        return avg_iteration, avg_average_happiness, True

    @staticmethod
    def neighbors(arr, x, y, k):
        """ Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]"""
        n = 2 * k + 1
        arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
        return arr[:n, :n]


def make_gif(fname, array_list, L, R, B, jr, jb, k):
    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    filenames = []
    images = []
    for i in range(len(array_list)):
        plt.figure(figsize=(6, 6))
        plt.imshow(array_list[i], cmap=cmap, vmin=0, vmax=2)
        plt.title('Iteration number {}'.format(i))
        plt.text(100, 20, 'L={}\n R={}\n B={}\n jr={}\n jb={}\n k={}'.format(L, R, B, jr, jb, k))
        image_name = 'graph{}.png'.format(i)
        plt.axis('off')
        plt.savefig(image_name)
        filenames.append(image_name)
        images.append(imageio.imread(image_name))
        plt.close()
    imageio.mimsave(fname, images, fps=12)
    for i in filenames:
        os.remove(i)

