import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os
from os.path import exists
from IPython.display import clear_output
from scipy import ndimage
import json


class ShellingSegregation:
    def __init__(self):
        self.lattice_list = []
        self.iteration = 0
        self.average_happiness = 0

    def simulate(self, L, R, B, j_r, j_b, k, save_history=True, value=None, MC_N=1, verbose=2):
        # starting setup
        lattice = np.zeros(L ** 2).astype(int)
        lattice[0:R] = 1
        lattice[R:B + R] = 2
        np.random.shuffle(lattice)
        lattice = np.reshape(lattice, (L, L))

        if save_history:
            self.lattice_list = [lattice.copy()]

        def happy(neighbours):
            center = neighbours[k * (2 * k + 1) + k]
            if center > 0:
                num_all = np.sum(neighbours > 0) - 1
                if num_all == 0:
                    return 1
                num_good = np.sum(neighbours == center) - 1
                return num_good / num_all
            else:
                return 0

        def get_J(lattice, j_r, j_b):
            J = ndimage.generic_filter(lattice, happy, size=2 * k + 1, output=float, mode="wrap")
            unhappy_red = np.logical_and(lattice == 1, J < j_r)
            unhappy_blue = np.logical_and(lattice == 2, J < j_b)
            return J, unhappy_red, unhappy_blue

        J, unhappy_red, unhappy_blue = get_J(lattice, j_r, j_b)
        n_unhappy_red = np.sum(unhappy_red)
        n_unhappy_blue = np.sum(unhappy_blue)
        avg_happiness = np.mean(J[lattice > 0])

        step = 0
        nochange_count = 0
        min_n_of_u_blue = L ** 2
        min_n_of_u_red = L ** 2

        while n_unhappy_red > 0 or n_unhappy_blue > 0:
            # moving peoples
            empty = lattice == 0
            places = np.logical_or(np.logical_or(unhappy_red, unhappy_blue), empty)
            values = lattice[places]
            if len(values)>1:
                idxs = np.arange(len(values))
                shuffler = np.random.permutation(len(values))
                while np.any(idxs==shuffler):
                    shuffler = np.random.permutation(len(values))
            
            lattice[places] = values[shuffler]

            if save_history:
                self.lattice_list.append(lattice.copy())

            J, unhappy_red, unhappy_blue = get_J(lattice, j_r, j_b)
            n_unhappy_red = np.sum(unhappy_red)
            n_unhappy_blue = np.sum(unhappy_blue)
            avg_happiness = np.mean(J[lattice > 0])

            # stopping when no improvement
            step += 1

            if n_unhappy_red >= min_n_of_u_red and n_unhappy_blue >= min_n_of_u_blue:
                nochange_count += 1
            else:
                nochange_count = 0
                min_n_of_u_red = n_unhappy_red
                min_n_of_u_blue = n_unhappy_blue

            if verbose == 2 or (verbose == 1.5 and step % 10 == 0):
                clear_output()
                if value is not None:
                    print("Value: " + str(value) + ", MC: " + str(MC_N))
                print(f"step: {step}, unhappy red: {n_unhappy_red}, unhappy blue: {n_unhappy_blue}, "
                      f"avg: {avg_happiness}"
                      f"{', no improvement since ' + str(nochange_count) + ' steps' if nochange_count > 0 else ''}")

            self.iteration = step
            self.average_happiness = avg_happiness
            if nochange_count == 10 or step >= 1000:
                return step, avg_happiness, False

        if verbose > 0:
            clear_output()
            if value is not None:
                print("Value: " + str(value) + ", MC: " + str(MC_N))
            print(f"step: {step}, unhappy red: {n_unhappy_red}, unhappy blue: {n_unhappy_blue}, "
                  f"avg: {avg_happiness}"
                  f"{', no improvement since ' + str(nochange_count) + ' steps' if nochange_count > 0 else ''}")

        return step, avg_happiness, True

    def monte_carlo(self, N, L, R, B, j_r, j_b, k, save_history=True, value=None, verbose=1.5):
        avg_iteration, avg_average_happiness = 0, 0
        n = 0
        stopped_in_the_row = 0
        while n < N:
            iteration, average_happiness, not_stopped = self.simulate(L, R, B, j_r, j_b, k, save_history, value, n,
                                                                       verbose)
            if verbose == 1:
                clear_output()
                if value is not None:
                    print("Value: " + str(value) + ", MC: " + str(n))
            n += 1
            if not not_stopped:
                stopped_in_the_row += 1
                n -= 1
            else:
                stopped_in_the_row = 0
                avg_iteration += iteration / N
                avg_average_happiness += average_happiness / N

            if stopped_in_the_row == 5:
                return avg_iteration, avg_average_happiness, False

        return avg_iteration, avg_average_happiness, True

    @staticmethod
    def neighbors(arr, x, y, k):
        """ Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]"""
        n = 2 * k + 1
        arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
        return arr[:n, :n]


def make_gif(fname, array_list, L, R, B, jr, jb, k):
    print("Creating gif...")
    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    filenames = []
    images = []
    for i in range(len(array_list)):
        plt.figure(figsize=(8, 8))
        plt.imshow(array_list[i], cmap=cmap, vmin=0, vmax=2)
        plt.title('Iteration number {}'.format(i), fontsize=18)
        plt.text(100, 20, 'L={}\n R={}\n B={}\n jr={}\n jb={}\n k={}'.format(L, R, B, jr, jb, k), fontsize=12)
        image_name = 'graph{}.png'.format(i)
        plt.axis('off')
        plt.savefig(image_name)
        filenames.append(image_name)
        images.append(imageio.imread(image_name))
        plt.close()
    imageio.mimsave(fname, images, fps=round(len(array_list)/15))
    for i in filenames:
        os.remove(i)
    print("Gif created")


def plot(segregation, nr, fname, MC_N, L, R, B, jr, jb, k, cont=False, verbose=1.5):
    if exists(fname):
        with open(fname) as f:
            data = json.load(f)
        Xs = data["Xs"]
        Ys = data["Ys"]
    if cont or not (exists(fname)):
        if cont and exists(fname):
            X = Xs[-1]
        else:
            Xs = []
            Ys = []
            X = 0
        while X < {1: 10000, 2: 1, 3: 5}[nr]:
            X += {1: 50, 2: 0.05, 3: 1}[nr]
            if nr == 1:
                Y, _, not_stopped = segregation.monte_carlo(MC_N, L, int(X / 2), int(X / 2), jr, jb, k, False, f"N={X}",
                                                            verbose)
            elif nr == 2:
                _, Y, not_stopped = segregation.monte_carlo(MC_N, L, R, B, X, X, k, False, f"j_r=j_b={X}", verbose)
            elif nr == 3:
                _, Y, not_stopped = segregation.monte_carlo(MC_N, L, R, B, jr, jb, X, False, f"k={X}", verbose)
            if not not_stopped:
                break
            Xs.append(X)
            Ys.append(Y)

            data = {"Xs": Xs, "Ys": Ys}

            with open(fname, "w") as f:
                json.dump(data, f)

    plt.plot(Xs, Ys)
