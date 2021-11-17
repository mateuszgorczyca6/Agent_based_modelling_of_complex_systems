import numpy as np


import numpy as np


class Periodic_lattice:
    def __init__(self, L):
        self.L = L
        self.arr = np.zeros((L, L))

    def get_pos(self, x):
        print(x)
        y = list(x)
        if x[0] >= 0:
            y[0] = x[0] % self.L
        else:
            y[0] = self.L - ((-x[0]) % self.L)

        if x[1] >= 0:
            y[1] = x[1] % self.L
        else:
            y[1] = self.L - ((-x[1]) % self.L)

        return tuple(y)

    def __getitem__(self, x):
        return self.arr[self.get_pos(x)]

    def __setitem__(self, x, value):
        self.arr[self.get_pos(x)] = value

