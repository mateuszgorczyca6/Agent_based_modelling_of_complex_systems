from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand


class Person:
    def __init__(self):
        self.state = False

    def __int__(self):
        return int(self.state)


class Innovator(Person):
    def update(self, p, *_):
        if rand() < p:
            self.state = True
            return 1, 0
        return 0, 0


class Imitator(Person):
    def update(self, _, q, innovation_ratio):
        if rand() < q * innovation_ratio:
            self.state = True
            return 0, 1
        return 0, 0


class Bass_Model:
    def __init__(self,
                 p_coefficient_of_innovation: float,
                 q_coefficient_of_imitation: float,
                 number_of_nodes: int = 100):
        assert 0 <= p_coefficient_of_innovation <= 1
        assert 0 <= q_coefficient_of_imitation <= 1
        self.p = p_coefficient_of_innovation
        self.q = q_coefficient_of_imitation
        self.N = number_of_nodes
        self.persons = self.__set_initial_types()
        print(self.persons)

    def __set_initial_types(self) -> np.ndarray:
        persons = deque([])
        for person in range(self.N):
            if rand() < self.p:
                persons.append(Innovator())
            else:
                persons.append(Imitator())
        return np.array(persons)

    def update(self):
        ratio = self.__calculate_ratio()
        new_ps, new_qs = 0, 0
        for person in self.persons:
            new_p, new_q = person.update(self.p, self.q, ratio)
            new_ps += new_p
            new_qs += new_q
        return ratio, new_ps, new_qs

    def __calculate_ratio(self):
        return self.persons.astype(int).mean()

    def __repr__(self):
        return f'Bass_Model({self.p}, {self.q})'


if __name__ == "__main__":
    model = Bass_Model(0.03, 0.38, 1000)
    ratios, ps, qs = [], [], []
    for _ in range(50):
        ratio, p, q = model.update()
        ratios.append(ratio)
        ps.append(p)
        qs.append(q)

    plt.plot(ratios)
    plt.show()

    plt.plot(ps)
    plt.plot(qs)
    plt.show()
