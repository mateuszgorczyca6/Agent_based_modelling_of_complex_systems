import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand


class Person:
    def __init__(self):
        self.state = False

    def update(self, p, q, innovation_ratio):
        U = rand()
        if not self.state and U < p + q * innovation_ratio:
            self.state = True
            if U < p:
                return 1, 0  # innovator
            return 0, 1  # imitator
        return 0, 0  # nothing

    def __int__(self):
        return int(self.state)


class Bass_Model:
    __Person = Person

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

    def __set_initial_types(self) -> np.ndarray:
        persons = np.array([Person() for _ in range(self.N)])
        return persons

    def update(self):
        ratio = self.__calculate_ratio()
        new_ps, new_qs = 0, 0
        for person in self.persons:
            new_p, new_q = person.update(self.p, self.q, ratio)
            new_ps += new_p
            new_qs += new_q
        new_all = new_ps + new_qs
        return ratio, new_ps, new_qs, new_all

    def __calculate_ratio(self):
        return self.persons.astype(int).mean()

    def __repr__(self):
        return f'Bass_Model({self.p}, {self.q})'


if __name__ == "__main__":
    model = Bass_Model(0.03, 0.38, 10000)
    ratios, ps, qs, all_news = [], [], [], []
    for _ in range(50):
        ratio, p_new, q_new, all_new = model.update()
        ratios.append(ratio)
        ps.append(p_new)
        qs.append(q_new)
        all_news.append(all_new)

    plt.plot(ratios)
    plt.show()

    plt.plot(ps)
    plt.plot(qs)
    plt.plot(all_news)
    plt.show()
