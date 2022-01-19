import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand

GRAY_B = "#CCCCFF"

GRAY_G = "#CCFFCC"

GRAY_R = "#FFCCCC"


class Person:
    def __init__(self):
        self.state = False

    def update(self, p, q, innovation_ratio, step):
        U = rand() / step
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

    def update(self, step):
        ratio = self.__calculate_ratio()
        new_ps, new_qs = 0, 0
        for person in self.persons:
            new_p, new_q = person.update(self.p, self.q, ratio, step)
            new_ps += new_p
            new_qs += new_q
        new_all = new_ps + new_qs
        return ratio, new_ps, new_qs, new_all

    def __calculate_ratio(self):
        return self.persons.astype(int).mean()

    def simulate(self, step, num_of_steps):
        ratios, ts, ps, qs, all_news = [], [], [], [], []
        t = 0
        for _ in range(num_of_steps):
            t += step
            ts.append(t)
            ratio, p_new, q_new, all_new = self.update(step)
            ratios.append(ratio)
            ps.append(p_new)
            qs.append(q_new)
            all_news.append(all_new)
        return ratios, ts, ps, qs, all_news

    def __repr__(self):
        return f'Bass_Model({self.p}, {self.q})'


def moving_average(data, steps):
    data = np.array([*[data[0]] * int((steps - 1) / 2), *data, *[data[-1]] * int((steps - 1) / 2)])
    data = np.convolve(data, np.ones(steps), "valid") / steps
    return data


def test():
    model = Bass_Model(0.03, 0.38, 10000)
    step = 0.1
    steps = 500
    MA_steps = 51

    ratios, ts, ps, qs, all_news = model.simulate(step, steps)
    all_news_MA, ps_MA, qs_MA, ts_MA = calculate_MA_for_p_q_t_all(MA_steps, all_news, ps, qs, ts)
    plot_adopters(ratios, ts)
    plot_new_adopters(MA_steps, all_news, all_news_MA, ps, ps_MA, qs, qs_MA, ts, ts_MA)


def plot_new_adopters(MA_steps, all_news, all_news_MA, ps, ps_MA, qs, qs_MA, ts, ts_MA, fname='new_adopters.pdf'):
    _, ax = plt.subplots(figsize=(12, 8))

    plt.plot(ts, ps, c=GRAY_R, label='Innovators - raw')
    plt.plot(ts, qs, c=GRAY_G, label='Imitators - raw')
    plt.plot(ts, all_news, c=GRAY_B, label='New adopters - raw')

    plt.plot(ts_MA, ps_MA, c='r', label=f'Innovators - MA({MA_steps})')
    plt.plot(ts_MA, qs_MA, c='g', label=f'Imitators - MA({MA_steps})')
    plt.plot(ts_MA, all_news_MA, c='b', label=f'New adopters - MA({MA_steps})')

    plt.title("Adopters", fontsize=24)
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Number of new adopters", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0]))
    plt.legend(handles, labels, fontsize=14)

    plt.savefig(fname)
    plt.show()


def plot_adopters(ratios, ts, fname='adopters.pdf'):
    plt.subplots(figsize=(12, 8))
    plt.plot(ts, ratios, c='black')
    plt.title("Adopters", fontsize=24)
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Ratio of adopters", fontsize=18)
    plt.savefig(fname)
    plt.show()


def calculate_MA_for_p_q_t_all(MA_steps, all_news, ps, qs, ts):
    ts_MA = moving_average(ts, MA_steps)
    ps_MA = moving_average(ps, MA_steps)
    qs_MA = moving_average(qs, MA_steps)
    all_news_MA = moving_average(all_news, MA_steps)
    return all_news_MA, ps_MA, qs_MA, ts_MA


if __name__ == "__main__":
    test()
