from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand

GRAY_B = "#CCCCFF"
GRAY_G = "#CCFFCC"
GRAY_R = "#FFCCCC"


def test():
    """Do the plots for assignment."""
    model = Bass_Model(0.03, 0.38, 10000)  # instead of doing 10 MC probes we will use 10 times more agents.
    step = 0.1
    steps = 500
    MA_steps = 51

    ratios, ts, ps, qs, all_news = model.simulate(step, steps)
    # compensate for 100 more agents
    ratios /= 10
    ts /= 10
    ps /= 10
    qs /= 10
    all_news /= 10

    all_news_MA, ps_MA, qs_MA, ts_MA = calculate_MA_for_p_q_t_all(MA_steps, all_news, ps, qs, ts)
    plot_adopters(ratios, ts)
    plot_new_adopters(MA_steps, all_news, all_news_MA, ps, ps_MA, qs, qs_MA, ts, ts_MA)


class Person:
    """Person that is interested in product. For details see https://en.wikipedia.org/wiki/Bass_diffusion_model."""

    def __init__(self):
        """Init method for Person class."""
        self.state = False

    def update(self, p: float, q: float, innovation_ratio: float, step: float) -> [int, int]:
        """Switch state of person to true if it buys a product.
        Chance that person will buy a product is step*(p+q*innovation_ratio).

        Parameters
        ----------
        p - coefficient of innovation
        q - coefficient of imitation
        innovation_ratio - ration of adopters in model
        step - step (in years) that simulation will be forwarded

        Returns
        -------
        p - 1 if person bought the product as innovator during this year, 0 elsewhere;
        >     value is divided by step to compensate step size
        q - 1 if person bought the product as imitator during this year, 0 elsewhere
        >     value is divided by step to compensate step size
        """
        U = rand() / step
        if not self.state and U < p + q * innovation_ratio:
            self.state = True
            if U < p:
                return 1 / step, 0  # innovator
            return 0, 1 / step  # imitator
        return 0, 0  # nothing

    def __int__(self) -> int:
        """Integer representation of Person. State as integer."""
        return int(self.state)


class Bass_Model:
    """Bass Model class. For details see https://en.wikipedia.org/wiki/Bass_diffusion_model."""
    __Person: type = Person

    def __init__(self,
                 p_coefficient_of_innovation: float,
                 q_coefficient_of_imitation: float,
                 number_of_agents: int = 100):
        """Init method for Bass_Model class.

        Parameters
        ----------
        p_coefficient_of_innovation - coefficient of innovation
        q_coefficient_of_imitation - coefficient of imitation
        number_of_nodes - number of agents
        """
        assert 0 <= p_coefficient_of_innovation <= 1, "p_coefficient_of_innovation must be between 0 and 1."
        assert 0 <= q_coefficient_of_imitation <= 1, "q_coefficient_of_imitation must be between 0 and 1."
        self.p = p_coefficient_of_innovation
        self.q = q_coefficient_of_imitation
        self.N = number_of_agents
        self.persons = self.__set_initial_persons()

    def __set_initial_persons(self) -> np.ndarray:
        """Set's initial persons."""
        persons = np.array([Person() for _ in range(self.N)])
        return persons

    def update(self, step: float) -> [float, float, float, float]:
        """Update simulation by one step of duration <step>.

        Parameters
        ----------
        step - coefficient of innovation

        Returns
        ----------
        ratio - adopters ratio after step
        new_ps - number of new innovators
        new_qs - number of new imitators
        new_all - number of new adopters
        """
        assert step > 0, "step must be greater than 0."
        ratio = self.__calculate_ratio()
        new_ps, new_qs = 0, 0
        for person in self.persons:
            new_p, new_q = person.update(self.p, self.q, ratio, step)
            new_ps += new_p
            new_qs += new_q
        new_all = new_ps + new_qs
        return ratio, new_ps, new_qs, new_all

    def __calculate_ratio(self) -> float:
        """Returns adopters ratio in simulation."""
        return self.persons.astype(int).mean()

    def simulate(self, step: float, num_of_steps: int) -> [np.array, np.array, np.array, np.array]:
        """Do simulation of model for given step siz and umber of steps.

        Parameters
        ----------
        step - time step size
        num_of_steps - number of steps to simulate

        Returns
        -------
        ratios - array of adopters ratio for each step
        ts - array of time for each step
        ps - array of number of new innovators for each step
        qs - array of number of new imitators for each step
        all_news - array of number of new adopters for each step
        """
        assert step > 0, "step must be greater than 0."
        assert isinstance(num_of_steps, int), "num_of_steps must be integer."
        assert num_of_steps > 0, "num_of_steps must greater than 0."
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
        return np.array(ratios), np.array(ts), np.array(ps), np.array(qs), np.array(all_news)

    def __repr__(self) -> str:
        """Representation of the class object."""
        return f'Bass_Model({self.p}, {self.q}, {self.N})'


def calculate_MA_for_p_q_t_all(MA_steps, all_news, ps, qs, ts):
    """Calculates MA for ps, qs, ts and all arrays."""
    ts_MA = moving_average(ts, MA_steps)
    ps_MA = moving_average(ps, MA_steps)
    qs_MA = moving_average(qs, MA_steps)
    all_news_MA = moving_average(all_news, MA_steps)
    return all_news_MA, ps_MA, qs_MA, ts_MA


def moving_average(data: Iterable, steps: int) -> np.array:
    """Returns moving average on the given data. Length of data remains unchanged.

    Parameters
    ----------
    data - data that has to be averaged
    steps - number of data from which average is counted
    """
    data = np.array([*[data[0]] * int((steps - 1) / 2), *data, *[data[-1]] * int((steps - 1) / 2)])
    data = np.convolve(data, np.ones(steps), "valid") / steps
    return data


def plot_adopters(ratios, ts, fname='adopters.pdf'):
    """Creates the plot of adapters.

    Parameters
    ----------
    ratios - array of aratio of adopters for each step
    ts - array of time for each step
    fname - file name for plot save
    """
    plt.subplots(figsize=(12, 8))
    plt.plot(ts, ratios, c='black')
    plt.title("Adopters", fontsize=24)
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Ratio of adopters", fontsize=18)
    plt.savefig(fname)
    plt.show()


def plot_new_adopters(MA_steps, all_news, all_news_MA, ps, ps_MA, qs, qs_MA, ts, ts_MA, fname='new_adopters.pdf'):
    """Creates the plot of new adapters.

    Parameters
    ----------
    MA_steps - number of data from which average is counted
    all_news - array of number of new adopters for each step
    all_news_MA - averaged array of number of new adopters for each step
    ps - array of number of new innovators for each step
    ps_MA - averaged array of number of new innovators for each step
    qs - array of number of new imitators for each step
    qs_MA - averaged array of number of new innovators for each step
    ts - array of time for each step
    ts_MA - averaged array of number of new innovators for each step
    fname - file name for plot save
    """
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


if __name__ == "__main__":
    test()
