from fire import Fire
import matplotlib.pyplot as plt
import numpy as np


def plot_of_next_side(L: int):
    """Function make 50 MC probes of burning forest and draw a plot with
    the chance of fire getting to other side by probability of tree spawn.

    :argument
    L: int
    > the length of the side of the grid"""
    chances = []
    ps = [1/10 * p for p in range(11)]
    for p in ps:
        fire = Fire(L, p)
        chances.append(fire.do_MC(50))
    tribe = "start"
    start = 0
    stop = -1
    for i in range(10):
        if chances[i] != chances[i+1]:
            if tribe == "start":
                tribe = "stop"
                start = i
        else:
            if tribe == "stop":
                stop = i
                break
    p_start = ps[start]
    p_stop = ps[stop]
    ps2 = np.linspace(p_start, p_stop, 40)
    chances2 = []
    for p in ps2:
        fire = Fire(L, p)
        chances2.append(fire.do_MC(50))
    ps = [*ps[:start], *ps2, *ps[stop:]]
    chances = [*chances[:start], *chances2, *chances[stop:]]
    plt.plot(ps, chances)
