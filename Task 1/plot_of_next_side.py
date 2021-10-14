from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


def help_func(args):
    L, p = args
    fire = Fire(L, p)
    print("Doing for p = ", p)
    return fire.do_MC(50)


def plot_of_next_side(L: int):
    """Function make 50 MC probes of burning forest and draw a plot with
    the chance of fire getting to other side by probability of tree spawn.

    :argument
    L: int
    > the length of the side of the grid"""

    # chances = []
    # clusters = []
    ps = [1 / 50 * p for p in range(51)]
    args = [[L, 1 / 50 * p] for p in range(51)]

    # for p in ps:
    #     fire = Fire(L, p)
    #     chance, cluster = fire.do_MC(50)
    #     chances.append(chance)
    #     clusters.append(cluster)

    pool = mp.Pool()
    result = np.array(pool.map(help_func, args))
    pool.close()
    pool.join()
    chances = result[:, 0]
    clusters = result[:, 1]

    start = 0
    stop = 50
    for i in range(50):
        if chances[i] != chances[i + 1]:
            start = i
            break
    for i in range(50):
        if chances[50 - i - 1] != chances[50 - i]:
            stop = 50 - i
            break
    p_start = ps[start]
    p_stop = ps[stop]
    ps2 = np.linspace(p_start, p_stop, 40)
    args2 = []
    for p in ps2:
        args2.append([L, p])
    # chances2, clusters2 = [], []
    # for p in ps2:
    #     fire = Fire(L, p)
    #     chance, cluster = fire.do_MC(50)
    #     chances2.append(chance)
    #     clusters2.append(cluster)

    pool = mp.Pool()
    result = np.array(pool.map(help_func, args2))
    pool.close()
    pool.join()
    chances2 = result[:, 0]
    clusters2 = result[:, 1]

    ps = [*ps[:start], *ps2, *ps[stop:]]
    chances = [*chances[:start], *chances2, *chances[stop:]]
    clusters = [*clusters[:start], *clusters2, *clusters[stop:]]

    # plotting
    plt.plot(ps, chances, color="b")
    plt.plot(ps2, chances2, color="r")
    plt.show()
    plt.plot(ps, clusters, color="b")
    plt.plot(ps2, clusters2, color="r")
    plt.show()