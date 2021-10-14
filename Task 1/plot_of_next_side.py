from fire import Fire
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


def help_func(args):
    L, p = args
    fire = Fire(L, p)
    print("Doing for p = ", p)
    return fire.do_MC(50)


def plot_of_next_side(L: int, fname1: str = "plot1.pdf", fname2: str = "plot2.pdf"):
    """Function make 50 MC probes of burning forest and draw a plot with
    the chance of fire getting to other side by probability of tree spawn.

    :argument
    L: int
    > the length of the side of the grid
    fname1: str
    > name of chances plot
    fname2: str
    > name of average highest cluster plot"""

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

    # add_MA
    chances_toMA = np.array([0, 0, *chances2, 1, 1])
    chances_MA = np.convolve(chances_toMA, np.ones(5), "valid") / 5

    # getting biggest slope
    chances_MA_max_diff = 0
    chances_MA_max_diff_i = 0
    for i in range(len(chances2) - 1):
        chances_MA_diff = chances_MA[i + 1] - chances_MA[i]
        if chances_MA_diff > chances_MA_max_diff:
            chances_MA_max_diff = chances_MA_diff
            chances_MA_max_diff_i = i
    chances_MA_max_diff_p = (ps2[chances_MA_max_diff_i] + ps2[chances_MA_max_diff_i + 1]) / 2
    chances_MA_max_diff_chance = (chances_MA[chances_MA_max_diff_i] + chances_MA[chances_MA_max_diff_i + 1]) / 2
    ps2_diff = ps2[1] - ps2[0]

    chances_MA_max_diff_p_star = chances_MA_max_diff_p - ps2_diff * chances_MA_max_diff_chance / chances_MA_max_diff

    # plotting
    plt.figure(figsize=(12, 8))
    plt.plot(ps, chances, color="#7777FF", label="normal precision")
    plt.plot(ps2, chances2, color="#FFCCCC", label="higher precision")
    plt.plot(ps2, chances_MA, color="green", label="MA(5)")
    plt.plot([chances_MA_max_diff_p, chances_MA_max_diff_p_star],
             [chances_MA_max_diff_chance, 0],
             "--", marker="o", color="green", label="counting of $p^*$")
    plt.text(chances_MA_max_diff_p_star + 0.02,
             0,
             "$p^*\\approx$" + " %.2f" % chances_MA_max_diff_p_star,
             verticalalignment='center', fontsize=12)
    plt.title('Probability that fire hits the opposite edge for different p\n L={}'.format(L), fontsize=18)
    plt.xlabel('p', fontsize=12)
    plt.ylabel('q', fontsize=12)  # q - fraction of simulations where fire hits the opposite edge
    plt.legend()
    plt.savefig(fname1)
    plt.show()
    plt.figure(figsize=(12, 8))
    plt.plot(ps, clusters, color="b", label="normal precision")
    plt.plot(ps2, clusters2, color="r", label="higher precision")
    plt.title('Average highest cluster of burnt trees for different p\n L={}'.format(L), fontsize=18)
    plt.xlabel('p', fontsize=12)
    plt.ylabel('Average highest cluster')
    plt.legend()
    plt.savefig(fname2)
    plt.show()
    print("Max change is ", chances_MA_max_diff_p_star)
