from itertools import pairwise

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class SituationQVoters:
    def __init__(self, n: int = 100, m: int = 5, graph: nx.Graph = None, p: float = 0.5):
        self.p = p
        self.Graph = self.__get_graph(graph, n, m)
        self.opinions = self.__get_states()

    @staticmethod
    def __get_graph(graph, n, m) -> nx.Graph:
        if isinstance(graph, nx.Graph):
            graph = graph
        else:
            graph = nx.barabasi_albert_graph(n, m)
        return graph

    def __get_states(self):
        n = len(self.Graph.nodes)
        opinions = np.random.binomial(1, 0.5, n).astype(bool)
        return dict(zip(self.Graph.nodes, opinions))

    def find_4_connected(self):
        # change to search by neighbours
        nodes = np.array([])
        while len(nodes) < 4:
            nodes = np.array([np.random.choice(self.Graph.nodes)])
            for _ in range(3):
                neighbors = np.array(list(self.Graph.neighbors(nodes[-1])))
                neighbors = np.setdiff1d(neighbors, nodes)
                if len(neighbors) > 0:
                    nodes = np.append(nodes, np.random.choice(neighbors))
        return nodes

    def do_they_agree(self, nodes: np.ndarray):
        opinions = 0
        for node in nodes:
            opinions += int(self.opinions[node])
        return opinions


if __name__ == "__main__":
    model = SituationQVoters(m=1)
    found = model.find_4_connected()
    print(found)

    def colour_nodes(item):
        if item in found:
            if model.opinions[item]:
                return "green"
            return "red"
        return "gray"


    def colour_edges(item):
        if item in pairwise(found):
            return "red"
        if item in pairwise(reversed(found)):
            return "red"
        return "black"


    colors_nodes = list(map(colour_nodes, model.Graph.nodes))
    colors_edges = list(map(colour_edges, model.Graph.edges))
    _, ax = plt.subplots(figsize=(10, 5))
    nx.draw(model.Graph, with_labels=True, node_color=colors_nodes, edge_color=colors_edges, ax=ax)
    legend_elements = [Line2D([0], [0], marker='o', color='gray', markerfacecolor='gray', markersize=15,
                              label='unselected', linewidth=0),
                       Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=15,
                              label='selected, disagree', linewidth=0),
                       Line2D([0], [0], marker='o', color='green', markerfacecolor='green', markersize=15,
                              label='selected, agree', linewidth=0),
                       Line2D([0], [0], color='red', label='selected connection', linewidth=1)]
    plt.legend(handles=legend_elements)
    agreement = model.do_they_agree(found)
    ax.set_title(f'selected: {found}, green nr: {agreement}')
    plt.show()
