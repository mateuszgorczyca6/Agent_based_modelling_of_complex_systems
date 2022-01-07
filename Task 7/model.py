from itertools import pairwise
from math import floor

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class SituationQVoters:
    influence_modes = ['no influence', 'independent, changed', 'independent, unchanged', 'conformist, changed',
                       'conformist, unchanged']

    def __init__(self, n: int = 100, m: int = 5, graph: nx.Graph = None, p: float = 0.5, f: float = 0.5):
        self.f = f
        self.p = p
        self.Graph = self.__get_graph(graph, n, m)
        self.opinions = self.__get_states()
        self.history = []

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

    def step(self):
        group = self.find_4_connected()
        trueness = self.level_of_trueness(group)
        if trueness % 4 == 0:
            neighbors = self.get_neightbours(group)
            victim = np.random.choice(neighbors)
            opinion_before = self.opinions[victim].copy()
            influence = self.change_opinion(victim, trueness)
            opinion_after = self.opinions[victim]
            return group, influence, neighbors, victim, opinion_before, opinion_after
        else:
            influence = self.influence_modes[0]
            return group, influence, None, None, None, None

    def change_opinion(self, chosen_victim, trueness):
        if np.random.random() < self.p:  # independent
            if np.random.random() < self.f:  # change opinion
                self.opinions[chosen_victim] = not self.opinions[chosen_victim]
                return self.influence_modes[1]
            return self.influence_modes[2]
        else:  # conformist
            new_opinion = bool(floor(trueness / 4))
            if self.opinions[chosen_victim] != new_opinion:
                self.opinions[chosen_victim] = new_opinion
                return self.influence_modes[3]
            return self.influence_modes[4]

    def find_4_connected(self):
        nodes = np.array([np.random.choice(self.Graph.nodes)])
        for _ in range(3):
            neighbors = np.array(list(self.Graph.neighbors(nodes[-1])))
            neighbors = np.setdiff1d(neighbors, nodes)
            if len(neighbors) > 0:
                nodes = np.append(nodes, np.random.choice(neighbors))
        return nodes

    def level_of_trueness(self, nodes: np.ndarray):
        opinions = 0
        for node in nodes:
            opinions += int(self.opinions[node])
        return opinions

    def get_neightbours(self, nodes: np.ndarray):
        neighbours_of_group = np.array([])

        for node in nodes:
            neighbours_of_node = np.array(list(self.Graph.neighbors(node)))
            neighbours_of_group = np.append(neighbours_of_group, neighbours_of_node)

        neighbours_of_group = np.unique(neighbours_of_group)
        neighbours_of_group = np.setdiff1d(neighbours_of_group, nodes)
        return neighbours_of_group


def save_all_influences(model):
    influence_gained = []
    step = 0
    while len(influence_gained) < len(model.influence_modes):
        step += 1

        found, influence, neighbors, victim, opinion_before, opinion_after = model.step()

        if influence not in influence_gained:
            influence_gained.append(influence)

            def colour_nodes(item):
                if influence != model.influence_modes[0] and item in neighbors:
                    if model.opinions[item]:
                        return "lightgreen"
                    return "pink"
                if item in found:
                    if model.opinions[item]:
                        return "green"
                    return "red"
                return "gray"

            def resize_nodes(item):
                if item == victim:
                    return 500
                if item in found:
                    return 500
                return 100

            def colour_edges(item):
                if item in pairwise(found):
                    return "red"
                if item in pairwise(reversed(found)):
                    return "red"
                return "black"

            colors_nodes = list(map(colour_nodes, model.Graph.nodes))
            sizes_nodes = list(map(resize_nodes, model.Graph.nodes))
            colors_edges = list(map(colour_edges, model.Graph.edges))
            _, ax = plt.subplots(figsize=(10, 5))
            nx.draw(model.Graph, with_labels=True, node_color=colors_nodes, node_size=sizes_nodes,
                    edge_color=colors_edges, ax=ax)

            def legend_node(color, label):
                return Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=15,
                              label=label, linewidth=0)

            legend_elements = [legend_node('gray', 'unselected'),
                               legend_node('red', 'selected, disagree'),
                               legend_node('green', 'selected, agree'),
                               legend_node('pink', 'neigtbour, disagree'),
                               legend_node('lightgreen', 'neighbour, agree'),
                               Line2D([0], [0], color='red', label='selected connection', linewidth=1)]
            plt.legend(handles=legend_elements)
            agreement = model.level_of_trueness(found)
            title = f'step: {step}, selected: {found}, influence: {influence}'
            if influence != "no influence":
                title += f', victim: {victim}, opinion: {opinion_before} ' + r'$\rightarrow$' + f' {opinion_after}'
            ax.set_title(title)
            plt.savefig(influence + '.pdf')
            plt.show()


if __name__ == "__main__":
    model = SituationQVoters(n=20, m=2)
    save_all_influences(model)
