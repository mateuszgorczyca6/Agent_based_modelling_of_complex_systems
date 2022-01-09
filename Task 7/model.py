from typing import List, Union, Optional

import networkx as nx
import numpy as np

from itertools import pairwise
from math import floor

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class SituationQVoters:
    """Q-Voter Situation Simulation Model class.
    :params:
    influence_modes: text repr of all influence types"""
    influence_modes: list[str] = ['no influence', 'independent, changed', 'independent, unchanged',
                                  'conformist, changed',
                                  'conformist, unchanged']

    def __init__(self, n: int = 100, m: int = 5, graph: nx.Graph = None, p: float = 0.5, f: float = 0.5):
        """
        Init function of SituationQVoters. Creates graph and initial opinions.
        Parameters
        ----------
        n - number of Nodes in Barabashi-Albert graph, used if graph is None (default: None)
        m - number of new edges on new Barabashi-Albert graph, used if graph is None (default: None)
        graph - graph on which the model is simulated, if None, then Barabashi-Albert graph will be generated \
        (default: None)
        p - probability of spinson react independently
        f - probability of spinson change of mind, when reacting independently
        """
        self.f = f
        self.p = p
        self.Graph = self.__get_graph(graph, n, m)
        self.opinions = self.__get_states()

    @staticmethod
    def __get_graph(graph, n, m) -> nx.Graph:
        """Set's initial graph of simulation"""
        if isinstance(graph, nx.Graph):
            graph = graph
        else:
            graph = nx.barabasi_albert_graph(n, m)
        return graph

    def __get_states(self) -> dict[any: bool]:
        """Set's initial opinions of spinsons"""
        n = len(self.Graph.nodes)
        opinions = np.random.binomial(1, 0.5, n).astype(bool)
        return dict(zip(self.Graph.nodes, opinions))

    def step(self) -> (np.ndarray, str, np.ndarray, any, bool, bool):
        """Move simulation by one step.
        Returns
        -------
        group - list of labels of 4 connected nodes
        influence - type of influence that happened during this step
        neighbors - neighbours of group
        victim - node on which influence is made
        opinion_before - opinion of victim before influence
        opinion_after - opinion of node after influence
        """
        group = self.__find_4_connected()
        trueness = self.__level_of_trueness(group)
        if trueness % 4 == 0:
            return self.__make_influence(group, trueness)
        else:
            influence = self.influence_modes[0]
            return group, influence, None, None, None, None

    def __find_4_connected(self) -> np.ndarray:
        """Returns group of 4 connected nodes."""

        def get_nodes() -> np.ndarray:
            """Returns nodes in form of numpy array."""
            return np.array([np.random.choice(self.Graph.nodes)])

        nodes = get_nodes()
        for _ in range(3):
            neighbors = np.array(list(self.Graph.neighbors(nodes[-1])))
            neighbors = np.setdiff1d(neighbors, nodes)
            if len(neighbors) > 0:
                nodes = np.append(nodes, np.random.choice(neighbors))
        return nodes

    def __level_of_trueness(self, nodes: np.ndarray) -> int:
        """Calculates number of true opinions for given nodes."""
        opinions = 0
        for node in nodes:
            opinions += int(self.opinions[node])
        return opinions

    def __make_influence(self, group: np.ndarray, trueness: int) -> (np.ndarray, str, np.ndarray, any, bool, bool):
        """Applies influence on victim node."""

        def change_opinion(chosen_victim: any, trueness: int) -> str:
            """Change opinion of victim depending on random behaviour and group trueness."""

            def independent_behaviour(victim: any) -> str:
                """Change opinion of victim when it behaves independent."""
                if np.random.random() < self.f:  # change opinion
                    self.opinions[victim] = not self.opinions[victim]
                    return self.influence_modes[1]
                return self.influence_modes[2]

            def conformist_behaviour(victim: any, trueness: int) -> str:
                """Change opinion of victim when it behaves conformist depending on trueness."""
                new_opinion = bool(floor(trueness / 4))
                if self.opinions[victim] != new_opinion:
                    self.opinions[victim] = new_opinion
                    return self.influence_modes[3]
                return self.influence_modes[4]

            if np.random.random() < self.p:  # independent
                return independent_behaviour(chosen_victim)
            else:  # conformist
                return conformist_behaviour(chosen_victim, trueness)

        def get_neighbours(nodes: np.ndarray) -> np.ndarray:
            """Gets neighbourhood of group of nodes."""
            neighbours_of_group = np.array([])

            for node in nodes:
                neighbours_of_node = np.array(list(self.Graph.neighbors(node)))
                neighbours_of_group = np.append(neighbours_of_group, neighbours_of_node)

            neighbours_of_group = np.unique(neighbours_of_group)
            neighbours_of_group = np.setdiff1d(neighbours_of_group, nodes)
            return neighbours_of_group

        neighbors = get_neighbours(group)
        victim = np.random.choice(neighbors)
        opinion_before = self.opinions[victim]
        influence = change_opinion(victim, trueness)
        opinion_after = self.opinions[victim]
        return group, influence, neighbors, victim, opinion_before, opinion_after


def save_all_influences(model: SituationQVoters):
    """Saves to file and plot all types of influences that are implemented in model.
    Parameters
    ----------
    model - model to process
    """
    influence_gained = []
    step = 0
    while len(influence_gained) < len(model.influence_modes):
        step += 1

        found, influence, neighbors, victim, opinion_before, opinion_after = model.step()

        if influence not in influence_gained:
            influence_gained.append(influence)

            __draw_graph(model, found, influence, neighbors, victim, opinion_before, opinion_after, step)


def __draw_graph(model, found, influence, neighbors, victim, opinion_before, opinion_after, step):
    """Draws a graph of model for given arguments.
    Parameters
    ----------
    model - model to process
    found - 4 connected nodes, which influence on victim
    influence - type of influence on this step
    neighbors - neightbours of found nodes
    opinion_before - opinion of victim before influence
    opinion_after - opinion of victim after influence
    step - step number
    victim - victim node
    """
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

    def legend_node(color, label):
        return Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=15,
                      label=label, linewidth=0)

    colors_nodes = list(map(colour_nodes, model.Graph.nodes))
    sizes_nodes = list(map(resize_nodes, model.Graph.nodes))
    colors_edges = list(map(colour_edges, model.Graph.edges))
    _, ax = plt.subplots(figsize=(10, 5))
    nx.draw(model.Graph, with_labels=True, node_color=colors_nodes, node_size=sizes_nodes,
            edge_color=colors_edges, ax=ax)

    legend_elements = [legend_node('gray', 'unselected'),
                       legend_node('red', 'selected, disagree'),
                       legend_node('green', 'selected, agree'),
                       legend_node('pink', 'neigtbour, disagree'),
                       legend_node('lightgreen', 'neighbour, agree'),
                       Line2D([0], [0], color='red', label='selected connection', linewidth=1)]

    plt.legend(handles=legend_elements)
    title = f'step: {step}, selected: {found}, influence: {influence}'
    if influence != "no influence":
        title += f', victim: {victim}, opinion: {opinion_before} ' + r'$\rightarrow$' + f' {opinion_after}'
    ax.set_title(title)
    plt.savefig(influence + '.pdf')
    plt.show()


if __name__ == "__main__":
    model = SituationQVoters(n=20, m=2)
    save_all_influences(model)
