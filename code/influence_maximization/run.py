import utils as ut
import numpy as np
from copy import deepcopy
import multiprocessing
from random import random
import logging
import networkx as nx
from typing import List, Dict
from sklearn_extra.cluster import KMedoids


class InfluenceMaximizationException(Exception):
    pass


class InfluenceMaximizationRunner:
    def __init__(self, graph: nx.Graph, parallel: bool = False):
        self.graph = graph
        self.parallel = parallel
        self.colors = np.unique(c for _, c in graph.nodes(data='color'))

    def run(self,
            IC_init_method: str = 'greedy',
            IC_init_num_seeds: int = 40,
            IC_init_greedy_sample_size: int = 500,
            repetitions: int = 500,
            rng_seed: int = 42):
        """
        :param init_method: 'greedy' or 'kmedoids'
        :param num_seeds: number of seeds to select
        :param greedy_sample_size: number of samples to take when selecting seeds using greedy method
        :param repetitions: number of repetitions to run the IC model for each seed set
        """
        self.reset_rng(rng_seed)

        # get influence seeds
        if IC_init_method == 'greedy':
            seeds = self.get_greedy_seeds(IC_init_num_seeds, IC_init_greedy_sample_size)
        elif IC_init_method == 'kmedoids':
            seeds = self.get_kmedoids_seeds(IC_init_num_seeds)
        else:
            raise ValueError('Unknown seed initialization method: {}'.format(IC_init_method))

        # calculate influence
        #I, I_grouped = self.run_experiments(seeds, repetitions)
        influenced_mean, influenced_mean_grouped = self.run_experiments(seeds, repetitions)
        color2influenced = self.group_by_color(list(self.graph))

        return influenced_mean, influenced_mean_grouped, color2influenced

    def group_by_color(self, nodes: List[int]) -> Dict[int, List[int]]:
        color2nodes = {c: [] for c in self.colors}
        for n in nodes:
            c = self.graph[n]['color']
            color2nodes[c].append(n)
        return color2nodes

    def run_experiments(self, seeds, repetitions=500):
        influenced_mean = 0.0
        influenced_mean_grouped = {c: 0. for c in self.colors}

        if self.parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
            results = pool.map(self.run_IC, [seeds] * repetitions)
            pool.close()
            pool.join()

            for influenced, color2influenced in results:
                num_influenced = float(len(influenced))
                influenced_mean += num_influenced / repetitions
                for c, t in color2influenced.items():
                    influenced_mean_grouped[c] += float(len(t)) / repetitions
        else:
            for _ in range(repetitions):
                influenced, color2influenced = self.run_IC(seeds)
                num_influenced = float(len(influenced))
                influenced_mean += num_influenced / repetitions
                for c, t in color2influenced.items():
                    influenced_mean_grouped[c] += float(len(t)) / repetitions

        return influenced_mean, influenced_mean_grouped

    def run_IC(self, seeds: List[int]) -> List[int]:  # Tuple[List[int], Dict[int, List[int]]]:
        """
        Runs the IC model on the graph with the given seeds.

        :param: seeds: list of seed nodes

        :return: list of influenced nodes
        """
        influenced = deepcopy(seeds)  # copy already selected nodes

        i = 0
        while i < len(influenced):  # len(T) is recalculated each iteration
            v = influenced[i]
            for u in self.graph[v]:  # iterate over neighbors of a selected node
                if u not in influenced:
                    w = self.graph[v][u]['weight']  # probability of infection
                    if random() <= w:  # i.e. now w is actually probability and there is only one edge between two nodes
                        logging.debug(f'{v} influences {u}')
                        influenced.append(u)
            i += 1

        return influenced

    def get_greedy_seeds(self, num_seeds: int, sample_size: int) -> List[int]:
        seeds = []
        influenced = []
        influenced_grouped = []
        for i in range(num_seeds):
            max_influenced = 0
            next_seed = None
            for v in self.graph.nodes():
                if v not in seeds:
                    influenced_mean = 0.
                    for _ in range(sample_size):
                        influenced = self.run_IC(seeds + [v])  # evaluate the score with v added to seeds
                        influenced_mean += len(influenced) / sample_size  # to calculate the mean progressively

                    # using > we disable adding nodes which don't influence any new nodes,
                    # but then we need to handle the following case:
                    #   no new nodes are influenced, but we did not reach the desired number of seeds
                    #   how to choose the next seed?
                    #   under which assumptions can we assume that we can add a node during every iteration?
                    if influenced_mean >= max_influenced:
                        max_influenced = influenced_mean
                        next_seed = v
            if next_seed is None:
                msg = '[seed selection - greedy] no next seed found'
                logging.debug(msg)
                raise InfluenceMaximizationException(msg)
            seeds.append(next_seed)

            influenced_mean, influenced_mean_grouped = self.run_IC(seeds)
            influenced.append(influenced_mean)
            influenced_grouped.append(influenced_mean_grouped)

            group = self.graph[next_seed]['color']
            print(f'{i + 1} selected node: {next_seed},  group {group} I_grouped = {influenced_mean_grouped}')

            color2seeds = {c: [] for c in self.colors}
            for n in seeds:
                c = self.graph[n]['color']
                color2seeds[c].append(n)

            seeds.append(color2seeds)  # id's of the seeds so the influence can be recreated

        return seeds

    def get_kmedoids_seeds(self, num_seeds: int) -> List[int]:
        G_emb = self.graph.nodes(data='emb')
        seeds = KMedoids(num_seeds).fit(G_emb).medoid_indices_
        return seeds



