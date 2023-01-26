import numpy as np
from copy import deepcopy
import multiprocessing
import random
import logging
import networkx as nx
from typing import List, Dict, Optional
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

class InfluenceMaximizationException(Exception):
    pass


class InfluenceMaximizationRunner:
    def __init__(self,
                 graph: nx.Graph,
                 graph_emb: np.array,
                 group_attr: str,
                 activation_prob: Optional[float] = 0.01,  # should be 0.01 for real and 0.03 for synthetic datasets
                 parallel: bool = False):
        self.graph = graph
        self.graph_emb = graph_emb
        self.parallel = parallel
        self.activation_prob = activation_prob
        self.group_attr = group_attr
        self.groups = np.unique(list(a for _, a in graph.nodes(data=group_attr)))

    def run(self,
            IC_init_method: str = 'greedy',
            num_seeds: int = 40,
            IC_init_greedy_sample_size: int = 500,
            repetitions: int = 500,
            rng_seed: int = 42):
        """
        :param init_method: 'greedy' or 'kmedoids'
        :param num_seeds: number of seeds to select
        :param greedy_sample_size: number of samples to take when selecting seeds using greedy method
        :param repetitions: number of repetitions to run the IC model for each seed set
        :param rng_seed: random number generator seed
        """
        self.reset_rng(rng_seed)

        # get influence seeds
        if IC_init_method == 'greedy':
            seeds = self.get_greedy_seeds(num_seeds, IC_init_greedy_sample_size)
        elif IC_init_method == 'kmedoids':
            seeds = self.get_kmedoids_seeds(num_seeds)
        else:
            raise ValueError('Unknown seed initialization method: {}'.format(IC_init_method))

        # calculate influence
        inf_mean_num, inf_mean_num_grouped = self.run_experiments(seeds, repetitions)

        return inf_mean_num, inf_mean_num_grouped

    def reset_rng(self, rng_seed):
        """
        Resets the random number generator seed.
        """
        random.seed(rng_seed)

    def group_nodes(self, nodes: List[int]) -> Dict[int, List[int]]:
        group2nodes = {c: [] for c in self.groups}
        for n in nodes:
            c = self.graph.nodes[n][self.group_attr]
            group2nodes[c].append(n)
        return group2nodes

    def run_experiments(self, seeds, repetitions=500):
        inf_mean_num = 0.0
        inf_mean_num_grouped = {c: 0. for c in self.groups}

        if self.parallel:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
            results = pool.map(self.run_IC, [seeds] * repetitions)
            pool.close()
            pool.join()
        else:
            results = [self.run_IC(seeds) for _ in range(repetitions)]

        for inf_ids in results:
            # calculate the mean influence for the whole graph
            inf_num = float(len(inf_ids))
            inf_mean_num += inf_num / repetitions

            # calculate the mean influence in each group
            inf_ids_grouped = self.group_nodes(inf_ids)
            for group, inf_ids_group in inf_ids_grouped.items():
                inf_mean_num_grouped[group] += float(len(inf_ids_group)) / repetitions

        return inf_mean_num, inf_mean_num_grouped

    def run_IC(self, seeds: List[int]) -> List[int]:  # Tuple[List[int], Dict[int, List[int]]]:
        """
        Runs the IC model on the graph with the given seeds.

        :param: seeds: list of seed nodes

        :return: list of influenced nodes
        """
        inf_ids = deepcopy(seeds)  # copy already selected nodes

        i = 0
        while i < len(inf_ids):  # len(T) is recalculated each iteration
            v = inf_ids[i]
            for u in self.graph[v]:  # iterate over neighbors of a selected node
                if u not in inf_ids:
                    if self.activation_prob is not None:
                        p = self.activation_prob  # we use constant probability of infection, as in the paper
                    else:
                        p = self.graph[v][u]['weight']  # we use weights of edges as the probability of infection
                    if random.random() <= p:
                        #logging.debug(f'{v} influences {u}')
                        inf_ids.append(u)
            i += 1

        return inf_ids

    def get_greedy_seeds(self, num_seeds: int, sample_size: int, track_stats: bool = True) -> List[int]:
        seeds = []
        # seeds_grouped = []
        # influenced = []
        # influenced_grouped = []

        for i in range(num_seeds):  # the number of seed we choose
            logging.info(f'Choosing seed {i + 1} out of {num_seeds}')
            next_seed, max_influence = self.get_next_seed(seeds, sample_size)
            logging.debug(f'Chosen seed: {next_seed} with influence {max_influence}')
            if next_seed is None:
                msg = '[seed selection - greedy] no next seed found'
                logging.debug(msg)
                raise InfluenceMaximizationException(msg)
            seeds.append(next_seed)

            if track_stats:
                # save the number of influenced nodes, also divided by group
                inf_mean, inf_mean_grouped = self.run_experiments(seeds)

                # influenced.append(influenced_mean)
                # influenced_grouped.append(influenced_mean_grouped)

                # print info about the current iteration
                group = self.graph.nodes[next_seed][self.group_attr]
                print(f'[Iteration {i + 1}] selected node {next_seed} belonging to group {group}')
                print(f'total influence = {inf_mean}')
                print(f'mean influence for each group = {inf_mean_grouped}')

                # save the seed for each color
                # color2seeds = self.group_by_color(seeds)
                # seeds_grouped.append(color2seeds)  # id's of the seeds so the influence can be recreated

        return seeds

    def get_next_seed(self, seeds, sample_size):
        max_influenced = 0  # we want to calculate which extension obtains the best score
        next_seed = None  # we start with no best extension
        for v in tqdm(self.graph.nodes()):
            logging.debug(f'Considering node {v} for an extension')
            if v not in seeds:
                influenced_mean = 0.
                for _ in range(sample_size):
                    influenced = self.run_IC(seeds + [v])  # evaluate the score with v added to seeds
                    influenced_mean += len(influenced) / sample_size  # to calculate the mean progressively
                logging.debug(f'Extension with {v} influences {influenced_mean} nodes on average')

                # using > we disable adding nodes which don't influence any new nodes,
                # but then we need to handle the following case:
                #   no new nodes are influenced, but we did not reach the desired number of seeds
                #   how to choose the next seed?
                #   under which assumptions can we assume that we can add a node during every iteration?
                if influenced_mean >= max_influenced:
                    max_influenced = influenced_mean
                    next_seed = v
        return next_seed, max_influenced

    def get_kmedoids_seeds(self, num_seeds: int) -> List[int]:
        seeds = KMedoids(num_seeds).fit(self.graph_emb).medoid_indices_
        return seeds.tolist()



