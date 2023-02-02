from typing import List, Optional, Tuple
from copy import deepcopy
import networkx as nx
from numpy.random import default_rng
import logging
from collections import defaultdict
import ray
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)  # we use a logger to print info messages


def run_averaged_ic(graph: nx.Graph,
                    seeds: List[int],
                    activation_prob_attr: str,
                    sample_size: int):
    """
    Runs the IC model on the graph with the given seeds batch_siz times

    :param graph: the graph
    :param seeds: the seeds
    :param activation_prob_attr: the attribute name of the activation probability
    :param sample_size: the number of samples to take (because the process is stochastic)

    :return: the total of infected nodes
    """
    inf_mean = 0.
    for _ in range(sample_size):
        inf_num, _, _ = run_ic(graph, seeds, activation_prob_attr)
        inf_mean += inf_num / sample_size
    return inf_mean


def run_averaged_ic_parallel(graph: nx.Graph,
                             seeds: List[int],
                             activation_prob_attr: str,
                             sample_size: int,
                             map_batch_size: int = 200,
                             reduce_batch_size: int = 1000):
    """
    Computes the average number of infected nodes in parallel.

    :param graph: the graph
    :param seeds: the seeds
    :param activation_prob_attr: the attribute name of the activation probability
    :param sample_size: the number of samples to take (because the process is stochastic)
    :param map_batch_size: batch size for one process to run IC and sum the results
    :param reduce_batch_size: batch size for main process to fetch partial results

    :return: the average number of infected nodes
    """

    # next we define a function that runs a batch of independent cascades
    # this will be deployed to the workers
    ray.init()

    # we'll use the version which sums the results, we take the mean later
    @ray.remote
    def run_ic_parallel(graph, seeds, activation_prob_attr, batch_size):
        res = run_averaged_ic(graph, seeds, activation_prob_attr, batch_size)
        return res * batch_size

    # we put graph and seeds in the object store to avoid copying them to each worker
    graph_ref = ray.put(graph)
    seeds_ref = ray.put(seeds)

    # we collect the results of the independent cascades as futures
    futures = []

    # map
    samples_remaining = sample_size
    while samples_remaining > 0:
        batch_size = min(map_batch_size, samples_remaining)
        futures.append(run_ic_parallel.remote(graph_ref, seeds_ref, activation_prob_attr, batch_size))
        samples_remaining -= batch_size

    # reduce
    mean_inf_num = 0.
    while futures:
        rdy_futures, futures = ray.wait(futures, num_returns=min(reduce_batch_size, len(futures)))
        results = ray.get(rdy_futures)
        for inf_num in results:
            mean_inf_num += inf_num / sample_size

    ray.shutdown()

    return mean_inf_num


def run_ic(graph: nx.Graph,
           seeds: List[int],
           activation_prob_attr: str,
           extra_node: Optional[int] = None,
           return_final_nodes: bool = False) -> Tuple[int, Optional[List[int]], Optional[int]]:
    """
    Runs the IC model on the graph with the given seeds.

    :param graph: the graph
    :param seeds: the seeds
    :param activation_prob_attr: the attribute name of the activation probability
    :param extra_node: an extra node to add to the seeds
    :param return_final_nodes: whether to return the final nodes or not

    :return: the number of nodes activated, the final nodes (if return_final_nodes is True), and the extra node
    """
    # we create a copy of seeds to avoid modifying the original list
    # which could be shared with other processes (is it really?)
    inf_ids = deepcopy(seeds)
    if extra_node is not None:
        print(f'Adding extra node {extra_node} to seeds')
        # self.logger.debug(f'Adding extra node {extra_node} to seeds')
        inf_ids.append(extra_node)

    i = 0
    while i < len(inf_ids):  # len(inf_ids) is recalculated each iteration
        v = inf_ids[i]  # inf_ids acts as a queue for nodes to be processed
        neighbors = [u for u in graph[v] if u not in inf_ids]
        n_u = len(neighbors)

        # we use the numpy random number generator to generate decisions for each neighbor
        rng = default_rng()
        weights = rng.random(n_u)
        probs = [graph[v][u][activation_prob_attr] for u in neighbors]

        # we check which neighbors are activated and add them to the queue
        for u, weight, prob in zip(neighbors, weights, probs):
            if weight < prob:
                inf_ids.append(u)
        i += 1

    if return_final_nodes:
        return len(inf_ids), inf_ids, extra_node
    else:
        return len(inf_ids), None, extra_node


def get_greedy_seeds(graph: nx.Graph,
                     num_seeds: int,
                     activation_prob_attr: str,
                     sample_size: int,
                     parallel: bool = False,
                     **kwargs) -> List[int]:
    """
    Computes the greedy seeds for the given graph.

    :param graph: the graph
    :param num_seeds: the number of seeds to compute
    :param sample_size: the number of samples to take for each node
    :param activation_prob_attr: the attribute name of the activation probability
    :param parallel: whether to run in parallel or not
    :param kwargs: additional arguments for parallelization

    :return: the list of seeds
    """
    seeds = []

    for i in tqdm(range(num_seeds)):
        logger.info(f'Choosing seed {i + 1} out of {num_seeds}')
        if parallel:
            next_seed, max_influence = get_next_seed_parallel(graph, seeds, activation_prob_attr, sample_size,
                                                              **kwargs)
        else:
            next_seed, max_influence = get_next_seed(graph, seeds, activation_prob_attr, sample_size)
        logger.info(f'Chosen seed: {next_seed} with influence {max_influence}')
        seeds.append(next_seed)

    return seeds


def get_next_seed(graph, seeds, activation_prob_attr, sample_size):
    """
    Computes the next seed to add to the seeds list.

    :param graph: the graph
    :param seeds: the seeds
    :param sample_size: the number of samples to take for each node
    :param activation_prob_attr: the attribute name of the activation probability

    :return: the next seed to add to the seeds list and its influence score
    """

    inf_means = defaultdict(float)
    for v in graph.nodes:
        if v not in seeds:
            inf_mean = run_averaged_ic(graph, seeds + [v], activation_prob_attr, sample_size)
            inf_means[v] = inf_mean

    # we extract the node with the highest mean influence
    next_seed, max_inf_mean = max(inf_means.items(), key=lambda x: x[1])

    return next_seed, max_inf_mean


def get_next_seed_parallel(graph: nx.Graph,
                           seeds: List[int],
                           activation_prob_attr: str,
                           sample_size: int,
                           map_batch_size: int=200,
                           reduce_batch_size: int=1000) -> Tuple[int, float]:
    """
    Computes the next seed to add to the seeds list.

    :param graph: the graph
    :param seeds: the seeds
    :param sample_size: the number of samples to take for each node
    :param activation_prob_attr: the attribute name of the activation probability
    :param map_batch_size: the batch size for the map step
    :param reduce_batch_size: the batch size for the reduce step

    :return: the next seed to add to the seeds list and its influence score
    """
    ray.init()

    # we'll use the version which sums the results, we take the mean later
    @ray.remote
    def run_ic_parallel(graph, seeds, activation_prob_attr, batch_size, extra_node):
        res = run_averaged_ic(graph, seeds + [extra_node], activation_prob_attr, batch_size)
        return res * batch_size, extra_node

    # compute all possible seeds in parallel using ray
    graph_ref = ray.put(graph)
    seeds_ref = ray.put(seeds)

    futures = []

    # map
    for v in graph.nodes:
        if v not in seeds:
            samples_remaining = sample_size
            while samples_remaining > 0:
                batch_size = min(map_batch_size, samples_remaining)
                futures.append(run_ic_parallel.remote(graph_ref, seeds_ref, activation_prob_attr, batch_size, v))
                samples_remaining -= batch_size

    # reduce
    inf_means = defaultdict(float)
    while futures:
        rdy_futures, futures = ray.wait(futures, num_returns=min(reduce_batch_size, len(futures)))
        results = ray.get(rdy_futures)
        for inf_num, v in results:
            inf_means[v] += inf_num / sample_size

    ray.shutdown()

    # we extract the node with the highest mean influence
    next_seed, max_inf_mean = max(inf_means.items(), key=lambda x: x[1])

    return next_seed, max_inf_mean


def get_kmedoids_seeds(graph_emb: np.array, num_seeds: int) -> List[int]:
    """
    Computes the k-medoids seeds for the given graph embedding.

    :param graph_emb: embedding of graph nodes
    :param num_seeds: the number of seeds to compute

    :return: the list of seeds
    """
    seeds = KMedoids(num_seeds).fit(graph_emb).medoid_indices_
    return seeds.tolist()
