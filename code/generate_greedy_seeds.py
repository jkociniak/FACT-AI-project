import networkx as nx
from embed_utils import data2graph
from independent_cascade.independent_cascade import get_greedy_seeds
import time
import logging
import os


def benchmark(graph, seeds, activation_prob_attr, sample_size, parallel, **kwargs):
    """
    Benchmark the sequential implementation.

    :param graph: the graph
    :param seeds: the seeds
    :param activation_prob_attr: the activation probability attribute
    :param sample_size: the number of samples
    :param parallel: whether to use the parallel implementation

    :return: the mean influence score and the time
    """
    start_time = time.time()
    seeds = get_greedy_seeds(graph, seeds, activation_prob_attr, sample_size, parallel, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    title = 'parallel' if parallel else 'sequential'
    print(f'{title} time: {duration}, seeds: {seeds}')
    return seeds, duration


def compute_greedy_seeds(dataset, directory):
    """
    Compute the greedy seeds for the given dataset.

    :param dataset: the dataset
    :param directory: the directory to save the results
    """
    graph = data2graph(dataset)
    activation_prob_attr = 'ic_activation_prob'
    nx.set_edge_attributes(graph, 0.01, activation_prob_attr)
    print(f'constructed {dataset} dataset')

    num_seeds = 40
    sample_size = 500
    print(f'sample_size: {sample_size}')

    parallel = True
    map_batch_size = 200
    reduce_batch_size = 1000

    seeds, duration = benchmark(graph, num_seeds, activation_prob_attr, sample_size, parallel,
                                map_batch_size=map_batch_size, reduce_batch_size=reduce_batch_size)

    print(f'Greedy seeds (parallel): {seeds}, duration: {duration}')
    path = os.path.join(directory, f'{dataset}_greedy_seeds.txt')
    with open(path, 'w') as f:
        f.write(str(seeds))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    directory = 'results/influence_maximization/greedy_seeds'
    if not os.path.exists(directory):
        os.makedirs(directory)

    datasets = ['twitter']
    for dataset in datasets:
        compute_greedy_seeds(dataset, directory)

