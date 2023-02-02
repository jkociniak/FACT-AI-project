import networkx as nx
import time
from independent_cascade import run_averaged_ic, run_averaged_ic_parallel
from embed_utils import data2graph


def benchmark_sequential(graph, seeds, activation_prob_attr, sample_size):
    """
    Benchmark the sequential implementation.

    :param graph: the graph
    :param seeds: the seeds
    :param activation_prob_attr: the activation probability attribute
    :param sample_size: the number of samples

    :return: the mean influence score and the time
    """
    start_time = time.time()
    mean_inf_num = run_averaged_ic(graph, seeds, activation_prob_attr, sample_size)
    end_time = time.time()
    seq_time = end_time - start_time
    print(f'sequential time: {seq_time}, mean inf num: {mean_inf_num}')
    return mean_inf_num, seq_time


def benchmark_parallel(graph, seeds, activation_prob_attr, sample_size,
                       map_batch_size, reduce_batch_size):
    """
    Benchmark the parallel implementation.

    :param graph: the graph
    :param seeds: the seeds
    :param sample_size: the number of samples
    :param map_batch_size: the batch size for the map step
    :param reduce_batch_size: the batch size for the reduce step
    """
    print(f'map_batch_size: {map_batch_size}, reduce_batch_size: {reduce_batch_size}')
    # parallel
    start_time = time.time()
    mean_inf_num = run_averaged_ic_parallel(graph, seeds, activation_prob_attr, sample_size,
                                            map_batch_size, reduce_batch_size)
    end_time = time.time()
    parallel_time = end_time - start_time
    print(f'parallel time: {parallel_time}, mean inf num: {mean_inf_num}')
    return mean_inf_num, parallel_time


def construct_clique(n):
    """
    Constructs a big clique graph.

    :param n: the number of nodes

    :return: the graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from([(i, j) for i in range(n) for j in range(i + 1, n)])
    return graph


def test_clique_500():
    n_nodes = 500
    graph = construct_clique(n_nodes)
    activation_prob_attr = 'ic_activation_prob'
    nx.set_edge_attributes(graph, 0.5, activation_prob_attr)
    print(f'constructed clique with {n_nodes} nodes')

    seeds = [0]
    print(f'seeds: {seeds}')

    sample_size = 250
    print(f'sample_size: {sample_size}')

    benchmark_sequential(graph, seeds, activation_prob_attr, sample_size)

    map_batch_sizes = [10, 20, 50]
    reduce_batch_size = 100
    for map_batch_size in map_batch_sizes:
        benchmark_parallel(graph, seeds, activation_prob_attr, sample_size, map_batch_size, reduce_batch_size)


def test_rice():
    graph = data2graph('rice')
    activation_prob_attr = 'ic_activation_prob'
    nx.set_edge_attributes(graph, 0.01, activation_prob_attr)
    print(f'constructed rice dataset')

    seeds = [0]
    print(f'seeds: {seeds}')

    sample_size = 500
    print(f'sample_size: {sample_size}')

    benchmark_sequential(graph, seeds, activation_prob_attr, sample_size)

    map_batch_sizes = [200]
    reduce_batch_size = 500
    for map_batch_size in map_batch_sizes:
        benchmark_parallel(graph, seeds, activation_prob_attr, sample_size, map_batch_size, reduce_batch_size)


if __name__ == '__main__':
    test_clique_500()
    test_rice()

