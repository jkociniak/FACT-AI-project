import unittest
import networkx as nx
from independent_cascade import run_averaged_ic, run_averaged_ic_parallel, get_greedy_seeds


class TestIndependentCascade(unittest.TestCase):
    @staticmethod
    def construct_H_graph():
        """
        Constructs the graph with 6 nodes. The graph is shown below.
        \_/
        / \

        Two central nodes act as optimal seeds.

        :return: the graph
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(6))
        graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
        return graph

    def test_run_ic_sequential(self):
        graph = self.construct_H_graph()
        nx.set_edge_attributes(graph, 0.5, 'ic_activation_prob')

        sample_size = 100000

        center_seeds = [1, 3]
        for seed in center_seeds:
            seeds = [seed]
            mean_inf_num = run_averaged_ic(graph, seeds, 'ic_activation_prob', sample_size)
            self.assertAlmostEqual(mean_inf_num, 3, delta=0.01)

        external_seeds = [0, 2, 4, 5]
        for seed in external_seeds:
            seeds = [seed]
            mean_inf_num = run_averaged_ic(graph, seeds, 'ic_activation_prob', sample_size)
            self.assertAlmostEqual(mean_inf_num, 2.25, delta=0.01)

    def test_run_ic_parallel(self):
        graph = self.construct_H_graph()
        nx.set_edge_attributes(graph, 0.5, 'ic_activation_prob')

        sample_size = 100000
        map_batch_size = 10
        reduce_batch_size = 100

        center_seeds = [1, 3]
        for seed in center_seeds:
            seeds = [seed]
            mean_inf_num = run_averaged_ic_parallel(graph, seeds, 'ic_activation_prob', sample_size,
                                                    map_batch_size, reduce_batch_size)
            self.assertAlmostEqual(mean_inf_num, 3, delta=0.01)

        external_seeds = [0, 2, 4, 5]
        for seed in external_seeds:
            seeds = [seed]
            mean_inf_num = run_averaged_ic_parallel(graph, seeds, 'ic_activation_prob', sample_size,
                                                    map_batch_size, reduce_batch_size)
            self.assertAlmostEqual(mean_inf_num, 2.25, delta=0.01)

    def greedy_test_framework(self, graph, sample_size, n_experiments, expected_seeds, expected_freq, parallel):
        assert expected_seeds
        n_seeds = len(expected_seeds[0])

        freqs = {s: 0 for s in expected_seeds}
        for i in range(n_experiments):
            seeds = get_greedy_seeds(graph, n_seeds, 'ic_activation_prob', sample_size, parallel)
            seeds = tuple(seeds)
            self.assertIn(seeds, expected_seeds)  # should pass with high probability due to big sample_size
            freqs[seeds] += 1

        # should pass with high probability for large n_experiments
        # but it is very time consuming, so delta is big
        for s in expected_seeds:
            self.assertAlmostEqual(freqs[s] / n_experiments, expected_freq, delta=0.1)

    def H_graph_greedy_test_framework(self, sample_size, n_experiments, parallel):
        graph = self.construct_H_graph()
        nx.set_edge_attributes(graph, 0.5, 'ic_activation_prob')

        expected_seeds = [(1,), (3,)]
        expected_freq = 0.5
        self.greedy_test_framework(graph, sample_size, n_experiments, expected_seeds, expected_freq, parallel)
        print('Passed single seed test')

        expected_seeds = [(1, 4), (1, 5), (3, 0), (3, 2)]
        expected_freq = 0.25
        self.greedy_test_framework(graph, sample_size, n_experiments, expected_seeds, expected_freq, parallel)
        print('Passed two seeds test')

    def test_get_greedy_seeds_sequential(self):
        sample_size = 1000  # increases one cascade accuracy
        n_experiments = 100  # increases the accuracy of the frequency estimation

        self.H_graph_greedy_test_framework(sample_size, n_experiments, parallel=False)

    def test_get_greedy_seeds_parallel(self):
        # it is very slow due to the overhead of parallelization on small graph

        sample_size = 1000  # increases one cascade accuracy
        n_experiments = 100  # increases the accuracy of the frequency estimation

        self.H_graph_greedy_test_framework(sample_size, n_experiments, parallel=True)
