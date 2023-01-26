from InfluenceMaximizationRunner import InfluenceMaximizationRunner
from embed_utils import data2graph, graph2embed, SENSATTR
import numpy as np
import logging

#logging.basicConfig(level=logging.DEBUG)

graph = data2graph('rice')
emb = graph2embed(graph, embed_method='deepwalk', reweight_method='default')
# from karateclub import DeepWalk
#
# model = DeepWalk()
# model.fit(graph)
# emb = model.get_embedding()


def calculate_metrics(inf_num, inf_num_grouped, group_counts):
    print(f'Total mean influence (nodes): {inf_num}')
    print(f'Total mean influence (%): {100 * inf_num / len(graph.nodes)}%')

    print(f'Mean influence in each group (nodes): {inf_num_grouped}')
    inf_num_grouped_frac = {g: 100 * inf_num_group / group_counts[g] for g, inf_num_group in inf_num_grouped.items()}
    print(f'Mean influence in each group (%): {inf_num_grouped_frac}')

    disparity = np.var(list(inf_num_grouped_frac.values()))
    print(f'Disparity: {disparity}')
    print()


def test_for_rng_seed(rng_seed):
    runner = InfluenceMaximizationRunner(graph, graph_emb=emb, group_attr=SENSATTR, activation_prob=0.01, parallel=False)
    inf_num, inf_num_grouped = runner.run(IC_init_method='kmedoids',
                                          num_seeds=40,
                                          IC_init_greedy_sample_size=500,
                                          repetitions=500,
                                          rng_seed=rng_seed)

    group_counts = {g: 0 for g in runner.groups}
    for _, g in graph.nodes(data=SENSATTR):
        group_counts[g] += 1
    for g, count in group_counts.items():
        print(f'Group {g} has {count} nodes')

    calculate_metrics(inf_num, inf_num_grouped, group_counts)


if __name__ == '__main__':
    for rng_seed in [10, 42, 100, 69, 420]:
        test_for_rng_seed(rng_seed)
