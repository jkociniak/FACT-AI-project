import os
import networkx as nx
import numpy as np
from collections import Counter
from independent_cascade.independent_cascade import run_ic, get_kmedoids_seeds
from embed_utils import data2graph, SENSATTR, graph2embed


def calculate_metrics(inf_num, inf_num_grouped, group_counts):
    # logging.debug(f'Total mean influence (nodes): {inf_num}')
    print(f'Total mean influence (%): {100 * inf_num / len(graph.nodes)}%')

    # logging.debug(f'Mean influence in each group (nodes): {inf_num_grouped}')
    inf_num_grouped_frac = {g: 100 * inf_num_group / group_counts[g] for g, inf_num_group in inf_num_grouped.items()}
    # logging.debug(f'Mean influence in each group (%): {inf_num_grouped_frac}')

    disparity = np.var(list(inf_num_grouped_frac.values()))



def run_infmax(graph, seeds, n_experiments=5, ic_sample_size=500):
    all_counter = Counter()
    for v, sens_val in graph.nodes(data=SENSATTR):
        all_counter[sens_val] += 1

    # we run the experiments
    for i in range(n_experiments):
        print(f'Experiment {i + 1}/{n_experiments}')

        influence_counter = Counter()
        for _ in range(ic_sample_size):
            _, inf_ids, _ = run_ic(graph, seeds, activation_prob_attr, return_final_nodes=True)
            for v in inf_ids:
                sens_val = graph.nodes[v][SENSATTR]
                influence_counter[sens_val] += 1

        mean_inf_nums = {}
        for g, count in influence_counter.items():
            mean_inf_nums[g] = count / ic_sample_size

        inf_frac = 100 * sum(mean_inf_nums.values()) / len(graph)
        print(f'Total mean influence (%): {inf_frac}%')

        inf_frac_grouped = {g: 100 * mean_inf_nums[g] / all_counter[g] for g in mean_inf_nums}
        disparity = np.var(list(inf_frac_grouped.values()))
        print(f'Disparity: {disparity}')
        print()


if __name__ == '__main__':
    real_datasets = ['rice', 'twitter', 'stanford']
    synth_datasets = ['synth_3layers', 'synth2', 'synth3']
    datasets = synth_datasets + real_datasets

    activation_prob_attr = 'ic_activation_prob'
    seeds_dir = '../results/influence_maximization/greedy_seeds'

    for dataset in datasets[3:4]:
        graph = data2graph(dataset)
        ic_activation_prob = 0.01 if dataset in real_datasets else 0.03
        nx.set_edge_attributes(graph, ic_activation_prob, activation_prob_attr)
        print(f'constructed {dataset} dataset')

        # seeds_name = f'{dataset}_greedy_seeds.txt'
        # seeds_path = os.path.join(seeds_dir, seeds_name)
        # print(f'Loading seeds from {seeds_path}')
        # with open(seeds_path, 'r') as f:
        #     seeds = eval(f.read())  # the seeds are saved as a list of integers

        graph_emb = graph2embed(graph, reweight_method='fairwalk', embed_method='deepwalk')
        seeds = get_kmedoids_seeds(graph_emb, 40)
        print(f'Loaded seeds: {seeds}')

        run_infmax(graph, seeds, n_experiments=5)
