import os
import networkx as nx
import numpy as np
from collections import Counter
from independent_cascade.independent_cascade import run_ic, get_kmedoids_seeds
from embed_utils import data2graph, SENSATTR, graph2embed
import json


def run_experiment(dataset,
                   ic_activation_prob,
                   ic_activation_prob_attr,
                   seed_method,
                   seed_params,
                   infmax_params):
    graph = data2graph(dataset)
    nx.set_edge_attributes(graph, ic_activation_prob, ic_activation_prob_attr)
    print(f'constructed {dataset} dataset')

    if seed_method == 'load_greedy':
        seeds = load_greedy_seeds(dataset, **seed_params)
    elif seed_method == 'kmedoids':
        graph_emb = graph2embed(graph, **seed_params)
        seeds = get_kmedoids_seeds(graph_emb, 40)
    else:
        raise ValueError(f'Unknown seed method: {seed_method}')

    print(f'Loaded seeds: {seeds}')

    inf, disp = run_infmax(graph, seeds, ic_activation_prob_attr, **infmax_params)
    print(f'Influence scores: {inf}, disparities: {disp}')
    print(f'Mean influence: {np.mean(inf)}, mean disparity: {np.mean(disp)} over {len(inf)} experiments')
    print()

    return inf, disp


def load_greedy_seeds(dataset, seeds_dir):
    seeds_name = f'{dataset}_greedy_seeds.txt'
    seeds_path = os.path.join(seeds_dir, seeds_name)
    print(f'Loading seeds from {seeds_path}')
    with open(seeds_path, 'r') as f:
        seeds = eval(f.read())  # the seeds are saved as a list of integers
    return seeds


def run_infmax(graph, seeds, activation_prob_attr, n_experiments=5, ic_sample_size=500):
    all_counter = Counter()
    for v, sens_val in graph.nodes(data=SENSATTR):
        all_counter[sens_val] += 1

    # we run the experiments
    influences = []
    disparities = []
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
        influences.append(inf_frac)
        print(f'Total mean influence (%): {inf_frac}%')

        inf_frac_grouped = {g: 100 * mean_inf_nums[g] / all_counter[g] for g in mean_inf_nums}
        disparity = np.var(list(inf_frac_grouped.values()))
        disparities.append(disparity)
        print(f'Disparity: {disparity}')
        print()
    return influences, disparities


if __name__ == '__main__':
    # common settings
    infmax_params = {'n_experiments': 5, 'ic_sample_size': 500}
    activation_prob_attr = 'ic_activation_prob'
    save_dir = '../results/influence_maximization/bigbatch'
    embed_method = 'deepwalk'

    # list of (dataset, activation_prob) tuples
    real_datasets = ['rice',
                     'twitter']
                     #'stanford']
    synth_datasets = [#'synth_3layers',
                      'synth2',
                      'synth3']

    # possible reweighing strategies
    reweight_methods = ['default', 'fairwalk', 'crosswalk']

    repetitions = 100

    for dataset in synth_datasets + real_datasets:
        ap = 0.01 if dataset in real_datasets else 0.03

        # greedy seeds
        # seed_params = {'seeds_dir': '../results/influence_maximization/greedy_seeds'}
        # run_experiment(dataset, ap, activation_prob_attr, 'load_greedy', seed_params, infmax_params, save_path)

        # save_name = f'{dataset}_greedy.json'
        # save_path = os.path.join(save_dir, save_name)
        # if save_path is not None:
        #     print(f'Saving results to {save_path}')
        #     with open(save_path, 'w') as f:
        #         json.dump(results, f, indent=4)

        # kmedoids seeds
        for rm in reweight_methods:
            seed_params = {'reweight_method': rm, 'embed_method': embed_method}
            if rm == 'crosswalk':
                seed_params['reweight_params'] = {
                    'alpha': 0.5 if dataset in real_datasets else 0.7,
                    'p': 2 if dataset == 'twitter' else 4
                }

            results = {
                'settings': {
                    'dataset': dataset,
                    'ic_activation_prob': ap,
                    'ic_activation_prob_attr': activation_prob_attr,
                    'seed_method': 'kmedoids',
                    'seed_params': seed_params,
                    'infmax_params': infmax_params
                },
            }

            infs, disps = [], []
            for i in range(repetitions):  # averaging over 1000 different seed sets
                inf, disp = run_experiment(dataset, ap, activation_prob_attr, 'kmedoids', seed_params, infmax_params)
                infs.append(np.mean(inf))
                disps.append(np.mean(disp))

            results['influences'] = infs
            results['inf_mean'] = np.mean(infs)
            results['inf_var'] = np.var(infs)
            results['disparities'] = disps
            results['disp_mean'] = np.mean(disps)
            results['disp_var'] = np.var(disps)

            save_name = f'{dataset}_{rm}_mean_over_{repetitions}_seedsets.json'
            save_path = os.path.join(save_dir, save_name)
            if save_path is not None:
                print(f'Saving results to {save_path}')
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)



