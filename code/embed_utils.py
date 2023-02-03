import numpy as np
from collections import Counter, defaultdict
from os import listdir
from os.path import splitext
import networkx as nx
from gensim.models import Word2Vec
import walker
import random

SENSATTR = "sensattr"
CLASS_NAME = "class"
DEFAULT_WEIGHT = 1
SEED = 0
P_NODE2VEC = 0.5
Q_NODE2VEC = 0.5
# TODO not sure about the values of the hyperparameters below
R = 1000
D = 5
WALKS_HYPER = {"n_walks": 10, "walk_len": 40}
SHARED_WORD2VEC_HYPER = {
    "vector_size": 64,
    "workers": 8,
    "min_count": 0,
    "sg": 1,
    "window": 5,
}
# hierarchical softmax
DEEPWALK_WORD2VEC_HYPER = {"hs": 1}
# sigmoid with negative sampling TODO add "negative": INT ?
NODE2VEC_WORD2VEC_HYPER = {"hs": 0}


def create_rice_target_ds():
    with open("../data/rice/rice_subset.attr") as f:
        ids_subset = set([node.split()[0] for node in f.read().strip().split("\n")])

    with open("../data/rice_raw/rice_raw.attr") as f:
        target_subset = [
            id + " " + target + "\n"
            for node in f.read().strip().split("\n")
            for id, target in [node.split()[:2]]
            if id in ids_subset
        ]

    with open("../data/rice/rice_subset.target", "w") as f:
        f.writelines(target_subset)


def check_input_formatting(**kwargs):
    """
    Check that the input strings follow the correct naming conventions
    """
    if "dataset" in kwargs:
        assert kwargs["dataset"] in [
            "rice",
            "synth_3layers",
            "synth2",
            "synth3",
            "twitter",
            "stanford",
        ], "dataset should be 'rice', 'synth_3layers', 'synth2', 'synth3', 'twitter' or 'stanford'"
    if "reweight_method" in kwargs:
        assert kwargs["reweight_method"] in [
            "default",
            "fairwalk",
            "crosswalk",
        ], "reweight_method should be 'default', 'fairwalk' or 'crosswalk'"
    if "embed_method" in kwargs:
        assert kwargs["embed_method"] in [
            "deepwalk",
            "node2vec",
        ], "embed_method should be 'deepwalk' or 'node2vec'"


def get_largest_connected_subgraph(graph):
    nodes_subgraph = max(nx.connected_components(graph), key=len)
    subgraph = nx.induced_subgraph(graph, nodes_subgraph)
    return subgraph


def data2graph(dataset: str):
    check_input_formatting(dataset=dataset)
    path = f"../data/{dataset}/"
    extensions = [splitext(file_name)[1] for file_name in listdir(path)]
    path += splitext(listdir(path)[0])[0]

    with open(path + ".attr") as f_attr, open(path + ".links") as f_links:
        attr = [
            (int(i), {SENSATTR: int(c)})
            for node in f_attr.read().strip().split("\n")
            for i, c in [node.split()]
        ]
        links = [
            (int(i1), int(i2), DEFAULT_WEIGHT)
            for node in f_links.read().strip().split("\n")
            for i1, i2 in [node.split()[:2]]
        ]

    graph = nx.Graph()
    graph.add_nodes_from(attr)
    graph.add_weighted_edges_from(links)

    if ".target" in extensions:
        with open(path + ".target") as f_class:
            classes = {
                int(i): {CLASS_NAME: int(c)}
                for node in f_class.read().strip().split("\n")
                for i, c in [node.split()]
            }

        nx.set_node_attributes(graph, classes)

    # for the twitter dataset only use the largest connected subgraph
    if dataset == "twitter":
        graph = get_largest_connected_subgraph(graph)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")
    return graph


def graph2embed(graph, reweight_method: str, embed_method: str, reweight_params=None):
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
    )

    if reweight_params is None:
        reweight_params = {}

    if reweight_method != "default":
        graph = reweight_edges(graph, reweight_method, **reweight_params)

    kwargs_word2vec = SHARED_WORD2VEC_HYPER.copy()
    if embed_method == "deepwalk":
        kwargs_word2vec.update(DEEPWALK_WORD2VEC_HYPER)
        p = 1
        q = 1
    elif embed_method == "node2vec":
        kwargs_word2vec.update(NODE2VEC_WORD2VEC_HYPER)
        p = P_NODE2VEC
        q = Q_NODE2VEC
    else:
        # should never happen due to asserts in check_input_formatting TODO remove?
        raise ValueError("embed_method should be 'deepwalk' or 'node2vec'")

    # Generate random walks
    walks = walker.random_walks(graph, p=p, q=q, verbose=False, **WALKS_HYPER)
    walks = [list(map(str, walk)) for walk in walks]

    # Generate embeddings
    model = Word2Vec(walks, **kwargs_word2vec)

    # Ensure that the indices stay consistent
    original_graph_indices = [
        model.wv.key_to_index[str(i)] for i in range(model.wv.__len__())
    ]
    # TODO is copy() necessary here?
    embed = model.wv[original_graph_indices].copy()

    return embed


def reweight_edges(graph, reweight_method, **kwargs):
    """
    reweight edge weights using either fairwalk or crosswalk
    Note that this does not normalize the weights, as that is done later in graph2embed
    by preprocess_transition_probs() anyway
    TODO allow alpha and p to be input
    """

    if reweight_method == "fairwalk":
        d_graph = reweight_fairwalk(graph)
    elif reweight_method == "crosswalk":
        d_graph = reweight_crosswalk(graph, **kwargs)
    # elif reweight_method == 'crosswalk_original_bfs':
    #     d_graph = reweight_crosswalk_original_bfs(graph)
    else:
        raise ValueError("reweight_method should be 'fairwalk' or 'crosswalk'")

    return d_graph


def reweight_fairwalk(graph):
    # Initiate new directed graph to store the new weights
    d_graph = graph.to_directed()
    node2group = nx.get_node_attributes(d_graph, SENSATTR)

    for node in d_graph.nodes:
        # Collect density for the groups of the neighbors
        density_per_group = defaultdict(lambda: 0)
        for neighbor in d_graph.neighbors(node):
            density_per_group[node2group[neighbor]] += graph[node][neighbor][
                "weight"
            ]

        # Compute the new weights
        for neighbor in d_graph.neighbors(node):
            old_weight = graph[node][neighbor]["weight"]
            new_weight = old_weight / density_per_group[node2group[neighbor]]
            d_graph[node][neighbor]["weight"] = new_weight

    return d_graph


def reweight_crosswalk(graph, alpha=0.5, p=4):
    d_graph = graph.to_directed()
    node2group = nx.get_node_attributes(d_graph, SENSATTR)

    # This follows equation 3 and from the paper
    # Estimate a measure of proximity (m) for each node to other groups in the graph
    # Generate the walks to estimate m for all nodes at once
    walks = walker.random_walks(
        d_graph, n_walks=R, walk_len=D, start_nodes=d_graph.nodes, verbose=False
    )

    # Precompute factors that are in the list comprehension
    n_nodes = d_graph.number_of_nodes()
    total_walks = walks.shape[0]
    denominator = R * D

    # Compute proximity for each batch of walks with the same starting node
    proximities = []
    for node in d_graph.nodes:
        prox = 0
        for walk in walks[np.arange(node, total_walks, n_nodes)]:
            prox += walk.tolist().count(node2group[node])
        proximities.append(prox / denominator + 0.00001)

    # Compute the new weights
    for node in d_graph.nodes():
        # Split the neighbors based on whether they share the class of the source node
        neighbors_groups = Counter()
        for neighbor in d_graph.neighbors(node):
            neighbors_groups[node2group[neighbor]] += 1

        if not neighbors_groups:
            print(f'Node {node} has no neighbors.')
            continue

        if node2group[node] in neighbors_groups and len(neighbors_groups) == 1:
            denominator = sum(
                graph[node][neighbor]["weight"] * proximities[neighbor] ** p
                for neighbor in d_graph.neighbors(node)
            )
            for neighbor in d_graph.neighbors(node):
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                        old_weight * proximities[
                    neighbor] ** p  # we drop multiplication by (1-alpha) here bc we have only 1 group
                        / denominator
                )
                d_graph[node][neighbor]["weight"] = new_weight
            continue

        print(f'There is at least one neighbor of {node} in a different group than {node}.')
        for group in neighbors_groups:
            print(f'Processing neighbors of group {group}')
            neighbors_in_group = [u for u in d_graph.neighbors(node) if node2group[u] == group]
            denominator = sum(
                graph[node][neighbor]["weight"] * proximities[neighbor] ** p
                for neighbor in neighbors_in_group
            )

            # we excluded the situation where node has only neighbors of the same class
            # so we have three more cases to consider
            if group == node2group[
                node]:  # neighbor of the same class, we multiply by (1-alpha) bc we are sure that we have at least one other group
                coeff = (1 - alpha)
            else:
                if node2group[node] in neighbors_groups:  # neighbor of a different class
                    coeff = alpha / (len(neighbors_groups) - 1)
                else:  # neighbor of a different class, but node has only neighbors of a different classes, so we don't need to multiply by alpha and subtract 1 from number of classes
                    coeff = 1 / len(neighbors_groups)

            for neighbor in neighbors_in_group:
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                        old_weight
                        * coeff
                        * proximities[neighbor] ** p
                        / denominator
                )
                d_graph[node][neighbor]["weight"] = new_weight

    return d_graph

# def crosswalk_original_bfs(graph, n_walks=10, walk_len=80):
#     """
#     based on crosswalk in the original code
#     """
#     node2group = nx.get_node_attributes(graph, SENSATTR)
#     border_distance = defaultdict(lambda: np.inf)
#     proximity = defaultdict(lambda: 0)
#     for group in set(node2group.values()):
#         # we run bfs starting from each node of the group and collect distances from the border
#         queue = [v for v in graph if node2group[v] == group]
#         head = 0
#         dis = {v: 0 for v in queue}
#         while head < len(queue):
#             cur = queue[head]
#             d_cur = dis[cur]
#             for u in graph[cur]:
#                 if (node2group[u] != group) and (u not in dis):
#                     queue.append(u)
#                     dis[u] = d_cur + 1
#                     border_distance[u] = d_cur + 1
#             head += 1
#
#     rand = random.Random()
#     # I assume this is alpha
#     p_ch = 0.5
#     paths = []
#     for _ in range(n_walks):
#         for start in graph.nodes:
#             path = [start]
#             while len(path) < walk_len:
#                 cur = path[-1]
#                 if len(graph[cur]) > 0:
#                     if border_distance[cur] == 1:
#                         l_1 = [
#                             u for u in graph[cur] if node2group[u] == node2group[cur]
#                         ]
#                         l_2 = [
#                             u for u in graph[cur] if node2group[u] != node2group[cur]
#                         ]
#                     else:
#                         l_1 = [
#                             u
#                             for u in graph[cur]
#                             if border_distance[u] >= border_distance[cur]
#                         ]
#                         l_2 = [
#                             u
#                             for u in graph[cur]
#                             if border_distance[u] < border_distance[cur]
#                         ]
#                     if (len(l_1) == 0) or (len(l_2) == 0):
#                         choice = rand.choice(list(graph[cur].keys()))
#                         path.append(choice)
#                     else:
#                         if np.random.rand() < p_ch:
#                             path.append(rand.choice(l_2))
#                         else:
#                             path.append(rand.choice(l_1))
#                 else:
#                     break
#             paths.append(path)
#     return paths


def get_embedding_path(dataset: str, reweight_method: str, embed_method: str):
    """
    Construct path to the embeddings files following our conventions
    """
    path = f"../embeddings/{dataset}/{dataset}_{reweight_method}_{embed_method}"
    return path


def save_embed(
    embed: np.array,
    dataset: str,
    reweight_method: str,
    embed_method: str,
):
    """
    Saves an embedding to the appropriate directionary and file.
    TODO Maybe hyperparameter info like the number of embedding dimensions should be added
    For now without pickle for possible compatability issues.
    """
    check_input_formatting(
        dataset=dataset,
        reweight_method=reweight_method,
        embed_method=embed_method,
    )
    path = get_embedding_path(dataset, reweight_method, embed_method)
    np.save(path, embed, allow_pickle=False)


def load_embed(dataset: str, reweight_method: str, embed_method: str):
    """
    load an embedding.
    """
    check_input_formatting(
        dataset=dataset,
        reweight_method=reweight_method,
        embed_method=embed_method,
    )
    path = get_embedding_path(dataset, reweight_method, embed_method)

    embed = np.load(path + ".npy", allow_pickle=False)

    return embed
