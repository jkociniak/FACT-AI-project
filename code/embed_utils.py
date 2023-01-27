import numpy as np
from collections import Counter, defaultdict
from os import listdir
from os.path import splitext
import networkx as nx
from gensim.models import Word2Vec
import walker

# determines whether to use random walks or breath first search to compute m/proximity
PROX_RW = False

# this should be practically infinite, but be compatible with sklearn.preprocessing.normalize as used in walker
INF = 10000

SENSATTR = "sensattr"
CLASS_NAME = "class"
DEFAULT_WEIGHT = 1
SEED = 0
P_NODE2VEC = 0.5
Q_NODE2VEC = 0.5
# TODO not sure about the values of the hyperparameters below
R = 1000
D = 10
WALKS_HYPER = {"n_walks": 10, "walk_len": 80}
SHARED_WORD2VEC_HYPER = {
    "vector_size": 128,
    "workers": 8,
    "min_count": 0,
    "sg": 1,
    "window": 5,
}
# hierarchical softmax
DEEPWALK_WORD2VEC_HYPER = {"hs": 1}
# sigmoid with negative sampling TODO add "negative": INT ?
NODE2VEC_WORD2VEC_HYPER = {"hs": 0}


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

    if ".class" in extensions:
        with open(path + ".class") as f_class:
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


def reweight_edges(graph, reweight_method, alpha=0.5, p=2):
    """
    reweight edge weights using either fairwalk or crosswalk
    Note that this does not normalize the weights, as that is done later in graph2embed
    by preprocess_transition_probs() anyway
    TODO allow alpha and p to be input
    """
    # Initiate new directed graph to store the new weights
    d_graph = graph.to_directed()
    node2group = nx.get_node_attributes(d_graph, SENSATTR)

    if reweight_method == "fairwalk":
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

    elif reweight_method == "crosswalk":
        # Estimate a measure of proximity (m) for each node to other groups in the graph
        if PROX_RW:
            # This follows equation 3 from the paper
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
                proximities.append(prox / denominator)
        else:
            # Follow the BFS based approach from the original code
            proximities = defaultdict(lambda: INF)
            for group in set(node2group.values()):
                queue = [v for v in graph if node2group[v] == group]
                head = 0
                dis = {v: 0 for v in queue}
                while head < len(queue):
                    cur = queue[head]
                    d_cur = dis[cur]
                    for u in d_graph[cur]:
                        if (node2group[u] != group) and (u not in dis):
                            queue.append(u)
                            dis[u] = d_cur + 1
                            proximities[u] = d_cur + 1
                    head += 1

        for node in d_graph.nodes:
            # Split the neighbors based on whether they share the group of the source node
            neigbors_same_group = []
            neigbors_diff_group = []
            for neighbor in d_graph.neighbors(node):
                if node2group[node] == node2group[neighbor]:
                    neigbors_same_group.append(neighbor)
                else:
                    neigbors_diff_group.append(neighbor)
            # Compute new weights from the source node to the neighbors of the same group
            # First compute the shared denominator
            neigbors_same_group_denominator = sum(
                [
                    graph[node][neighbor]["weight"] * proximities[neighbor] ** p
                    for neighbor in neigbors_same_group
                ]
            )
            # Now individually update all the weights
            for neighbor in neigbors_same_group:
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                    old_weight
                    * (1 - alpha)
                    * proximities[neighbor] ** p
                    / neigbors_same_group_denominator
                )
                d_graph[node][neighbor]["weight"] = new_weight
            # Compute new weights from the source node to the neighbors of a different group
            # First compute the shared denominator
            n_neigbors_diff_group = len(neigbors_diff_group)
            neigbors_diff_group_denominator = sum(
                [
                    n_neigbors_diff_group
                    * graph[node][neighbor]["weight"]
                    * proximities[neighbor] ** p
                    for neighbor in neigbors_diff_group
                ]
            )
            # Now individually update all the weights
            for neighbor in neigbors_diff_group:
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                    old_weight
                    * alpha
                    * proximities[neighbor] ** p
                    / neigbors_diff_group_denominator
                )
                d_graph[node][neighbor]["weight"] = new_weight

    return d_graph


def graph2embed(graph, reweight_method: str, embed_method: str):
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
    )
    if reweight_method != "default":
        graph = reweight_edges(graph, reweight_method)
    kwargs_word2vec = SHARED_WORD2VEC_HYPER.copy()

    if embed_method == "deepwalk":
        kwargs_word2vec.update(DEEPWALK_WORD2VEC_HYPER)
        p = 1
        q = 1
    elif embed_method == "node2vec":
        kwargs_word2vec.update(NODE2VEC_WORD2VEC_HYPER)
        p = P_NODE2VEC
        q = Q_NODE2VEC

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
