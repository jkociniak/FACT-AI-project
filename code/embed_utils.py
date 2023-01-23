import numpy as np
from collections import Counter
from os import listdir
from os.path import splitext
import networkx as nx
from gensim.models import Word2Vec

# pip install pybind11
# pip install graph-walker
import walker


CLASS_NAME = "class"
DEFAULT_WEIGHT = 1
SEED = 0
P_NODE2VEC = 0.5
Q_NODE2VEC = 0.5
# TODO not sure about the values of the hyperparameters below
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
        ], "dataset should be 'rice', 'synth_3layers', 'synth2', 'synth3' or 'twitter'"
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
    path = f"../data/{dataset}/"
    path += splitext(listdir(path)[0])[0]

    with open(path + ".attr") as f_attr, open(path + ".links") as f_links:
        attr = [
            (int(i), {CLASS_NAME: int(c)})
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
    # for the twitter dataset only use the largest connected subgraph
    if dataset == "twitter":
        graph = get_largest_connected_subgraph(graph)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")
    return graph


def estimate_proximity(graph, node: int, class_node, r=10, d=80):
    """
    Estimates a measure of proximity of a node to other groups in the graph
    In the paper also known as m()
    TODO what values to use for r and d in estimate_proximity()?
    """
    walks = walker.random_walks(
        graph, n_walks=r, walk_len=d, start_nodes=[node], verbose=False
    )
    proximity = np.count_nonzero(walks != class_node) / (r * d)
    return proximity


def reweight_edges(graph, reweight_method, alpha=0.5, p=2):
    """
    reweight edge weights using either fairwalk or crosswalk
    Note that this does not normalize the weights, as that is done later in graph2embed
    by preprocess_transition_probs() anyway
    """
    # Initiate new directed graph to store the new weights
    d_graph = graph.to_directed()
    node2class = nx.get_node_attributes(d_graph, CLASS_NAME)

    if reweight_method == "fairwalk":
        # Get the unique classes
        classes = np.unique(list(node2class.values()))
        for node in d_graph.nodes():
            # Collect class information of the neighbors
            classes_neighbors = [
                node2class[neighbor] for neighbor in d_graph.neighbors(node)
            ]
            n_per_class = Counter(classes_neighbors)
            n_dif_classes = len(n_per_class)

            # Compute the new weights
            for neighbor in d_graph.neighbors(node):
                old_weight = graph[node][neighbor]["weight"]
                new_weight = old_weight / (
                    n_per_class[node2class[neighbor]] * n_dif_classes
                )
                d_graph[node][neighbor]["weight"] = new_weight

    elif reweight_method == "crosswalk":
        # TODO Probably faster if compute all at ones by passing all nodes to start_nodes and then restructuring the list
        proximities = [
            estimate_proximity(graph, node, node2class[node])
            for node in d_graph.nodes()
        ]

        for node in d_graph.nodes():
            # Split the neighbors based on whether they share the class of the source node
            neigbors_same_class = []
            neigbors_diff_class = []
            for neighbor in d_graph.neighbors(node):
                if node2class[node] == node2class[neighbor]:
                    neigbors_same_class.append(neighbor)
                else:
                    neigbors_diff_class.append(neighbor)
            # Compute new weights from the source node to the neighbors of the same class
            # First compute the shared denominator
            neigbors_same_class_denominator = sum(
                [
                    graph[node][neighbor]["weight"] * proximities[neighbor] ** p
                    for neighbor in neigbors_same_class
                ]
            )
            # Now individually update all the weights
            for neighbor in neigbors_same_class:
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                    old_weight
                    * (1 - alpha)
                    * proximities[neighbor] ** p
                    / neigbors_same_class_denominator
                )
                d_graph[node][neighbor]["weight"] = new_weight
            # Compute new weights from the source node to the neighbors of a different class
            # First compute the shared denominator
            n_neigbors_diff_class = len(neigbors_diff_class)
            neigbors_diff_class_denominator = sum(
                [
                    n_neigbors_diff_class
                    * graph[node][neighbor]["weight"]
                    * proximities[neighbor] ** p
                    for neighbor in neigbors_diff_class
                ]
            )
            # Now individually update all the weights
            for neighbor in neigbors_diff_class:
                old_weight = graph[node][neighbor]["weight"]
                new_weight = (
                    old_weight
                    * alpha
                    * proximities[neighbor] ** p
                    / neigbors_diff_class_denominator
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
        reweight_method=reweight_method,
        embed_method=embed_method,
    )
    path = get_embedding_path(dataset, reweight_method, embed_method)

    embed = np.load(path + ".npy", allow_pickle=False)

    return embed
