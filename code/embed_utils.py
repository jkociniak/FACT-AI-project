import numpy as np
import random
from os import listdir
from os.path import splitext
import networkx as nx
from gensim.models import Word2Vec
import node2vec
import deepwalk


DEFAULT_WEIGHT = 1
SEED = 0
NODE2VEC_HYPER = {"p": 0.5, "q": 0.5}
# TODO not sure about the values of the hyperparameters below
WALKS_HYPER = {"num_walks": 10, "walk_length": 80}
SHARED_WORD2VEC_HYPER = {"vector_size": 128, "workers": 8, "min_count": 0}
# skipgram with hierarchical softmax
DEEPWALK_WORD2VEC_HYPER = {"sg": 1, "hs": 1, "window": 5}
# skipgram with negative sampling TODO add "negative": 5 ?
NODE2VEC_WORD2VEC_HYPER = {"sg": 1, "hs": 0, "window": 10}


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
            (int(i), {"class": int(c)})
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


def reweight_edges(graph, reweight_method):
    """
    reweight edge weights using either fairwalk or crosswalk
    """
    if reweight_method == "fairwalk":
        # TODO
        pass
    elif reweight_method == "crosswalk":
        # TODO
        pass
    else:
        pass
    return graph


def graph2embed(graph, reweight_method: str, embed_method: str):
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
    )
    graph = reweight_edges(graph, reweight_method)

    if embed_method == "deepwalk":
        kwargs_word2vec = SHARED_WORD2VEC_HYPER | DEEPWALK_WORD2VEC_HYPER
        graph_deepwalk = deepwalk.from_networkx(graph)
        walks = deepwalk.build_deepwalk_corpus(
            graph_deepwalk,
            num_paths=WALKS_HYPER["num_walks"],
            path_length=WALKS_HYPER["walk_length"],
            rand=random.Random(SEED),
        )

    elif embed_method == "node2vec":
        kwargs_word2vec = SHARED_WORD2VEC_HYPER | NODE2VEC_WORD2VEC_HYPER
        graph_node2vec = node2vec.Graph(
            graph, is_directed=graph.is_directed(), **NODE2VEC_HYPER
        )
        graph_node2vec.preprocess_transition_probs()
        walks = graph_node2vec.simulate_walks(**WALKS_HYPER)
        # walks = [list(map(str, walk)) for walk in walks]

    # TODO verifying that order with respect to initial indices is preserved
    model = Word2Vec(walks, **kwargs_word2vec)
    # TODO is copy() necessary here?
    embed = model.wv.vectors.copy()

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
