import numpy as np
from os import listdir
from os.path import splitext
import networkx as nx
from karateclub import DeepWalk, Node2Vec
from gensim.models import Word2Vec
import walker

# if fairwalk doesn't import properly, try to (re)install it in your env. See its readme for instructions
from fairwalk import FairWalk


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
        ], (
            "dataset should be either 'rice', 'synth_3layers',"
            " 'synth2', 'synth3' or 'twitter'"
        )
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
    if "implementation" in kwargs:
        assert kwargs["implementation"] in [
            "karateclub",
            "singer",
        ], "implementation should be 'karateclub' or 'singer' (for now)"


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
            (int(i1), int(i2))
            for node in f_links.read().strip().split("\n")
            for i1, i2 in [node.split()[:2]]
        ]

    graph = nx.Graph()
    graph.add_nodes_from(attr)
    graph.add_edges_from(links)
    # for the twitter dataset only use the largest connected subgraph
    if dataset == "twitter":
        graph = get_largest_connected_subgraph(graph)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="label")
    return graph


# def reweight_edges(graph, reweight_method):
#     """
#     reweight edge weights using either fairwalk or crosswalk
#     """
#     if reweight_method == "fairwalk":
#         # TODO
#         pass
#     elif reweight_method == "crosswalk":
#         # TODO
#         pass
#     else:
#         pass
#     return graph


# def graph2embed(
#     graph, reweight_method: str, embed_method: str, implementation: str, p=1, q=1
# ):
#     check_input_formatting(
#         reweight_method=reweight_method,
#         embed_method=embed_method,
#         implementation=implementation,
#     )
#     graph = reweight_edges(graph, reweight_method)
#     # TODO, not sure about the hyperparameters for word2vec
#     kwargs_walks = {"n_walks": 10, "walk_len": 80}
#     kwargs_word2vec = {"vector_size": 128, "workers": 4, "min_count": 0, "window": 10}
#     if embed_method == "deepwalk":
#         kwargs_walks.update({"p": 1, "q": 1})
#         # skipgram with hierarchical softmax
#         kwargs_word2vec.update({"sg": 1, "hs": 1})
#     elif embed_method == "node2vec":
#         kwargs_walks.update({"p": p, "q": q})
#         # skipgram with negative sampling
#         kwargs_word2vec.update({"sg": 1, "hs": 0, "negative": 5})
#     # get walks
#     walks = walker.random_walks(graph, **kwargs_walks)

#     model = Word2Vec(walks, **kwargs_word2vec)
#     # TODO verifying that order with respect to initial indices is preserved
#     embed = model.wv.vectors.copy()


def graph2embed(graph, reweight_method, embed_method, implementation="karateclub"):
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
        implementation=implementation,
    )
    graph = reweight_graph(graph, reweight_method)
    if embed_method == "deepwalk":
        if implementation == "karateclub":
            # TODO CHANGE HYPERPARAMETERS TO THE ONES FROM THE CROSSWALK PAPER
            model = DeepWalk()
            model.fit(graph)
            embed = model.get_embedding()
    elif embed_method == "node2vec":
        if implementation == "karateclub":
            # TODO CHANGE HYPERPARAMETERS TO THE ONES FROM THE CROSSWALK PAPER
            model = Node2Vec()
            model.fit(graph)
            embed = model.get_embedding()
    elif embed_method == "fairwalk":
        if implementation == "singer":
            n = len(graph.nodes())
            node2group = {
                node: group
                for node, group in zip(
                    graph.nodes(), (5 * np.random.random(n)).astype(int)
                )
            }
            nx.set_node_attributes(graph, node2group, "group")

            # Precompute probabilities and generate walks
            # TODO CHANGE HYPERPARAMETERS TO THE ONES FROM THE CROSSWALK PAPER
            # FOR NOW IT'S THE SAME AS THE DEFAULT KARATECLUB DEEPWALK ONES
            model = FairWalk(graph, workers=4)

            # Get embedding
            model = model.fit()

            # Save embeddings as numpy array
            embed = model.wv.vectors.copy()

    return embed


def get_embedding_path(
    dataset: str, reweight_method: str, embed_method: str, implementation: str
):
    """
    Construct path to the embeddings files following our conventions
    """
    path = f"../embeddings/{dataset}/{dataset}_{reweight_method}_{embed_method}_{implementation}"
    return path


def save_embed(
    embed: np.array,
    dataset: str,
    reweight_method: str,
    embed_method: str,
    implementation: str,
):
    """
    Saves an embedding to the appropriate directionary and file.
    TODO Maybe hyperparameter info like the number of embedding dimensions should be added
    For now without pickle for possible compatability issues.
    """
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
        implementation=implementation,
    )
    path = get_embedding_path(dataset, reweight_method, embed_method, implementation)
    np.save(path, embed, allow_pickle=False)


def load_embed(
    dataset: str, reweight_method: str, embed_method: str, implementation: str
):
    """
    load an embedding.
    """
    check_input_formatting(
        reweight_method=reweight_method,
        embed_method=embed_method,
        implementation=implementation,
    )
    path = get_embedding_path(dataset, reweight_method, embed_method, implementation)
    # if implementation == "perozzi":
    #     # this doesn't work yet, since the perozzi implementation seems to discard unconnected nodes
    #     # which messes up the (re-)indexing
    #     path = get_embedding_path(dataset, reweight_method, embed_method, implementation)
    #     embed = np.genfromtxt(
    #         path, dtype=np.single, skip_header=1, usecols=np.arange(1, 65)
    #     )
    #     indices = np.genfromtxt(path, dtype=np.uintc, skip_header=1, usecols=[0])
    #     # sort the embeddings
    #     embed = embed[indices]
    # else:
    embed = np.load(path + ".npy", allow_pickle=False)

    return embed
