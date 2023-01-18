import numpy as np
from os import listdir
from os.path import splitext
import networkx as nx
from karateclub import DeepWalk

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
    if "method" in kwargs:
        assert kwargs["method"] in [
            "deepwalk",
            "fairwalk",
        ], "method should be 'deepwalk' or 'fairwalk' (for now)"
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
            (i, {"class": int(c)})
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


def graph2embed(graph, method="deepwalk", implementation="karateclub"):
    check_input_formatting(method=method, implementation=implementation)
    if method == "deepwalk":
        if implementation == "karateclub":
            # TODO CHANGE HYPERPARAMETERS TO THE ONES FROM THE CROSSWALK PAPER
            model = DeepWalk()
            model.fit(graph)
            embed = model.get_embedding()
    elif method == "fairwalk":
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


def get_embedding_path(dataset: str, method: str, implementation: str):
    """
    Construct path to the embeddings files following our conventions
    """
    path = f"../embeddings/{dataset}/{dataset}_{method}_{implementation}"
    return path


def save_embed(embed: np.array, dataset: str, method: str, implementation: str):
    """
    Saves an embedding to the appropriate directionary and file.
    TODO Maybe hyperparameter info like the number of embedding dimensions should be added
    For now without pickle for possible compatability issues.
    """
    check_input_formatting(
        dataset=dataset,
        method=method,
        implementation=implementation,
    )
    path = get_embedding_path(dataset, method, implementation)
    np.save(path, embed, allow_pickle=False)


def load_embed(dataset: str, method: str, implementation: str):
    """
    load an embedding.
    """
    check_input_formatting(
        dataset=dataset,
        method=method,
        implementation=implementation,
    )
    path = get_embedding_path(dataset, method, implementation)
    if implementation == "perozzi":
        # this doesn't work yet, since the perozzi implementation seems to discard unconnected nodes
        # which messes up the (re-)indexing
        path += ".embeddings"
        embed = np.genfromtxt(
            path, dtype=np.single, skip_header=1, usecols=np.arange(1, 65)
        )
        indices = np.genfromtxt(path, dtype=np.uintc, skip_header=1, usecols=[0])
        # sort the embeddings
        embed = embed[indices]
    else:
        embed = np.load(path + ".npy", allow_pickle=False)

    return embed
