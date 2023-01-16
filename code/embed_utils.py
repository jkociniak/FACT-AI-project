import numpy as np
from os import listdir
from os.path import splitext
import networkx as nx
from karateclub import DeepWalk


def data2graph(dataset: str, attribute_name: str):
    path = f"../data/{dataset}/"
    path += splitext(listdir(path)[0])[0]
    # TODO? Datasets may have more than one attribute (or is that case not relevant here?)
    with open(path + ".attr") as f:
        attr = [(int(i)-1, {attribute_name: int(c)}) for node in f.read().strip().split('\n') for i, c in [node.split()]]
    with open(path + ".links") as f:
        links = [(int(i0)-1, int(i1)-1) for edge in f.read().strip().split('\n') for i0, i1 in [edge.split()]]
    
    G = nx.Graph()
    G.add_nodes_from(attr)
    G.add_edges_from(links)
    return G


def graph2embed(graph, method="karateclub"):
    if method=="karateclub":
        model = DeepWalk()
        model.fit(graph)
        embed = model.get_embedding()
        return embed
    else:
        raise Exception("Unknown method. Try 'karateclub'.")


def save_embed(embed: np.array, dataset: str, method: str, implementation: str):
    """"
    Saves an embedding to the appropriate directionary and file. 
    TODO Maybe hyperparameter info like the number of embedding dimensions should be added
    For now without pickle for possible compatability issues. 
    """
    path = f"embeddings/{dataset}/{dataset}_{method}_{implementation}"
    np.save(path, embed, allow_pickle=False)


def load_embed(dataset: str, method: str, implementation: str):
    """"
    load an embedding. 
    """
    path = f"embeddings/{dataset}/{dataset}_{method}_{implementation}"
    if implementation=="perozzi":
        # this doesn't work yet, since the perozzi implementation seems to discard unconnected nodes
        path += ".embeddings"
        embed = np.genfromtxt(path, dtype=np.single, skip_header=1, usecols=np.arange(1,65))
        indices = np.genfromtxt(path, dtype=np.uintc, skip_header=1, usecols=[0])
        # sort the embeddings
        embed = embed[indices]
    else:
        embed = np.load(path + ".npy", allow_pickle=False)

    return embed