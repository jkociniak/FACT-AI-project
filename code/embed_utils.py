import networkx as nx
from karateclub import DeepWalk


def data2graph(files_path):
    with open(files_path + ".attr") as f:
        attr = [(int(i)-1, {"class": int(c)}) for node in f.read().strip().split('\n') for i, c in [node.split()]]
    with open(files_path + ".links") as f:
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