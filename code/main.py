from graph import Graph
import networkx as nx
import karateclub
from data import *


def load_data():
    """
    Loads data
    """

    with open("./data/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr") as f:

        nodes = f.read()

    with open("./data/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.links") as f:

        links = f.read()

    data = {
        "nodes": nodes,
        "links": links
    }

    return data


def make_graph(data) -> Graph:

    nx.DiGraph(data["links"])
    graph = None
    return graph


def reweight_graph(graph) -> Graph:

    return


def embed_graph(graph):

    return 


def evaluate(ting):
    return ting


def main():

    # Load data
    data = load_data()

    # Make a graph of the data
    graph = make_graph(data)

    # Crosswalk / nothing for reweighting the graph
    pass

    # Embed the graph to nodes with a walk
    graph_embedded = embed_graph(graph)

    # Evaluate fairness and accuracy on the embeddings
    evaluate(ting=graph_embedded)

    return
