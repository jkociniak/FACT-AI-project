from graph import Graph
from data import *


def load_data():
    """
    Loads data
    """

    data = load_syn_data()

    return data


def make_graph(data) -> Graph:

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
