import networkx as nx
import numpy
import karateclub
import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm

from embed_utils import *

from sklearn.linear_model import LogisticRegression

def adj_matrix(G):
    """
    Create adjacency matrix from graph
    """
    nodelist = list(G.nodes)
    M = nx.to_numpy_matrix(G, nodelist)
    return M

if __name__ == "__main__":

    # Create Graph
    G = data2graph("synth2")

    # Create adjancecy matrix
    adj_G = adj_matrix(G)
    print(adj_G)
