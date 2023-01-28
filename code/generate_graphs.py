"""
Code for generating new synthetic graphs for the extension
TODO Allow  the sensitive and target attribute to be more than binary?
TODO add the graph topology with a measure of control
"""
import numpy as np
import networkx as nx
import embed_utils


def generate_graph(n, p_within=0.75, p_across=0.1):
    """
    Generate a graph with n nodes, controlling the correlation between the sensitive and target attributes
    and the way the graph topology relates to the (or of the?) attributes
    """

    theta_x = 0.5
    # I'm not sure yet how this is related to well known measures of correlation
    # This just determines the conditional probablities symmetrically
    # <0.5 is positive correlation
    # 0.5 is uncorrelated
    # >0.5 is negative correlation
    correlation = 0.5
    X = np.random.binomial(1, theta_x, n)
    theta_y = np.abs(X - correlation)
    Y = np.random.binomial(1, theta_y, n)
    G = nx.Graph()
    for i, (x, y) in enumerate(zip(X, Y)):
        G.add_node(i, **{embed_utils.SENSATTR: x, embed_utils.CLASS_NAME: y})

    # for i, j in itertools.combinations(np.arange(n), 2):
    #     if G.nodes[i][embed_utils.SENSATTR] == G.nodes[j][embed_utils.SENSATTR]:
    #         Y = np.random.binomial(1, p_within, 1)[0]
    #         if Y == 1:
    #             G.add_edge(i, j)
    #             G[i][j]['weight'] = np.random.uniform(0,0.1,1)[0]
    #     else:
    #         Y = np.random.binomial(1, p_across, 1)[0]
    #         if Y == 1:
    #             G.add_edges_from([(i,j)])
    #             G[i][j]['weight'] = np.random.uniform(0,.1,1)[0]

    return G
