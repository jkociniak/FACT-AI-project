"""
Code for generating new synthetic graphs for the extension
TODO Allow  the sensitive and target attribute to be more than binary?
TODO integrate sensitive and target attribute distribution with the topology generation

Useful literature for background:
https://pdodds.w3.uvm.edu/files/papers/others/2002/newman2002a.pdf

Useful literature for constructing synthetic graphs:
https://www.worldscientific.com/doi/10.1142/S0218127408022536
https://arxiv.org/abs/1909.03448
https://www.win.tue.nl/~rhofstad/NotesRGCN.pdf:

Possible approaches:
-configuration model
    -repeated configuration model
    -erased configuration model
-We base or model on https://www.nature.com/articles/s41598-022-22843-4
-And maybe we'll also use 

"""
import numpy as np
import networkx as nx
import embed_utils
import random


def generate_graph(n, p_within=0.75, p_across=0.1):
    """
    Generate a graph with n nodes, controlling the correlation between the sensitive and target attributes
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

    return G


# taken from https://github.com/almusawiaf2/Identifying-Accurate-Link-Predictors-based-on-Assortativity-of-Complex-Networks/blob/main/Code/assort3.py
# Corresponding paper: https://www.nature.com/articles/s41598-022-22843-4
def findK(G, N, D, km):
    E = int(len(G.edges()) - (D * N * (N - 1) / 2.0))
    # Changed round to np.floor
    diff2 = -int(np.floor(float(E) / float(N - len(G))))
    # Added if else to handle case where diff2==0
    if diff2 > 0:
        k = random.randint(1, min(diff2, km))
    else:
        k = 1
    return k


def gen(N, diff, mode, D, n0=10, km=250):

    G = nx.complete_graph(n0)

    for t in range(1, N - n0 + 1):
        # k = random.randint(1, km)
        k = findK(G, N, D, km)

        L = {u: abs(G.degree(u) - k) for u in G.nodes()}
        # print (L)

        s = []
        for i in range(max(L.values()) + 1):
            s.extend([u for u in G.nodes() if L[u] == i])

        ind = int(diff * len(s))

        if mode == 0:
            s = s[ind:]
        if mode == 1:
            s = s[:ind]
            s = s[::-1]

        # print ([L[u] for u in s])

        v = len(G)
        G.add_node(v)
        while G.degree(v) < k and len(s) > 0:

            u = s.pop(0)
            if G.degree(u) >= km:
                continue

            G.add_edge(u, v)

    return "Feasible input.", G


def density(G):
    return float(len(G.edges()) * 2) / float(len(G) * (len(G) - 1))


# i = 0
# while i < 1:
#     # Mode: 0 --> Assortative; Mode: 1 --> Disassortative
#     N = 100
#     mode = 0
#     diff = -0.99
#     D = 0.05
#     n0 = 1
#     report, G = gen(N, diff, mode, D, n0=n0)
#     if 'Infeasible' in report:
#         continue

#     print(len(G), len(G.edges()))
#     r = nx.degree_assortativity_coefficient(G)
#     d = density(G)

#     print(i, r, d, diff, len(G.edges()))
#     i = i + 1


# # Visualization
# # Taken from https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# dmax = max(degree_sequence)

# fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# # Create a gridspec for adding subplots of different sizes
# axgrid = fig.add_gridspec(5, 4)

# ax0 = fig.add_subplot(axgrid[0:3, :])
# # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# Gcc = G
# pos = nx.spring_layout(Gcc, seed=10396953)
# nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
# nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
# ax0.set_title("Connected components of G")
# ax0.set_axis_off()

# ax1 = fig.add_subplot(axgrid[3:, :2])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")

# ax2 = fig.add_subplot(axgrid[3:, 2:])
# ax2.bar(*np.unique(degree_sequence, return_counts=True))
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("# of Nodes")

# fig.tight_layout()
# plt.show()


# taken from https://networkx.org/documentation/stable/_modules/networkx/generators/community.html#stochastic_block_model
def stochastic_block_model(
    sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True
):
    """Returns a stochastic block model graph.

    This model partitions the nodes in blocks of arbitrary sizes, and places
    edges between pairs of nodes independently, with a probability that depends
    on the blocks.

    Parameters
    ----------
    sizes : list of ints
        Sizes of blocks
    p : list of list of floats
        Element (r,s) gives the density of edges going from the nodes
        of group r to nodes of group s.
        p must match the number of groups (len(sizes) == len(p)),
        and it must be symmetric if the graph is undirected.
    nodelist : list, optional
        The block tags are assigned according to the node identifiers
        in nodelist. If nodelist is None, then the ordering is the
        range [0,sum(sizes)-1].
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : boolean optional, default=False
        Whether to create a directed graph or not.
    selfloops : boolean optional, default=False
        Whether to include self-loops or not.
    sparse: boolean optional, default=True
        Use the sparse heuristic to speed up the generator.

    Returns
    -------
    g : NetworkX Graph or DiGraph
        Stochastic block model graph of size sum(sizes)

    Raises
    ------
    NetworkXError
      If probabilities are not in [0,1].
      If the probability matrix is not square (directed case).
      If the probability matrix is not symmetric (undirected case).
      If the sizes list does not match nodelist or the probability matrix.
      If nodelist contains duplicate.

    Examples
    --------
    >>> sizes = [75, 75, 300]
    >>> probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    >>> g = nx.stochastic_block_model(sizes, probs, seed=0)
    >>> len(g)
    450
    >>> H = nx.quotient_graph(g, g.graph["partition"], relabel=True)
    >>> for v in H.nodes(data=True):
    ...     print(round(v[1]["density"], 3))
    ...
    0.245
    0.348
    0.405
    >>> for v in H.edges(data=True):
    ...     print(round(1.0 * v[2]["weight"] / (sizes[v[0]] * sizes[v[1]]), 3))
    ...
    0.051
    0.022
    0.07

    See Also
    --------
    random_partition_graph
    planted_partition_graph
    gaussian_random_partition_graph
    gnp_random_graph

    References
    ----------
    .. [1] Holland, P. W., Laskey, K. B., & Leinhardt, S.,
           "Stochastic blockmodels: First steps",
           Social networks, 5(2), 109-137, 1983.
    """
    # Check if dimensions match
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException("nodelist contains duplicate.")
    else:
        nodelist = range(0, sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    for i, j in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])
        if sparse:
            if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
                for e in edges:
                    g.add_edge(*e)
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        # consume "skip" edges
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        g.add_edge(*e)  # __safe
                    except StopIteration:
                        break
        else:
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)  # __safe
    return g
