"""
Code for generating new synthetic graphs for the extension
TODO Allow  the sensitive and target attribute to be more than binary?
TODO integrate sensitive and target attribute distribution with the topology generation

Useful literature for background:
https://pdodds.w3.uvm.edu/files/papers/others/2002/newman2002a.pdf
https://www.nature.com/articles/s41598-022-22843-4

Useful literature for constructing synthetic graphs:
https://www.worldscientific.com/doi/10.1142/S0218127408022536
https://arxiv.org/abs/1909.03448
https://www.win.tue.nl/~rhofstad/NotesRGCN.pdf:

Possible approaches:
-configuration model
    -repeated configuration model
    -erased configuration model

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
