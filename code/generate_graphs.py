"""
Code for generating new synthetic graphs for the extension

Useful literature for background:
https://pdodds.w3.uvm.edu/files/papers/others/2002/newman2002a.pdf

Useful literature for constructing synthetic graphs:
https://www.worldscientific.com/doi/10.1142/S0218127408022536
https://arxiv.org/abs/1909.03448
https://www.win.tue.nl/~rhofstad/NotesRGCN.pdf:

"""
import numpy as np
import networkx as nx
import embed_utils
import random
import itertools


# findK, gen and density are taken from https://github.com/almusawiaf2/Identifying-Accurate-Link-Predictors-based-on-Assortativity-of-Complex-Networks/blob/main/Code/assort3.py
def findK(G, N, D, km):
    E = int(len(G.edges()) - (D * N * (N - 1) / 2.0))
    # Changed round to np.floor
    diff2 = -int(round(float(E) / float(N - len(G))))
    k = random.randint(1, min(diff2, km))
    # if diff2 > 0:
    #     k = random.randint(1, min(diff2, km))
    # else:
    #     k = 1
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


def gen_graph(N, diff, mode, D_within, n_edges_between_groups):
    # N: number nodes per group
    # Mode: 0 --> Assortative; Mode: 1 --> Disassortative
    # If mode = 0:  diff = 0 highest assortative
    # If mode = 1:  diff = 1.0 highest disassortative
    # D_within: edge densities within the group. Length of this list also determines the number of groups
    # n_edges_between_groups: number of edges between each group. Determines attribute assortativity
    groups = []
    for D in D_within:
        while True:
            report, group = gen(N, diff, mode, D, n0=1)
            # TODO handle endlessly unfeasible input better
            if "Feasible" in report:
                break
        groups.append(group)

    G = nx.Graph()

    all_edges = np.array(list(itertools.product(np.arange(N), np.arange(N, 2 * N))))
    for i, group in enumerate(groups):
        group = nx.convert_node_labels_to_integers(group, first_label=N * i)
        nx.set_node_attributes(group, i, embed_utils.SENSATTR)
        G = nx.compose(G, group)
        if i > 0:
            edges_between_groups = all_edges[
                np.random.randint(0, all_edges.shape[0], n_edges_between_groups)
            ]
            edges_between_groups += N * (i - 1)
            G.add_edges_from(edges_between_groups)

    return G


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
