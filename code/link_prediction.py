import networkx as nx
import numpy as np
from tqdm import tqdm

from embed_utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""In the Rice-Facebook dataset with two groups of nodes, A
and B, there exist three types of links; A to A, B to B, and A
to B connections. Similarly, in the Twitter dataset with three
groups of nodes there exist six types of links. For each 
connection type, we select equal number of positive and negative 
test samples (10% of each group). We train a logistic regression 
on the embeddings obtained by FairWalk and
CrossWalk applied to DeepWalk (see Section 5 for more details). 
We report the average result over 5 runs. Figures 7a
and 7b illustrate the total accuracy and disparity according
to Eq. (2), for link prediction. It confirms the superiority of
CrossWalk over FairWalk in reducing disparity with a slight
decrease in accuracy.
"""
def get_train_links(G):
    """
    Output: 
    List of tuples (node1, node2, label1, label2, connected_bool)
    node1 is the node where the link starts
    node2 is the node where the link ends
    label1 is the "class" of node 1
    label2 is the "class" of node 2
    connected_bool is a boolean that determines if the link is positive or negative (connected or unconnected nodes)
    """
    connected_nodes = G.edges
    train_links = []
    nodes = list(G.nodes)
    
    checked_nodes = []
    for node1 in tqdm(nodes):
        for node2 in nodes:
            if node2 in checked_nodes or node1 == node2:
                continue

            label1 = G.nodes("class")[node1]
            label2 = G.nodes("class")[node2]
            
            if (node1,node2) in connected_nodes or (node2, node1) in connected_nodes:
                connected_bool = True
            else:
                connected_bool = False

            train_links.append((node1, node2, label1, label2, connected_bool))
        checked_nodes.append(node1)

    return train_links


def get_train_test(links_group):
    pos_data = [link for link in A if link[4]]
    neg_data = [link for link in A if not link[4]]
    
    pos = train_test_split(pos_data, test_size=0.1)
    neg = train_test_split(neg_data, test_size=0.1)

    train = shuffle(pos[0] + neg[0])
    test = shuffle(pos[1] + neg[1])

    return train, test


def read_sensitive_attr(G):
    sens_attr = dict()

    for (id, label) in G.nodes.items():
        # Why do they check if the node is in the embedding in their code?
        # if id is in embedding:
        sens_attr[id] = label["class"]
    return sens_attr

def get_groups(links):
    A = []
    B = []
    AB = []

    for link in links:
        if link[2] == 0 and link[3] == 0:
            A.append(link)
        elif link[2] == 1 and link[3] == 1:
            B.append(link)
        else:
            AB.append(link)

    return A, B, AB
    

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])


if __name__ == "__main__":
    
    all_labels = [0, 1]

    label_pairs = [str(all_labels[i]) + ',' + str(all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
    accuracy_keys = label_pairs + ['max_diff', 'var', 'total']

    accuracy = {k : [] for k in accuracy_keys}

    emb = load_embed("rice", "fairwalk", "singer")

    G = data2graph("rice")
    sens_attr = read_sensitive_attr(G)

    train_links = get_train_links(G)

    A, B, AB = get_groups(train_links) 


    for iter in [str(k) for k in range(1,6)]:
        
        print('iter: ', iter)

        # New train test set
        train_A, test_A = get_train_test(A)
        train_B, test_B = get_train_test(B)
        train_AB, test_AB = get_train_test(AB)

        for (key, train, test) in zip(label_pairs, [train_A, train_B, train_AB], [test_A, test_B, test_AB]):
            
            clf = LogisticRegression(solver='lbfgs')
            x_train = np.array([extract_features(emb[l[0]], emb[l[1]]) for l in train])
            y_train = np.array([l[4] for l in train])
            x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in test])
            y_test = np.array([l[4] for l in test])
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy[key].append(100 * np.sum(y_test == y_pred) / x_test.shape[0])

        last_accs = [accuracy[k][-1] for k in label_pairs]
        accuracy['max_diff'].append(np.max(last_accs) - np.min(last_accs))
        accuracy['var'].append(np.var(last_accs))

        print(accuracy)
        print()

    print(accuracy)
    print()
    for k in accuracy_keys:
        print(k + ':', np.mean(accuracy[k]), '(' + str(np.std(accuracy[k])) + ')')





