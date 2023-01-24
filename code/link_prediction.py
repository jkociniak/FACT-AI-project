import numpy as np
import numpy.ma as ma
import networkx as nx
from tqdm import tqdm

import embed_utils

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
def get_label_pairs(G):
    """
    Return list of label pair as tuple
    E.g. for Rice dataset with attribute labels 0 and 1 it returns
    [(0,0), (0,1), (1,1)]
    """
    all_labels = get_node_labels(G)
    label_pairs = [(all_labels[i], all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
    
    return label_pairs

def get_train_links(G):
    """
    Output: 
    List of tuples (node1, node2, label, connected_bool)
    node1 is the node where the link starts
    node2 is the node where the link ends
    label is the type of connection (e.g. for Rice we have 3 types: 0,0 1,1 or 1,0)
    connected_bool is a boolean that determines if the link is positive or negative (connected or unconnected nodes)
    """
    all_labels = get_node_labels(G)
    label_pairs = [(all_labels[i], all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]

    connected_nodes = G.edges
    train_links = []
    nodes = list(G.nodes)
    
    checked_nodes = []
    for node1 in tqdm(nodes):
        for node2 in nodes:
            if node2 in checked_nodes or node1 == node2:
                continue
            
            for i, pair in enumerate(label_pairs):
                if G.nodes("class")[node1] == pair[0] and G.nodes("class")[node2] == pair[1]:
                    label = i

            if (node1,node2) in connected_nodes or (node2, node1) in connected_nodes:
                connected_bool = 1
            else:
                connected_bool = 0

            train_links.append((node1, node2, label, connected_bool))
        checked_nodes.append(node1)

    return train_links

def get_train_test(G, links_group):
    """
    Get the train and test split 
    Returns two list of tuples, same format as get_train_links
    [(node1, node2, label, connected_bool)]
    """
    pos_data = [link for link in links_group if link[3]]
    neg_data = [link for link in links_group if not link[3]]

    label_pairs = get_label_pairs(G)

    train = []
    test = []

    for group in range(len(label_pairs)):
        # Split 1:10 test:train per group
        pos_group_x = shuffle([d for d in pos_data if d[2] == group])
        neg_group_x = shuffle([d for d in neg_data if d[2] == group])

        pos_train, pos_test = train_test_split(pos_group_x, test_size=0.1)
        neg_train, neg_test = train_test_split(neg_group_x, test_size=0.1)

        # Add alle train and test data together
        train += pos_train + neg_train
        test += pos_test + shuffle(neg_test)[:len(pos_test)]

    return shuffle(train), shuffle(test)

def get_train_test_positive_only(G, links_group):
    """
    Get the train and test split 
    Returns two list of tuples, same format as get_train_links
    [(node1, node2, label, connected_bool)]
    """
    pos_data = [link for link in links_group if link[3]]

    label_pairs = get_label_pairs(G)

    train = []
    test = []

    for group in range(len(label_pairs)):
        # Split 1:10 test:train per group
        pos_group_x = shuffle([d for d in pos_data if d[2] == group])

        pos_train, pos_test = train_test_split(pos_group_x, test_size=0.1)

        # Add alle train and test data together
        train += pos_train 
        test += pos_test

    return shuffle(train), shuffle(test)
    
def get_node_labels(G):
    """
    Return list of different labels 
    """
    nodes = list(G.nodes(data="class"))
    distinct_labels = list(set([n[1] for n in nodes]))
    
    return distinct_labels

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])

def add_negative_links(G, pos_links):
    """
    Get as many negative as positive links and add it to the training dataset
    """
    all_links = get_train_links(G)
    # Remove positive links and make the postive links from test set negative
    neg_links = [link[:-1]+(0,) for link in all_links if link not in pos_links]
    # Shuffle and acces as many negative as positive links
    neg_links = shuffle(neg_links)[:len(pos_links)]

    return shuffle(pos_links + neg_links)


if __name__ == "__main__":

    ##################
    #####  Rice  #####
    ##################
    
    # Get embedding
    # emb = load_embed("rice", "fairwalk", "singer")
    G = embed_utils.data2graph("rice")
    
    # Find the different class labels
    all_labels = get_node_labels(G)

    # Print the different type of connection groups
    label_pairs = [(all_labels[i], all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
    print(f"We have {len(label_pairs)} groups where")
    for i, pair in enumerate(label_pairs):
        print(f"Group {i} is connection type {pair}")

    # Create the train links
    # For rice we have labels {0: link 0 to 0, 1: link 0 to 1, 1:link 1 to 1}
    all_links = get_train_links(G)

    # For every iteration
        # Verdeel in de classes

        # For every class get 90% of links as train
        # Divide in 10 percent test and 90 percent train
        # We take this together back as one graph
        
        # TRAIN:
        # Now we can also get all negative links that are not in train and keep seperate
        # We do this most efficient by just sampling random points and checking if it is not a positive link 
        # For the length of the positive links.
        
        # From training data links get graph embedding

            # Extract features for every link in training but remember the label for y_
        
        # TEST:

            # 
    
    accuracy = {"accuracy per iteration": [], "disparity per iteration": []}

    for iter in [str(k) for k in range(1,6)]:
        
        print('iter: ', iter)

        train, test = get_train_test_positive_only(G, all_links)
        train_all = add_negative_links(G, train)
        
        test_all = add_negative_links(G, test)

        new_G = nx.Graph()
        new_G.add_nodes_from(G, attr='class')

        emb = embed_utils.load_embed("rice", "default", "node2vec")# embed_utils.graph2embed(new_G, "default", 'fairwalk')

        clf = LogisticRegression(solver='lbfgs')

        x_train_group_indices = np.array([l[2] for l in train_all])
        x_train = np.array([extract_features(emb[l[0]], emb[l[1]]) for l in train_all])
    
        y_train = np.array([l[3] for l in train_all])

        x_test_group_indices = np.array([l[2] for l in test_all])
        x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in test_all])
        
        y_test = np.array([l[3] for l in test_all])

        clf.fit(x_train, y_train)
        
        y_pred = clf.predict(x_test)
        old = y_pred.copy()
        accuracy["accuracy per iteration"].append(100 * np.sum(y_test == y_pred) / x_test.shape[0])
        
        accuracy_iter = []
        accuracy_group_xs = 0.0
        disparity_list = []
        # Make mask with False for indices to drop and True for indices of group to keep
        for group in range(len(label_pairs)):
            mask_test = ma.masked_not_equal(x_test_group_indices, group)
            # mask_train = ma.masked_not_equal(x_train_group_indices, group)

            y_test_group_x = y_test[mask_test == group]
            y_pred_group_x = y_pred[mask_test == group]

            x_test_group_x = x_test[mask_test == group]
           
            accuracy_group_x = 100 * np.sum(y_test_group_x == y_pred_group_x) / x_test_group_x.shape[0]
            accuracy_group_xs += accuracy_group_x * x_test_group_x.shape[0]
            disparity_list.append(accuracy_group_x)
            accuracy_iter.append((f"group: {group}", accuracy_group_x))
        
        
        accuracy["weighted accuracy"] = accuracy_group_xs / x_test.shape[0]
        accuracy["disparity"] = np.var(disparity_list)
        accuracy["disparity per iteration"].append(accuracy_iter)

    print(accuracy)
    print()

    accuracy["mean accuracy"] = np.mean(accuracy["accuracy per iteration"])
    accuracy["standard deviation"] = np.std(accuracy["accuracy per iteration"])





