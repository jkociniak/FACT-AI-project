import numpy as np
import numpy.ma as ma
import networkx as nx
from tqdm import tqdm
import random as rd

import embed_utils

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, random

from karateclub import DeepWalk

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

def get_positive_links(G):
    """
    Output: 
    List of tuples (node1, node2, label, connected_bool)
    node1 is the node where the link starts
    node2 is the node where the link ends
    label is the type of connection (e.g. for Rice we have 3 types: 0,0 1,1 or 1,0)
    connected_bool is a boolean that determines if the link is positive or negative (connected or unconnected nodes)
    """
    label_pairs = get_label_pairs(G)

    positive_links = []
    for link in tqdm(list(G.edges)):
        for i, pair in enumerate(label_pairs):
                if G.nodes(embed_utils.SENSATTR)[link[0]] == pair[0] and G.nodes(embed_utils.SENSATTR)[link[1]] == pair[1]:
                    label = i
                elif G.nodes(embed_utils.SENSATTR)[link[1]] == pair[0] and G.nodes(embed_utils.SENSATTR)[link[0]] == pair[1]:
                    label = i
        
        positive_links.append(link + (label, 1))

    positive_links_partitioned = []
    for group in tqdm(range(len(label_pairs))):
        # Isolate group data
        positive_links_partitioned.append([d for d in positive_links if d[2] == group])

    return positive_links_partitioned


def train_test_from_pos_links(G, pos_links_groups):
    label_pairs = get_label_pairs(G)

    train, test = [], []
    
    train_groups = []
    test_groups = []
    # Split the postive links in test and train per group and append together
    for pos_group_x in pos_links_groups:
        # Split 1:10 test:train per group
        pos_train, pos_test = train_test_split(shuffle(pos_group_x), test_size=0.1)
        # Add alle train and test data together
        train += pos_train
        test += pos_test

        train_groups.append(pos_train)
        test_groups.append(pos_test)

    n = G.number_of_nodes()
    # Keep track of links we don't want as negative train and test samples
    checked_links = [(link[0], link[1]) for link in train] + [(link[1], link[0]) for link in train]

    # Get negative samples for train data
    negative_per_group = [0 for _ in range(len(label_pairs))]
    maxs_per_group = [len(group) for group in train_groups]
    for _ in tqdm(range(len(train))):
        not_valid = True
        # Get negative links
        while not_valid:
            # Randomly sample negative links
            node1_ix = rd.randint(0, n-1)
            node2_ix = rd.randint(0, n-1)
            if (node2_ix, node1_ix) in checked_links or (node1_ix, node2_ix) in checked_links or node1_ix == node2_ix:
                continue
            # Get the correct label
            for i, pair in enumerate(label_pairs):
                if G.nodes(embed_utils.SENSATTR)[node1_ix] == pair[0] and G.nodes(embed_utils.SENSATTR)[node2_ix] == pair[1]:
                    label = i
                elif G.nodes(embed_utils.SENSATTR)[node1_ix] == pair[1] and G.nodes(embed_utils.SENSATTR)[node2_ix] == pair[0]:
                    label = i

            # Append negative per corresponding group if not already maximum is achieved
            for group_name, (group_i, max_per_group) in enumerate(zip(negative_per_group, maxs_per_group)):
                if label == group_name:
                    if group_i < max_per_group:
                        train.append((node1_ix, node2_ix, label ,0))
                        checked_links.append((node1_ix, node2_ix))

                        maxs_per_group[group_name] += 1
                        not_valid = False


    # For the test data we cannot sample from test data itself
    checked_links+= [(link[0], link[1]) for link in test]
    
    # Get negative samples for test data
    negative_per_group = [0 for _ in range(len(label_pairs))]
    maxs_per_group = [len(group) for group in test_groups]
    for _ in tqdm(range(len(test))):
        not_valid = True
        # Get negative links
        while not_valid:
            # Randomly sample negative links
            node1_ix = rd.randint(0, n-1)
            node2_ix = rd.randint(0, n-1)
            if (node2_ix, node1_ix) in checked_links or (node1_ix, node2_ix) in checked_links or node1_ix == node2_ix:
                continue
            # Get the correct label
            for i, pair in enumerate(label_pairs):
                if G.nodes(embed_utils.SENSATTR)[node1_ix] == pair[0] and G.nodes(embed_utils.SENSATTR)[node2_ix] == pair[1]:
                    label = i
                elif G.nodes(embed_utils.SENSATTR)[node1_ix] == pair[1] and G.nodes(embed_utils.SENSATTR)[node2_ix] == pair[0]:
                    label = i
                # Append negative per corresponding group if not already maximum is achieved
                for group_name, (group_i, max_per_group) in enumerate(zip(negative_per_group, maxs_per_group)):
                    if label == group_name:
                        if group_i < max_per_group:
                            test.append((node1_ix, node2_ix, label ,0))
                            checked_links.append((node1_ix, node2_ix))

                            maxs_per_group[group_name] += 1
                            not_valid = False
        
    return shuffle(train), shuffle(test)

   
def get_node_labels(G):
    """
    Return list of different labels 
    """
    nodes = list(G.nodes(data=embed_utils.SENSATTR))
    distinct_labels = list(set([n[1] for n in nodes]))
    
    return distinct_labels

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])


if __name__ == "__main__":
    datasets = ["rice", "twitter"]
    for dataset in datasets:
        # Get graph
        G = embed_utils.data2graph(dataset)

        # LOG: Find the different class labels
        all_labels = get_node_labels(G)
        # LOG: Print the different type of connection groups
        label_pairs = [(all_labels[i], all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
        print(f"For {dataset} dataset, we have {len(label_pairs)} groups where")
        for i, pair in enumerate(label_pairs):
            print(f"Group {i} is connection type {pair}")
        
        # Extract all positive links as list of positive links per group
        pos_links = get_positive_links(G)
        
        ratios = []
        for (reweight_method, embed_method) in [("default", "deepwalk"), ("fairwalk", "deepwalk"), ("crosswalk", "deepwalk")]:
            results = {"average weighted accuracy": [], "average disparity": []}
            accuracy = {"accuracy per iteration": [], "accuracy per group": []}
            # Train and test
            for iter in [str(k) for k in range(1,6)]:
                print('iter: ', iter)
                
                # Split in train and test
                train, test = train_test_from_pos_links(G, pos_links)

                pos_train_links = [(link[0], link[1]) for link in train if link[3]]
                
                new_G = G.copy()
                new_G.remove_edges_from(G.edges())
                # Add train edges
                new_G.add_edges_from(pos_train_links, weight=1)

                emb = embed_utils.graph2embed(new_G, reweight_method, embed_method)
                # model = DeepWalk()
                # model.fit(new_G)
                # emb = model.get_embedding()

                clf = LogisticRegression(solver='lbfgs')

                x_train_group_indices = np.array([l[2] for l in train])
                x_train = np.array([extract_features(emb[l[0]], emb[l[1]]) for l in train])
            
                y_train = np.array([l[3] for l in train])

                x_test_group_indices = np.array([l[2] for l in test])
                x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in test])
                
                y_test = np.array([l[3] for l in test])

                clf.fit(x_train, y_train)
                
                y_pred = clf.predict(x_test)
                
                accuracy["accuracy per iteration"].append(100 * np.sum(y_test == y_pred) / x_test.shape[0])
                
                accuracy_iter = []
                accuracy_group_xs = 0.0
                disparity_list = []
                accuracy["accuracy per group"] = []
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
                results["average weighted accuracy"].append(accuracy_group_xs / x_test.shape[0])
                accuracy["disparity"] = np.var(disparity_list)
                results["average disparity"].append(np.var(disparity_list))
                accuracy["accuracy per group"].append(accuracy_iter)


                print("iteration:", iter)
                
                print()
            print("accuracy report:\n", accuracy)
            print()
            print(f"average weighted accuracy for {dataset}:\n      {np.mean(results['average weighted accuracy'])}")
            print(f"variance for weighted accuracy for {dataset}:\n      {np.var(results['average weighted accuracy'])}")
            print(f"average disparity for {dataset}:\n      {np.mean(results['average disparity'])}")
            print(f"variance disparity for {dataset}:\n      {np.var(results['average disparity'])}")

            print(f"For reweight method {reweight_method} and embed method {embed_method}\nRatio of accuracy/disparity is {np.mean(results['average disparity'])/np.mean(results['average weighted accuracy'])}")
            
            acc = np.mean(results['average weighted accuracy'])
            disp = np.mean(results['average disparity'])
            print(f"\nVisual representation of accuracy/disparity ratio:")
            print(f"Accuracy :{'#'*int(acc/(acc+disp)*50)}")
            print(f"Disparity:{'#'*int(disp/(acc+disp)*50)}")
            print()
