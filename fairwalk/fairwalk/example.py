import numpy as np
import networkx as nx
from fairwalk import FairWalk
from os import listdir
from os.path import splitext

def raw_to_graph_format(attr_str, links_str):
    # list of tuples
    attr = [(int(i), int(c)) for node in attr_str.strip().split('\n') for i, c in [node.split()]]
    links = [(int(i1), int(i2)) for node in links_str.strip().split('\n') for i1, i2 in [node.split()]]
    
    # Make the mapping
    mapping = {}
    new_node = 0
    for node, label in attr:
        if node not in mapping.keys():
            mapping[node] = new_node
            new_node += 1
        else:
            raise ValueError("Duplicate node in attr data")
            
    # Map to ordered and complete
    # Attr
    attr_oac = [] 
    for node, label in attr:
        attr_oac.append((mapping[node], {"class": label}))
        
    # Links
    links_oac = [] 
    for node1, node2 in links:
        links_oac.append((mapping[node1], mapping[node2]))
    
    return attr_oac, links_oac


def data2graph(dataset: str):
    # check_input_formatting(dataset=dataset)
    path = f"{dataset}/"
    path += splitext(listdir(path)[0])[0]
    # TODO hardcode coupling attribute_name to the respective dataset within this function
        # but maybe we can all give them the same name?
    # TODO? Datasets may have more than one attribute (or is that case not relevant here?)
    with open(path + ".attr") as f_attr, open(path + ".links") as f_links:
        attr, links = raw_to_graph_format(f_attr.read(), f_links.read())

    G = nx.Graph()
    G.add_nodes_from(attr)
    G.add_edges_from(links)
    return G

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'

# Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph = data2graph("rice")
n = len(graph.nodes())
node2group = {node: group for node, group in zip(graph.nodes(), (5*np.random.random(n)).astype(int))}
nx.set_node_attributes(graph, node2group, 'group')

# Precompute probabilities and generate walks
model = FairWalk(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
# fairwalk = FairWalk(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = model.fit(window=10, min_count=1,
                  batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the FairWalk constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
