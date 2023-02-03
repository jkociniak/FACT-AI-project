# FACT-AI-project
Paper reproduction for FACT-AI course at UvA.

To install `walker', run:
pip install pybind11
pip install graph-walker

## Environment setup (Mac M1)

```
conda update -n base -c conda-forge conda
conda create -n fact-ai python=3.10.8
conda activate fact-ai
conda install networkx gensim numpy tqdm 
conda install joblib
conda install karateclub
cd fairwalk
pip install setuptools==58.2.0
python setup.py install
```

## Description of the files
- `embed_utils.py` contains the functions to load datasets and to generate embeddings. You can set the hyperparameters of graph embedding models here.

### Influence Maximization
- `independent_cascade` is a folder containing both sequential and parallel implementations for Independent Cascade model.
- `parallel_ic_tuning.py` contains benchmark for parallel implementation for Independent Cascade. If you want to run the parallel IC, you should adjust map batch size and reduce batch size accordingly to obtain proper speedup.
- `influence_maximization.py` contains the code to run influence maximization experiments
- `influence_maximization_barplots.ipynb` contains the code to generate the plots for influence maximization experiments
- `generate_greedy_seeds.py` contains the code to generate seeds via greedy algorithm. It should work out of the box, but you might want to adjust map batch size and reduce batch size as if in `parallel_ic_tuning.py`.

### Link Prediction
- `link_prediction.py` contains the code to run link prediction experiments
- `link_prediction_barplots.ipynb` contains the code to generate the comparison plots for link prediction experiments

### Node Classification
- `node_classification.ipynb` contains the code to both run node classification experiments on Rice dataset and plot the comparison.

### Extension
- `generate_greedy_seeds.py` contains the code to generate synthetic graphs described in extension part of our reproduction study.
- `generate_graph.py` contains code to generate graph with control over the assortativities. 

## Reproduction of influence maximization / link prediction experiments

1. Use the file `influence_maximization.py`, `link_prediction.py` or `link_prediction_extension.py` to generate the data for the experiments. Use the `save_dir` variable defined in the main function of the program to save your results in chosen directory.
2. Use the corresponding notebook to generate the comparison plots using results from step 1. To do it, you just need to set the results directories in the second cell of the notebook. Use the right function for the extension at the bottom. 
