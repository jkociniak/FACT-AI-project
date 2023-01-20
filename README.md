# FACT-AI-project
Paper reproduction for FACT-AI course at UvA.

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