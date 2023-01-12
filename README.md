# FACT-AI-project
Paper reproduction for FACT-AI course at UvA.


The code associated with the paper contains a fork of the original implementation of Deepwalk on the github (https://github.com/phanein/deepwalk/) of an author of the original paper on Deepwalk (https://arxiv.org/abs/1403.6652). The fork is messy and it is not yet clear that is necessary, so here is how you can use the original version based on the original README.rst and experimentation with different versions of the packages. We chose to find a working combination of old packages instead of trying to make the code compatible with the newest versions of the modules.

clone https://github.com/phanein/deepwalk.git and go to the dir

$conda create -n dpwalk python=3.6

$conda activate dwalk

change requirements.txt to: 
wheel>=0.23.0
Cython>=0.20.2
six>=1.7.3
gensim==1.0.0
scipy==1.2.0
psutil>=2.1.1
networkx>=2.0

$pip install -r requirements.txt

'$conda list' should give something like this:
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
ca-certificates           2022.10.11           h06a4308_0  
certifi                   2021.5.30        py36h06a4308_0  
cython                    0.29.33                  pypi_0    pypi
decorator                 4.4.2                    pypi_0    pypi
gensim                    1.0.0                    pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
ncurses                   6.3                  h5eee18b_3  
networkx                  2.5.1                    pypi_0    pypi
numpy                     1.19.5                   pypi_0    pypi
openssl                   1.1.1s               h7f8727e_0  
pip                       21.2.2           py36h06a4308_0  
psutil                    5.9.4                    pypi_0    pypi
python                    3.6.13               h12debd9_1  
readline                  8.2                  h5eee18b_0  
scipy                     1.2.0                    pypi_0    pypi
setuptools                58.0.4           py36h06a4308_0  
six                       1.16.0                   pypi_0    pypi
smart-open                6.3.0                    pypi_0    pypi
sqlite                    3.40.1               h5082296_0  
tk                        8.6.12               h1ccaba5_0  
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.2.8                h5eee18b_0  
zlib                      1.2.13               h5eee18b_0 

$python setup.py install

In general:
$deepwalk --input {INPUT_PATH(.links)} --format edgelist --output {OUTPUT(.embeddings)}

For example for the second synthetic dataset (note that the file path may have changed when you read this):
$deepwalk --input ../data/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.links --format edgelist --output synth2.embeddings
You should now have a file 'synth2.embeddings' in the dir
