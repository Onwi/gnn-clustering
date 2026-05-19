# Pooling GNNs for Genomic Data Classification

## Installation
For reproducing the results
* It is recommended that a new virtual environment with conda is used. We executed the experiments with python 3.10 and conda.
* Install pytorch and related libraries (torchvision and torchaudio). We've used pytorch version 1.12.1. Get the correct version depending on your CUDA toolkit version (or if you are only using CPU) in https://pytorch.org/get-started/previous-versions/#v1121
* Install pytorch geometric https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
* Install pytorch scatter https://github.com/rusty1s/pytorch_scatter with `conda install pytorch-scatter -c pyg`
* Install other requirements running from inside the project folder `pip install -r requirements.txt`
* Install this project's library running `pip install -e .`

## Testing
Check if everything was installed properly by running a helper, like
`python scripts/experiments/coarsening_levels.py --help`.

Download example data from https://drive.google.com/file/d/1I06IRIO5htdmA-dhfBPV1I5IgqKmdjtC/view?usp=sharing.
Extract the files.
The contents include an in-folder dataset of BRCA subtype classification and pytorch tensors with graph information.
Links to these should be passed to the scripts, as in the example below.
Now test by executing the training of a fully-connected neural network and a GNN with 1 level of pooling for classifying BRCA subtypes. 
The script below tunes the hyperparameters of the model using random search and then train a final model using the best parameters found.

`
python scripts/experiments/coarsening_levels.py <absolute_path_to_data_dir>/tcga_brca_subtypes_classification <absolute_path_to_data_dir>/networks/levels/ --tune --max-n-levels 2 --n-cycles 1 --path-output outputs_pooling_env --n-holdouts 1
`
Note that it is necessary to use the absolute paths for raytune to work.
Add the argument `--cuda` to train using cuda.

## Data Files
networks: files related to the generation of the STRINGDB networks used in this work. networks/levels contain the multiple level descriptions used for the hierarchical gnn.

## Scripts
`generate_graph_levels.py`: uses the previously computed stringdb and the genes in the selected dataset to construct multiple levels of the graph, including edge weights, indices and the parents of the nodes at each level.

`experiments/coarsening_levels.py`: compare the performance of graph networks with pure pooling and weighted pooling approaches and with fully-connected neural networks.

TODO