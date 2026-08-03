# BoxLitE

This repository contains the official source code for the BoxLitE model, presented at **KR 2026** in our paper "**BoxLitE: A Faithful Knowledge Base Embedding Based
on Convex Optimization**". The repository includes:

1. The datasets F_v1-F_v4 derived from the family dataset [Imenes et al., 2023],
2. The implementation of BoxLitE,
3. The code for training and testing BoxLitE on F_v1-F_v4 to reproduce the knowledge base completion (KBC) results of our paper,
4. The code for visualizing learned BoxLitE embeddings,
5. The code to train and evaluate standard knowledge base embedding models (KBEs) that are based on stochastic-gradient descent (SGD), and
6. An `environment.yml` to automatically set up a conda environment with all dependencies.

# Requirements

* Python 3.12
* CVXPY 1.7.2
* Mosek 11.0.30 (requires a valid academic or commercial license)
* Matplotlib >= 3.10.0
* tqdm >= 4.67.1


# Installation

We have provided an `environment.yml` file that can be used to create a conda environment with all required
dependencies. Simply run `conda env create -f environment.yml` to create the conda environment `BoxLitE`.
Afterwards, use `conda activate BoxLitE` to activate the environment before running our experiments.

# Datasets F_v1-F_v4

We have derived four datasets (F_v1-F_v4) of increasing sizes from the family dataset [Imenes et al., 2023].
We use these datasets for evaluating the performance and scalability of BoxLitE. 
For reference and comparison, we provide the original family dataset in the directory `Family_Dataset/original`.
Furthermore, the `Family_Dataset` directory holds the `train`, `validation`, and `test` folders that contain the
respective splits of the derived datasets F_v1-F_v4.

# Running BoxLitE

Training and evaluation of BoxLitE are done by running the `main.py` file. In particular, a configuration file
must be specified for a BoxLitE model, containing all model, training, and evaluation parameters. The best
configuration files for F_v1-F_v4 are provided in the `Configurations` directory and 
can be adapted to try out different parameter configurations. 
To run an experiment, the following parameters need to be specified:

- `config` contains the path to the model configuration (e.g., `config=Configurations/F_v1.json`).
- `plot` specifies whether the optimized BoxLitE embedding is visualized (`plot=true`) or not (`plot=false`).

Finally, execute `python main.py config=<config> plot=<plot>` to run an experiment, 
where the angle brackets indicate parameter values.

# Reproducing the Results

In the following, we provide the commands to reproduce the results of our paper:

## F_v1-F_v4 Benchmarks

To reproduce BoxLitE's KBC results reported in Table 2, simply run the following commands:

* `python main.py plot=false config=Configurations/F_v1.json`
* `python main.py plot=false config=Configurations/F_v2.json`
* `python main.py plot=false config=Configurations/F_v3.json`
* `python main.py plot=false config=Configurations/F_v4.json`

Each of these commands will produce the following `.tsv` files in the folder `Benchmarking/BoxLitE/<exp_name>` 
(where `<exp_name>` represents the value of the corresponding parameter in the configuration file,
e.g., `<exp_name> = F_v1` for the configuration `F_v1.json`):

* `Train_result_table` represents the evaluation result on the train set, i.e., the ABox, to quantify its KBC performance.
* `Validation_result_table` represents the evaluation result on the validation set used to select the best embedding solution during hyperparameter optimization.
* `Test_result_table` represents the evaluation result on the test set, used for quantifying how well the found BoxLitE embedding solution learns to reason over the corresponding knowledge base assertions.
  
## Running SGD Models

Hyperparameter optimization of KBEs based on stochastic-gradient descent (SGD) is performed by running the `train_SGD_model.py` file. 
In particular, the following parameters need to be specified:
* `model` contains the name of the KBE (e.g., `BoxE`, `ComplEx`, or `RotatE`)  that shall be optimized.
* `dataset` contains the name of the dataset for the hyperparameter optimization (i.e., `F_v1`, `F_v2`, `F_v3`, or `F_v4`).

Run `python train_SGD_model.py model=<model> dataset=<dataset>` to perform the hyperparameter optimization of the specified model on the specified dataset (e.g., `python train_SGD_model.py model=BoxE dataset=F_v1`).
On completion, the parameters of the best-performing model on the validation set are stored in the Benchmarking directory under `Benchmarking/<model>/<dataset>/best_pipeline`.
Finally, to reproduce the KBC results of Table 2 and, thus, to evaluate the best-performing `BoxE`, `ComplEx`, and `RotatE` model on the test datasets of F_v1-F_v4, run `python eval_SGD_model.py model=<model> dataset=<dataset>`. The evaluation script runs each model three times with different seeds, reports the performance metrics together with their mean and standard deviation over the three runs, and stores these results under `Benchmarking/<model>/<dataset>/best_pipeline/rep_results/rep_metrics.json`.

# Citation 

If you use this code or its corresponding paper, please cite our work as follows:

```
@inproceedings{
lourenco2026boxlite,
title={BoxLitE: A Faithful Knowledge Base Embedding Based on Convex Optimization},
author={Bruno F. Louren\c{c}o, Hesham Morgan, Ana Ozaki, Aleksandar Pavlovi{\'c}, Emanuel Sallinger},
booktitle={Proceedings of the 23rd International Conference on Principles of
           Knowledge Representation and Reasoning, {KR} 2026, Lisbon, Portugal.
           July 20-23, 2026},
year={2026}
}
```

# Contact

Aleksandar Pavlović

Research Center AI, Software and IT-Security

University of Applied Sciences Vienna (HCW)

Vienna, Austria

<aleksandar.pavlovic@hcw.ac.at>

# Licenses

Both this project and PyKEEN are released under the MIT License.

Copyright (c) 2026 Aleksandar Pavlović

# References
[Imenes et al., 2023] Anders Imenes, Ricardo Guimaraes, and Ana Ozaki. Marrying Query Rewriting and Knowledge Graph Embeddings. In RuleML+RR, pages 126-140, Berlin, Heidelberg, 2023. Springer-Verlag
