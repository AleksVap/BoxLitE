import sys

from pykeen.hpo import hpo_pipeline
import random
import numpy as np
import torch
import pykeen.hpo.hpo as _hpo_module
from pykeen.pipeline import pipeline as _original_pipeline
from Utils import parse_kwargs

SEED = 6934

def _seeded_pipeline(**kwargs):
    kwargs.setdefault('random_seed', SEED)
    return _original_pipeline(**kwargs)

_hpo_module.pipeline = _seeded_pipeline

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def optimize_BoxE(KGE_model, dataset):
    set_seeds(SEED)
    train_path = "./Family_Dataset/train/" + dataset + ".tsv"
    val_path = "./Family_Dataset/validation/" + dataset + ".tsv"
    test_path = "./Family_Dataset/test/" + dataset + ".tsv"
    result = hpo_pipeline(
        n_trials=100,
        training=train_path,
        testing=test_path,
        validation=val_path,
        model=KGE_model,
        model_kwargs=dict(embedding_dim=32),
        epochs=500,
        stopper='early',
        stopper_kwargs=dict(frequency=10, patience=10, relative_delta=0.01),
    )
    result.save_to_directory("Benchmarking/" + KGE_model + "/" + dataset)

if __name__ == '__main__':
    model, dataset = parse_kwargs(**dict(arg.split('=') for arg in sys.argv[1:]))
    optimize_BoxE(model, dataset)