import json
import sys

import numpy as np
from pykeen.pipeline import replicate_pipeline_from_config
from Utils import parse_kwargs

def evaluate_SGD_model(model_name, dataset):
    pipeline_path = f"./Benchmarking/{model_name}/{dataset}/best_pipeline/pipeline_config.json"
    rep_dir = f"./Benchmarking/{model_name}/{dataset}/best_pipeline/rep_results"
    metrics_path = f"./Benchmarking/{model_name}/{dataset}/best_pipeline/rep_metrics.json"

    with open(pipeline_path) as f:
        config = json.load(f)

    config["pipeline"]["random_seed"] = 6934

    replicate_pipeline_from_config(
        config=config,
        directory=rep_dir,
        replicates=3,
        keep_seed=True,
    )

    # Read metrics from saved results
    metrics_per_rep = []
    for i in range(3):
        with open(f"{rep_dir}/replicates/replicate-{i:05d}/results.json") as f:
            results = json.load(f)
        mr = results["metrics"]
        metrics_per_rep.append({
            "MRR": mr["both"]["realistic"]["inverse_harmonic_mean_rank"],
            "Hits@1": mr["both"]["realistic"]["hits_at_1"],
            "Hits@3": mr["both"]["realistic"]["hits_at_3"],
            "Hits@10": mr["both"]["realistic"]["hits_at_10"],
        })

    summary = {
        "replicates": metrics_per_rep,
        "mean": {key: float(np.mean([m[key] for m in metrics_per_rep])) for key in metrics_per_rep[0]},
        "std": {key: float(np.std([m[key] for m in metrics_per_rep])) for key in metrics_per_rep[0]},
    }

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    model, dataset = parse_kwargs(**dict(arg.split('=') for arg in sys.argv[1:]))
    evaluate_SGD_model(model, dataset)