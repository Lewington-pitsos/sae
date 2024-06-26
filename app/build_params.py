import json

from app.constants import *

def build():
    default = {
        "batch_size": 32,
        "hidden_size": 768,
        "lr": 1e-5,
        "epochs": 5,
        "model_size": "gpt2",
        "freeze": True,
        "dataset_name": "data/raft_tweet_eval_hate",
        "model_type": "probability",
        "skip_wandb": True,
        "max_seq_len": 256,
        "embedding_params": None,
        "skip_training": False,
        "dry_run": True
    }

    all_params = []
    for dataset_name in COMPILED_DATASETS:
        clone = default.copy()
        clone['dataset_name'] = dataset_name
        all_params.append(clone)
    
    with open('.params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    