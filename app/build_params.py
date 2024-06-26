import json

from app.constants import *

COMPILED_DATASETS = ['imdb'] + [f'{LOCAL_DATA_PATH}/raft_' + f for f in RAFT_DATASETS if f != 'banking_77']


def build():
    default = {
        "batch_size": 32,
        "hidden_size": 768,
        "lr": 1e-5,
        "epochs": 10,
        "model_size": "gpt2",
        "freeze": True,
        "dataset_name": "data/raft_tweet_eval_hate",
        "model_type": "probability",
        "skip_wandb": False,
        "max_seq_len": 256,
        "embedding_params": None,
        "skip_training": False,
        "dry_run": True
    }

    all_params = []
    for dataset_name in COMPILED_DATASETS:
        for model_type in ['probability', 'simple', 'sae-classifier-gpt', 'random', 'gpt2-classifier']:
            clone = default.copy()
            clone['model_type'] = model_type
            clone['dataset_name'] = dataset_name
            all_params.append(clone)
    

    print(f'built {len(all_params)} params')
    with open('.params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    