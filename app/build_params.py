import json

from app.constants import *

COMPILED_DATASETS = ['imdb'] + [f'{LOCAL_DATA_PATH}/raft_' + f for f in RAFT_DATASETS if f != 'banking_77']


def get_batch_size(model_size):
    if model_size == 'gpt2-medium':
        return 128
    elif model_size == 'gpt2-large':
        return 64
    elif model_size == 'gpt2-xl':
        return 36

    raise ValueError(f'Unknown model size {model_size}')
def build():
    default = {
        "batch_size": 256,
        "lr": 1e-4,
        "epochs": 8,
        "model_size": "gpt2",
        "freeze": True,
        "dataset_name": "data/raft_tweet_eval_hate",
        "model_type": "probability",
        "skip_wandb": False,
        "max_seq_len": 256,
        "activation": "relu",
        "skip_training": False,
        "dry_run": False
    }

    all_params = []
    for dataset_name in COMPILED_DATASETS:
        for model_type in ['simple', 'sae-classifier-gpt']:
            clone = default.copy()
            clone['model_type'] = model_type
            clone['dataset_name'] = dataset_name
            if dataset_name in ['data/raft_tweet_eval_hate', 'data/ade_corpus_v2', 'data/raft_overruling']:
                clone['epochs'] = 15
            all_params.append(clone)

    print(f'built {len(all_params)} params')
    with open('.params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    