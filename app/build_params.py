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
        "lr": 1e-5,
        "epochs": 5,
        "model_size": "gpt2",
        "freeze": True,
        "dataset_name": "data/raft_tweet_eval_hate",
        "model_type": "probability",
        "skip_wandb": False,
        "max_seq_len": 256,
        "skip_training": False,
        "dry_run": False
    }

    all_params = []
    for dataset_name in COMPILED_DATASETS:
        for model_type in ['probability', 'simple', 'sae-classifier-gpt', 'random', 'gpt2-classifier']:
            clone = default.copy()

            if model_type in ['random', 'probability']:
                clone['epochs'] = 1

            clone['model_type'] = model_type
            clone['dataset_name'] = dataset_name
            all_params.append(clone)

            if model_type in ['simple', 'gpt2-classifier', 'probability']:
                for model_size in ['gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    clone = clone.copy()
                    clone['model_size'] = model_size
                    clone['batch_size'] = get_batch_size(model_size)
                    all_params.append(clone)
            elif model_type in ['sae-classifier-gpt']:
                clone = clone.copy()
                clone['freeze'] = False
                clone['batch-size'] = 32
                all_params.append(clone) 

    print(f'built {len(all_params)} params')
    with open('.params.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    