import json

from app.constants import *

COMPILED_DATASETS = ['imdb'] + [f'{LOCAL_DATA_PATH}/raft_' + f for f in RAFT_DATASETS if f != 'banking_77']

def build(output_file='.params.json'):
    default = {
        "batch_size": 256,
        "test_batch_size": 256,
        "lr": 1e-4,
        "epochs": 8,
        "model_size": "gpt2",
        "freeze": True,
        "dataset_name": "imdb",
        "model_type": 'big-head',
        "skip_wandb": False,
        "max_seq_len": 256,
        "activation": "relu",
        "hidden_size": 1024,
        "dropout": 0.3,
        "test_every": 1,
        "skip_training": False,
        'second_attention_mask': True,
        "dry_run": False
    }

    all_params = []
    for model_name in ['simple', 'sae-classifier-gpt']:
        clone = default.copy()
        clone['model_type'] = model_name
        all_params.append(clone)


    print(f'built {len(all_params)} params')
    with open(output_file, 'w') as f:
        json.dump(all_params, f, indent=2)
    