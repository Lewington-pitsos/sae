import torch
import json
import os

SAE_EMBEDDING_LAYER_NUMBER=8
MAX_GPT2_BATCH_SIZE=256
LOCAL_DATA_PATH = "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    'gpt2-256': {
        "path": f"{LOCAL_DATA_PATH}/imdb-full-256-sae-classifier-gpt2.pt",
        "description": "IMDb dataset with GPT2 embeddings of sequences of length 256 tokens",
    },
    'mistral-256': {
        "path": f"{LOCAL_DATA_PATH}/imdb-full-256-sae-classifier-mistral7b.pt",
        "description": "IMDb dataset with Mistral embeddings of sequences of length 256 tokens",
    }
}


NON_TEXT_COLUMNS = ['ID', 'Label']

CREDENTIALS_FILE = '.credentials.json'
if os.path.exists(CREDENTIALS_FILE):
    with open(CREDENTIALS_FILE) as f:
        CRED = json.load(f)
else:
    env = os.environ
    CRED = { **env }


os.environ['HF_TOKEN'] = CRED['HF_TOKEN']


RAFT_DATASETS = [
    'tweet_eval_hate',
    'ade_corpus_v2',
    'overruling',
    'banking_77'
]

COMPILED_DATASETS = ['imdb'] + [f'{LOCAL_DATA_PATH}/raft_' + f for f in RAFT_DATASETS]