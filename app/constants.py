import torch
import json

SAE_EMBEDDING_LAYER_NUMBER=8
MAX_GPT2_BATCH_SIZE=256
LOCAL_DATA_PATH = "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    'gpt2-256': {
        "path": "cruft/datasets/imdb-full-256-sae-classifier-gpt2.pt",
        "description": "IMDb dataset with GPT2 embeddings of sequences of length 256 tokens",
    }
}


with open('.credentials.json') as f:
    CREDS = json.load(f)


