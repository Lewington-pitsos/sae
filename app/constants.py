import torch
import json

SAE_EMBEDDING_LAYER_NAME = 'blocks.8.hook_resid_pre'
SAE_EMBEDDING_LAYER_NUMBER=8
MAX_GPT2_BATCH_SIZE=256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open('.credentials.json') as f:
    CREDS = json.load(f)