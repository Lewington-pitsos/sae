import torch
import json

SAE_EMBEDDING_LAYER_NUMBER=8
MAX_GPT2_BATCH_SIZE=256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open('.credentials.json') as f:
    CREDS = json.load(f)



            #     release = , # see other options in sae_lens/pretrained_saes.yaml
            # sae_id = , # won't always be a hook point