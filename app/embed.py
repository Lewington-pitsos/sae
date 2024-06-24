import tqdm
import torch
from torch.utils.data import DataLoader

from app.models import SAEFeaturesModel, get_sae_model_config, masked_avg
from app.constants import *
from app.tok import load_tokenizer
from app.load import smart_load_dataset

def create_embeddings(dataset_name, max_seq_len=256, model_name='sae-classifier-mistral7b'):
    embedder = SAEFeaturesModel(
        device=DEVICE,
        max_seq_len=max_seq_len,
        **get_sae_model_config(model_name)
    )

    dataset = smart_load_dataset(dataset_name)

    tokenizer = load_tokenizer(model_name)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)
    dataset = dataset.map(tokenize_function, batched=True)

    def embed(examples):
        input_ids = torch.stack(examples['input_ids']).to(DEVICE).transpose(1, 0)
        attention_mask = torch.stack(examples['attention_mask']).to(DEVICE).transpose(1, 0)

        return masked_avg(embedder(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

    with torch.no_grad():
        for ds_name in ['train', 'test']:
            loader = DataLoader(dataset[ds_name], batch_size=16, shuffle=True)
            all_avg_fts = []
            for i, batch in tqdm.tqdm(enumerate(loader)):
                embedding = embed(batch).to('cpu')

                embeddings_and_labels = torch.cat([embedding, batch['label'].unsqueeze(-1)], dim=1)

                all_avg_fts.append(embeddings_and_labels)
            
            torch.save(torch.cat(all_avg_fts).squeeze(), f'{LOCAL_DATA_PATH}/avg-emb-gpt2-mistral-{ds_name}-{dataset_name.split("/")[-1]}.pt')


create_embeddings(f'./{LOCAL_DATA_PATH}/raft_ade_corpus_v2', model_name='sae-classifier-gpt2')
create_embeddings(f'./{LOCAL_DATA_PATH}/raft_tweet_eval_hate', model_name='sae-classifier-gpt2')
