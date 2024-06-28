import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from app.models import SAEFeaturesModel, get_sae_model_config, masked_avg
from app.constants import *
from app.tok import load_tokenizer
from app.load import get_text_column
from app.ingest import ingest_raft

def embed_datasets(dataset_names, model_name, max_seq_len=256):
    embedder = SAEFeaturesModel(
        device=DEVICE,
        max_seq_len=max_seq_len,
        **get_sae_model_config(model_name)
    )
    tokenizer = load_tokenizer(model_name)

    for raft_dataset_name in dataset_names:
        embed_dataset(raft_dataset_name, model_name, tokenizer, embedder, max_seq_len)

def get_dataset_name(raft_dataset_name, embedding_name, split):
    return f'{LOCAL_DATA_PATH}/{embedding_name}-{raft_dataset_name}-{split}.pt'

def embed_dataset(raft_dataset_name, model_name, tokenizer, embedder, max_seq_len=256):
    # dataset = ingest_raft(raft_dataset_name)
    # text_column = get_text_column(dataset)
    text_column = 'text'
    dataset = load_dataset(raft_dataset_name)
    del dataset['unsupervised']


    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding='max_length', truncation=True, max_length=max_seq_len)
    dataset = dataset.map(tokenize_function, batched=True)

    def embed(examples):
        input_ids = torch.stack(examples['input_ids']).to(DEVICE).transpose(1, 0)
        attention_mask = torch.stack(examples['attention_mask']).to(DEVICE).transpose(1, 0)

        sae_features, hidden_states, final_hidden_states = embedder(input_ids=input_ids, attention_mask=attention_mask)

        sae_features = masked_avg(sae_features, attention_mask)
        hidden_states = masked_avg(hidden_states, attention_mask)
        final_hidden_states = masked_avg(final_hidden_states, attention_mask)

        return {
            'sae_ft': sae_features,
            'hs': hidden_states,
            'hs_final': final_hidden_states
        }

    with torch.no_grad():
        for split in ['train', 'test']:
            batch_size=32
            loader = DataLoader(dataset[split], batch_size=batch_size, shuffle=True)
            all_avg_fts = {}
            for i, batch in tqdm.tqdm(enumerate(loader)):
                embedding_dict = embed(batch)
                for embedding_name, embedding in embedding_dict.items():
                    embedding = embedding.to('cpu')
                 
                    all_avg_fts.setdefault(embedding_name, [])
                    for j, emb in enumerate(embedding):
                        dataset[split][i * 16 + j][model_name] = emb.numpy().tolist()

                    embeddings_and_labels = torch.cat([embedding, batch['label'].unsqueeze(-1)], dim=1)

                    all_avg_fts[embedding_name].append(embeddings_and_labels)
            
            for embedding_name, embeddings in all_avg_fts.items():
                name = get_dataset_name(raft_dataset_name, embedding_name, split)
                print('Saving', name)
                torch.save(torch.cat(embeddings).squeeze(), name)

    dataset.save_to_disk(f'{LOCAL_DATA_PATH}/{raft_dataset_name}-with-{model_name}-embeddings')


if __name__ == '__main__':
    embed_datasets(
        dataset_names=['imdb'],
        model_name='sae-classifier-gpt',
        max_seq_len=256
    )

