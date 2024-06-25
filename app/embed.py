import tqdm
import torch
from torch.utils.data import DataLoader

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

def embed_dataset(raft_dataset_name, model_name, tokenizer, embedder, max_seq_len=256):
    dataset = ingest_raft(raft_dataset_name)
    text_column = get_text_column(dataset)

    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding='max_length', truncation=True, max_length=max_seq_len)
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

                for j, emb in enumerate(embedding):
                    dataset[ds_name][i * 16 + j][model_name] = emb.numpy().tolist()

                embeddings_and_labels = torch.cat([embedding, batch['Label'].unsqueeze(-1)], dim=1)

                all_avg_fts.append(embeddings_and_labels)
            
            torch.save(torch.cat(all_avg_fts).squeeze(), f'{LOCAL_DATA_PATH}/avg-emb-{max_seq_len}-{model_name}-{ds_name}-{raft_dataset_name}.pt')

    dataset.save_to_disk(f'{LOCAL_DATA_PATH}/{raft_dataset_name}-with-{model_name}-embeddings')


if __name__ == '__main__':
    embed_datasets(
        dataset_names=RAFT_DATASETS,
        model_name='sae-classifier-mistral7b',
        max_seq_len=256
    )

