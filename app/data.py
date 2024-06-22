import os

from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset

from app.constants import *
from app.tok import load_tokenizer
from app.models import SAEFeaturesModel, get_sae_model_config, masked_avg

class IMDBDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def get_label(self, idx):
        return self.tokenized_dataset[idx]['label']

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        label = torch.tensor(item['label'])

        return input_ids, attention_mask, label


def _build_dataset(data_mode, max_seq_len, model_name, embedding_params):
    tokenizer = load_tokenizer(model_name)
    dataset = load_dataset('imdb')

    if data_mode in ['dry-run', 'one-batch']:
        if data_mode == 'dry-run':
            max_samples=100
        elif data_mode == 'one-batch':
            max_samples=4
        else:
            raise ValueError

        dataset['train'] = dataset['train'].shuffle().select(range(max_samples))
        dataset['test'] = dataset['test'].shuffle().select(range(max_samples))

    del dataset['unsupervised'] 

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_seq_len)

    dataset = dataset.map(tokenize_function, batched=True)

    if embedding_params is not None:
        embedder = SAEFeaturesModel(
            device=DEVICE,
            max_seq_len=max_seq_len,
            **get_sae_model_config(model_name)
        )

        def embed(examples):
            input_ids = torch.tensor(examples['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(examples['attention_mask']).to(DEVICE)
            return {"avg_features": masked_avg(embedder(input_ids=input_ids, attention_mask=attention_mask), attention_mask)}

        dataset = dataset.map(embed, batched=True, batch_size=128)

    return dataset

def load_imdb(data_mode, max_seq_len, model_name, embedding_params=None) -> IMDBDataset:
    if data_mode not in ['one-batch', 'dry-run', 'full']:
        raise ValueError(f"Invalid setting: {data_mode}")
    ds_name = data_mode
    
    embedding_name_part = "-emb" if embedding_params is not None else ""
    local_file = os.path.join('cruft', 'datasets', f'imdb-{ds_name}-{max_seq_len}-{model_name}{embedding_name_part}.pt')

    if os.path.exists(local_file):
        print(f"Loading dataset from {local_file}")
        dataset = torch.load(local_file)
    else:
        print("Building dataset...")
        dataset = _build_dataset(data_mode, max_seq_len, model_name, embedding_params)

        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        torch.save(dataset, local_file)
        print(f"Dataset saved to {local_file}")

    train_dataset = IMDBDataset(dataset['train'].shuffle())
    test_dataset = IMDBDataset(dataset['test'].shuffle())

    return train_dataset, test_dataset
