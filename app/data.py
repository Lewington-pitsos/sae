import os

from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset

from transformers import GPT2Tokenizer

from app.constants import *
from app.models import Embedder

class IMDBDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        label = torch.tensor(item['label'])
        return input_ids, attention_mask, label

def build_dataset(setting, generate_embeddings=False) -> IMDBDataset:
    if setting not in ['one-batch', 'dry-run', 'full']:
        raise ValueError(f"Invalid setting: {setting}")
    ds_name = setting
    embeddings = '-embeddings' if generate_embeddings else ''
    dataset_local = os.path.join('cruft', 'datasets', f'imdb-{ds_name}{embeddings}.pt')

    if os.path.exists(dataset_local):
        print(f"Loading dataset from {dataset_local}")
        tokenized_datasets = torch.load(dataset_local)
    else:
        print("Building dataset...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token 
        dataset = load_dataset('imdb')

        if setting == 'dry-run':
            max_samples=100
        elif setting == 'one-batch':
            max_samples=4

        if setting in ['dry-run', 'one-batch']:
            dataset['train'] = dataset['train'].shuffle().select(range(max_samples))
            dataset['test'] = dataset['test'].shuffle().select(range(max_samples))

        del dataset['unsupervised'] 

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        if generate_embeddings:
            embedder = Embedder(EMBEDDING_LAYER)
            tokenized_datasets = dataset.map(embedder.embed, batched=True)

        os.makedirs(os.path.dirname(dataset_local), exist_ok=True)
        torch.save(tokenized_datasets, dataset_local)
        print(f"Dataset saved to {dataset_local}")

    train_dataset = IMDBDataset(tokenized_datasets['train'].shuffle())
    test_dataset = IMDBDataset(tokenized_datasets['test'].shuffle())

    return train_dataset, test_dataset
