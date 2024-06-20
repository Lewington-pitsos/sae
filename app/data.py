import os

from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset

from transformers import GPT2Tokenizer

from app.constants import *
from app.models import ActivationModel

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


def _build_dataset(data_mode):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token 
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
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)

    dataset = dataset.map(tokenize_function, batched=True)

    return dataset

def load_imdb(data_mode) -> IMDBDataset:
    if data_mode not in ['one-batch', 'dry-run', 'full']:
        raise ValueError(f"Invalid setting: {data_mode}")
    ds_name = data_mode
    
    local_file = os.path.join('cruft', 'datasets', f'imdb-{ds_name}.pt')

    if os.path.exists(local_file):
        print(f"Loading dataset from {local_file}")
        dataset = torch.load(local_file)
    else:
        print("Building dataset...")
        dataset = _build_dataset(data_mode)

        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        torch.save(dataset, local_file)
        print(f"Dataset saved to {local_file}")

    train_dataset = IMDBDataset(dataset['train'].shuffle())
    test_dataset = IMDBDataset(dataset['test'].shuffle())

    return train_dataset, test_dataset
