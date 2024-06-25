import os

from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset, load_from_disk

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

def load_imdb(max_seq_len, model_name) -> IMDBDataset:
    local_file = os.path.join(LOCAL_DATA_PATH, f'imdb-{max_seq_len}-{model_name}.pt')

    if os.path.exists(local_file):
        print(f"Loading dataset from {local_file}")
        dataset = torch.load(local_file)
    else:
        print("Building dataset...")
        tokenizer = load_tokenizer(model_name)
        dataset = load_dataset('imdb')

        if 'unsupervised' in dataset:
            del dataset['unsupervised'] 

        text_col = get_text_column(dataset)

        def tokenize_function(examples):
            return tokenizer(examples[text_col], padding='max_length', truncation=True, max_length=max_seq_len)

        dataset = dataset.map(tokenize_function, batched=True)

        dataset_dict = {'train': list(dataset['train']), 'test': list(dataset['test'])}     
        
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        torch.save(dataset_dict, local_file)
        print(f"Dataset saved to {local_file}")

    train_dataset = IMDBDataset(dataset_dict['train'])
    test_dataset = IMDBDataset(dataset_dict['test'])

    return train_dataset, test_dataset

def smart_load_dataset(name):
    if os.path.exists(name) and LOCAL_DATA_PATH in name:
        return load_from_disk(name)
    return load_dataset(name)

def get_text_column(dataset):
    return list(set(dataset['train'].column_names) - set(NON_TEXT_COLUMNS))[0]
