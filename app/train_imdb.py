import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

from app.models import SimpleGPT2SequenceClassifier

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

def build_dataset(dry_run=False) -> IMDBDataset:
    def print_label_ratio(dataset, name):
        true_count = sum(1 for item in dataset.tokenized_dataset if item['label'] == 1)
        false_count = len(dataset) - true_count
        print(f"{name} True labels: {true_count}, False labels: {false_count}")
        print(f"{name} Ratio (True/False): {true_count/false_count:.2f}")

    ds_name = 'try' if dry_run else 'wet'
    dataset_local = os.path.join('cruft', 'datasets', f'imdb-{ds_name}.pt')

    if os.path.exists(dataset_local):
        print(f"Loading dataset from {dataset_local}")
        tokenized_datasets = torch.load(dataset_local)
    else:
        print("Building dataset...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token 
        dataset = load_dataset('imdb')

        if dry_run:
            max_samples=100
            dataset['train'] = dataset['train'].shuffle().select(range(max_samples))
            dataset['test'] = dataset['test'].shuffle().select(range(max_samples))

        del dataset['unsupervised'] 

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        os.makedirs(os.path.dirname(dataset_local), exist_ok=True)
        torch.save(tokenized_datasets, dataset_local)
        print(f"Dataset saved to {dataset_local}")

    train_dataset = IMDBDataset(tokenized_datasets['train'])
    test_dataset = IMDBDataset(tokenized_datasets['test'])

    print_label_ratio(train_dataset, 'train')
    print_label_ratio(test_dataset, 'test')

    return train_dataset, test_dataset

def train(model, train_dataset, test_dataset, batch_size=16):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}")

        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss}")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

MAX_SEQ_LEN = 1024

train_dataset, test_dataset = build_dataset(dry_run=True)

hidden_size = 768
model = SimpleGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=MAX_SEQ_LEN, gpt_model_name='gpt2')

train(model, train_dataset, test_dataset)