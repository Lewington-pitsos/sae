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


def build_dataset(max_samples=None) -> IMDBDataset:
    dataset_local = os.path.join('cruft', 'datasets', f'imdb-{max_samples}.pt')

    if os.path.exists(dataset_local):
        print(f"Loading dataset from {dataset_local}")
        tokenized_datasets = torch.load(dataset_local)
    else:
        print("Building dataset...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token 
        dataset = load_dataset('imdb')

        if max_samples:
            dataset['train'] = dataset['train'].shuffle().select(range(max_samples))
            dataset['test'] = dataset['test'].shuffle().select(range(max_samples))

        # delete the 'unsupervised' section from the dataset
        del dataset['unsupervised'] 

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Save the tokenized dataset to disk
        os.makedirs(os.path.dirname(dataset_local), exist_ok=True)
        torch.save(tokenized_datasets, dataset_local)
        print(f"Dataset saved to {dataset_local}")

    train_dataset = IMDBDataset(tokenized_datasets['train'])

    return train_dataset

def train(model, tokenized_dataset):
    train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

    hidden_size = 768  # GPT-2 hidden size
    num_classes = 2
    model = SimpleGPT2SequenceClassifier(hidden_size=hidden_size, num_classes=num_classes, max_seq_len=MAX_SEQ_LEN, gpt_model_name='gpt2')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

    # Evaluation loop
    model.eval()
    total_correct = 0
    total_samples = 0


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

MAX_SEQ_LEN = 1024

tokenized_dataset = build_dataset(100)
train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

for i in range(2):
    for input_ids, attention_mask, labels in tqdm(train_loader):
        print(input_ids)
        print(attention_mask)
        print(labels)