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
import wandb

from app.models import SimpleGPT2SequenceClassifier
from app.viz import model_parameters_info

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

def build_dataset(setting) -> IMDBDataset:
    if setting not in ['one-batch', 'dry-run', 'full']:
        raise ValueError(f"Invalid setting: {setting}")

    def print_label_ratio(dataset, name):
        true_count = sum(1 for item in dataset.tokenized_dataset if item['label'] == 1)
        false_count = len(dataset) - true_count
        print(f"{name} True labels: {true_count}, False labels: {false_count}")
        print(f"{name} Ratio (True/False): {true_count/false_count:.2f}")

        wandb.log({f"{name}_data_true_count": true_count, f"{name}_data_false_count": false_count, f"{name}_data_ratio": true_count/false_count})

    ds_name = setting
    dataset_local = os.path.join('cruft', 'datasets', f'imdb-{ds_name}.pt')

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

        os.makedirs(os.path.dirname(dataset_local), exist_ok=True)
        torch.save(tokenized_datasets, dataset_local)
        print(f"Dataset saved to {dataset_local}")

    train_dataset = IMDBDataset(tokenized_datasets['train'])
    test_dataset = IMDBDataset(tokenized_datasets['test'])

    print_label_ratio(train_dataset, 'train')
    print_label_ratio(test_dataset, 'test')

    return train_dataset, test_dataset

def train(model, train_dataset, test_dataset, learning_rate, epochs, batch_size):
    
    batch1_table = wandb.Table(columns=["epoch", "idx", "input_text", "label", "prediction", "logits"])


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")
        wandb.log({"train_loss": avg_train_loss})

        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                input_ids, attention_mask, labels = batch

                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                if batch_idx == 0:
                    # Convert input_ids to strings
                    input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:4]]

                    # Get predictions and labels
                    predictions = torch.argmax(outputs, dim=-1)


                    # Log details to Weights and Biases
                    for i in range(min(len(input_texts), 4)):
                        batch1_table.add_data(epoch, i, input_texts[i], labels[i], predictions[i], outputs[i])

        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {avg_test_loss}")
        wandb.log({"test_loss": avg_test_loss})

    wandb.log({
        f"batch_1": batch1_table,
    })
    wandb.finish()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

MAX_SEQ_LEN = 1024
BATCH_SIZE = 16
hidden_size = 768
learning_rate = 1e-5
epochs=100
freeze = True
setting='one-batch'
wandb.init(project="imdb-gpt2-classification", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": learning_rate,
    "num_epochs": epochs,
    "max_seq_len": MAX_SEQ_LEN,
    "hidden_size": hidden_size,
    "freeze": freeze,
    "setting": setting
})


train_dataset, test_dataset = build_dataset(setting=setting)

model = SimpleGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=MAX_SEQ_LEN, gpt_model_name='gpt2', freeze=True)
model_parameters_info(model)

train(model, train_dataset, test_dataset, batch_size=BATCH_SIZE, epochs=epochs, learning_rate=learning_rate)
