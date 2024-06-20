import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
import wandb

from app.models import build_model
from app.viz import model_parameters_info
from app.constants import *
from app.data import load_imdb

class MetricsLogger():
    def __init__(self):
        self.batch1_table = wandb.Table(columns=["epoch", "idx", "input_text", "label", "prediction", "logits"])
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.epoch = 0

    def step_epoch(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_train_accuracy = sum(self.train_acc) / len(self.train_acc)
        self.train_losses = []
        self.train_acc = []
        print(f"Epoch {self.epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")
        wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})

        avg_test_loss = sum(self.test_losses) / len(self.test_losses)
        avg_test_accuracy = sum(self.test_acc) / len(self.test_acc)
        print(f"Epoch {self.epoch + 1}/{epochs}, Test Loss: {avg_test_loss}")
        wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy})


        self.epoch += 1

    def log_train_batch(self, loss, labels, outputs, lr):
        self.train_losses.append(loss)
        self.train_acc.append((torch.argmax(outputs, dim=-1) == labels).sum().item())
        wandb.log({"train_batch_loss": loss, "batch_learning_rate": lr})

    def log_test_batch(self, batch_idx, loss, labels, outputs, input_ids):
        self.test_losses.append(loss)
        self.test_acc.append((torch.argmax(outputs, dim=-1) == labels).sum().item())

        if batch_idx == 0:
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:4]]

            predictions = torch.argmax(outputs, dim=-1)

            for i in range(min(len(input_texts), 4)):
                self.batch1_table.add_data(self.epoch, i, input_texts[i], labels[i], predictions[i], outputs[i])

    def finalize(self):
        wandb.log({f"batch_1": self.batch1_table})
        wandb.finish()

def train(model, train_dataset, test_dataset, learning_rate, epochs, batch_size, device=DEVICE):
    metrics = MetricsLogger()

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model.to(device)

    for epoch in range(epochs):
        model.train()

        for input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            metrics.log_train_batch(
                loss=loss.item(), 
                labels=labels, 
                outputs=outputs, 
                lr=scheduler.get_last_lr()[0]
            )

        model.eval()
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask, labels) in tqdm(enumerate(test_loader)):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                metrics.log_test_batch(
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    labels=labels,
                    outputs=outputs,
                    input_ids=input_ids,
                )

        scheduler.step()
        metrics.step_epoch()

    metrics.finalize()


def log_label_ratio(dataset, name):
    labels = []
    for i in range(min(300, len(dataset))):
        labels.append(dataset.get_label(i))
    
    true_count = sum(1 for item in labels if item == 1)
    false_count = len(labels) - true_count
    print(f"{name} True labels: {true_count}, False labels: {false_count}")
    print(f"{name} Ratio (True/False): {true_count/false_count:.2f}")

    wandb.log({f"{name}_data_true_count": true_count, f"{name}_data_false_count": false_count, f"{name}_data_TF_ratio": true_count/false_count})


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

BATCH_SIZE = 8
hidden_size = 768
learning_rate = 1e-5
epochs=5
freeze = True
data_mode='one-batch'
model_name = 'big-head'
wandb.init(project="imdb-gpt2-classification", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": learning_rate,
    "num_epochs": epochs,
    "max_seq_len": MAX_SEQ_LEN,
    "hidden_size": hidden_size,
    "freeze": freeze,
    "data_mode": data_mode,
    "model_name": model_name
})


train_dataset, test_dataset = load_imdb(data_mode=data_mode)

log_label_ratio(train_dataset, 'train')
log_label_ratio(test_dataset, 'test')

model = build_model(model_name, gpt_model_name='gpt2', hidden_size=hidden_size, freeze=freeze, max_seq_len=MAX_SEQ_LEN, device=DEVICE)

model_parameters_info(model)

train(model, train_dataset, test_dataset, batch_size=BATCH_SIZE, epochs=epochs, learning_rate=learning_rate, device=DEVICE)
