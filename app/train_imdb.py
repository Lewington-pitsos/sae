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
from app.data import build_dataset

def train(model, train_dataset, test_dataset, learning_rate, epochs, batch_size):
    batch1_table = wandb.Table(columns=["epoch", "idx", "input_text", "label", "prediction", "logits"])

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        for activations, input_ids, attention_mask, labels in tqdm(train_loader):
            activations, input_ids, attention_mask, labels = activations.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(activations, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_accuracy += (torch.argmax(outputs, dim=-1) == labels).sum().item()
            wandb.log({"train_batch_loss": loss.item(), "batch_learning_rate": scheduler.get_last_lr()[0]})
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")
        wandb.log({"train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy})

        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for batch_idx, (activations, input_ids, attention_mask, labels) in enumerate(test_loader):

                activations, input_ids, attention_mask, labels = activations.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(activations, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                test_accuracy += (torch.argmax(outputs, dim=-1) == labels).sum().item()


                if batch_idx == 0:
                    # Convert input_ids to strings
                    input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:4]]

                    # Get predictions and labels
                    predictions = torch.argmax(outputs, dim=-1)


                    # Log details to Weights and Biases
                    for i in range(min(len(input_texts), 4)):
                        batch1_table.add_data(epoch, i, input_texts[i], labels[i], predictions[i], outputs[i])

        
        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {avg_test_loss}")
        wandb.log({"test_loss": avg_test_loss, "test_accuracy": avg_test_accuracy})

        scheduler.step()

    wandb.log({
        f"batch_1": batch1_table,
    })
    wandb.finish()

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

BATCH_SIZE = 254
hidden_size = 768
learning_rate = 1e-5
epochs=5
freeze = True
setting='full'
model_name = 'big-head'
wandb.init(project="imdb-gpt2-classification", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": learning_rate,
    "num_epochs": epochs,
    "max_seq_len": MAX_SEQ_LEN,
    "hidden_size": hidden_size,
    "freeze": freeze,
    "setting": setting,
    "model_name": model_name
})


train_dataset, test_dataset = build_dataset(setting=setting, generate_embeddings=False)

log_label_ratio(train_dataset, 'train')
log_label_ratio(test_dataset, 'test')

model = build_model(model_name, gpt_model_name='gpt2', hidden_size=hidden_size, freeze=freeze, max_seq_len=MAX_SEQ_LEN, device=DEVICE)

model_parameters_info(model)

train(model, train_dataset, test_dataset, batch_size=BATCH_SIZE, epochs=epochs, learning_rate=learning_rate)
