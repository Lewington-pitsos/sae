import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.models import build_model
from app.constants import *
from app.load import load_imdb
from app.logging import MetricsLogger

def train(metrics: MetricsLogger, model, train_dataset, test_dataset, lr, epochs, batch_size, device=DEVICE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    model.to(device)

    for epoch in range(epochs):
        model.train()

        for input_ids, attention_mask, labels in tqdm(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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

def _set_seed():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run(params):
    if ('skip_training' in params and params['skip_training']) and not params['skip_wandb']:
        raise ValueError("If you want to skip training, you should also skip wandb logging")
    
    _set_seed()
    metrics = MetricsLogger(model_name=params['model_name'], skip_wandb=params['skip_wandb'])

    metrics.init(project="imdb-gpt2-classification", config=params.copy())

    train_dataset, test_dataset = load_imdb(max_seq_len=params['max_seq_len'], model_name=params['model_name'])
    metrics.log_label_ratio(train_dataset, 'train')
    metrics.log_label_ratio(test_dataset, 'test')

    model = build_model(
        params['model_name'], 
        hidden_size=params['hidden_size'], 
        freeze=params['freeze'], 
        max_seq_len=params['max_seq_len'], 
        device=DEVICE,
    )
    metrics.log_model_params(model)

    if 'skip_training' in params and params['skip_training']:
        print('skipping the actual training')
        metrics.finalize()
        return

    train(metrics, model, train_dataset, test_dataset, batch_size=params['batch_size'], epochs=params['epochs'], lr=params['lr'], device=DEVICE)

def run_all():
    with open('.params.json') as f:
        runs = json.load(f)

    for r in runs:
        print("Starting new run...")
        run(r)