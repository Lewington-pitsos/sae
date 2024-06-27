import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.models import build_model
from app.constants import *
from app.load import load_ds
from app.logging import MetricsLogger
from app.ingest import ingest_and_save_raft

def train(
        metrics: MetricsLogger, 
        model, train_dataset, 
        test_dataset, 
        lr, 
        epochs, 
        batch_size, 
        test_batch_size,
        test_every=1,  
        device=DEVICE
    ):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

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
            # some models have no trainable parameters. Check if there are any gardients to calculate 

            if any(param.requires_grad for param in model.parameters()):
                loss.backward()
                optimizer.step()

            metrics.log_train_batch(
                loss=loss.item(), 
                labels=labels, 
                outputs=outputs, 
                lr=scheduler.get_last_lr()[0]
            )

        if epoch % test_every == 0 or epoch == epochs - 1:
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

def run(params, project):
    if ('skip_training' in params and params['skip_training']) and not params['skip_wandb']:
        raise ValueError("If you want to skip training, you should also skip wandb logging")
    
    if not project and not params['skip_wandb']:
        raise ValueError("You need to provide a project name unless you are skipping wandb logging")

    _set_seed()

    name = f"{params['model_type']}_{params['dataset_name'].split('/')[-1]}"
    metrics = MetricsLogger(model_type=params['model_type'], skip_wandb=params['skip_wandb'], name=name)

    print('run in project', project)
    metrics.init(project=project, config=params.copy())

    train_dataset, test_dataset = load_ds(dataset_name=params['dataset_name'], max_seq_len=params['max_seq_len'], model_type=params['model_type'])
    metrics.log_label_ratio(train_dataset, 'train')
    metrics.log_label_ratio(test_dataset, 'test')

    model = build_model(
        params['model_type'], 
        model_size=params['model_size'],
        dataset_name=params['dataset_name'],
        freeze=params['freeze'], 
        max_seq_len=params['max_seq_len'],
        activation=params['activation'], 
        dropout=params['dropout'],
        hidden_size=params['hidden_size'],
        second_attention_mask=params['second_attention_mask'],
        device=DEVICE,
    )
    metrics.log_model_params(model)

    if 'skip_training' in params and params['skip_training']:
        print('skipping the actual training')
        metrics.finalize()
        return

    if 'dry_run' in params and params['dry_run']:
        # filter both datasets down by randomly selecting 10 samples
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset))[:10])
        test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset))[:10])


    train(
        metrics, 
        model, 
        train_dataset, 
        test_dataset, 
        batch_size=params['batch_size'],
        test_batch_size=params['test_batch_size'] if 'test_batch_size' in params else params['batch_size'], 
        epochs=params['epochs'], 
        lr=params['lr'],
        test_every=params['test_every'], 
        device=DEVICE
    )

def run_all(project='imdb-gpt2-classification', param_file='.params.json'):
    with open(param_file) as f:
        runs = json.load(f)

    datasets = set()

    for r in runs:
        datasets.add(r['dataset_name'])

    for dataset in datasets:
        if dataset in RAFT_DATASETS:
            ingest_and_save_raft(dataset)

    for r in runs:
        print("Starting new run...")
        run(r, project)