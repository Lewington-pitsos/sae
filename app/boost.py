import torch
import xgboost as xgb
import numpy as np
import wandb
from sklearn.metrics import f1_score

from app.constants import *
from app.embed import embed_datasets


class LogEvaluation(xgb.callback.TrainingCallback):
    def __init__(self, period=1):
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            result = {}
            for data_name, data in evals_log.items():
                for metric_name, score in data.items():
                    result[f'{data_name}_{metric_name}'] = score[-1]
            wandb.log(result)
        return False

def load_sae_feature_dataset(train_filename=f'{LOCAL_DATA_PATH}/avg-emb-gpt2-256-train.pt', test_filename=f'{LOCAL_DATA_PATH}/avg-emb-gpt2-256-test.pt'):    
    train = torch.load(train_filename)
    test = torch.load(test_filename)

    X = train[:, :-1].numpy()
    y = train[:, -1].numpy()

    dtrain = xgb.DMatrix(X, label=y)

    X_test = test[:, :-1].numpy()
    y_test = test[:, -1].numpy()

    X_val = X_test[:500]
    y_val = y_test[:500]

    dval = xgb.DMatrix(X_val, label=y_val)
    dholdout = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dval, dholdout

def f1(preds, ds):
    labels = ds.get_label()

    f1 = f1_score(labels, preds, average='macro')
    return f1

def f1_metric(preds, ds):
    return 'f1', f1(preds, ds)

def accuracy(preds, ds):
    labels = ds.get_label()
    preds = preds > 0.5

    accuracy = np.sum(preds == labels) / len(labels)
    return accuracy

def accuracy_metric(preds, ds):
    return 'accuracy', accuracy(preds, ds)

def combined_metrics(preds, ds):
    return [f1_metric(preds, ds), accuracy_metric(preds, ds)]

def train(train_filename, test_filename):
    config_defaults = {
        'boosting': 'gbtree',  # Use tree based models
        'max_depth': 5,  # Increase depth
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],  # Include both logloss and error
        'subsample': 0.85,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'alpha': 0.1,  # L1 regularization term
        'device': 'cuda',
        'learning_rate': 0.05,
    }
    
    dtrain, dval, dholdout = load_sae_feature_dataset(train_filename, test_filename)
    watchlist = [(dtrain, 'train'), (dholdout, 'test')]

    wandb.init(config=config_defaults, project="llm-cls-2")  # defaults are over-ridden during the sweep

    bst = xgb.train(config_defaults, dtrain, 2000, watchlist, custom_metric=accuracy_metric, early_stopping_rounds=50, maximize=True, callbacks=[LogEvaluation(1)])


    wandb.finish()

if __name__ == '__main__': 
        embed_datasets(
            dataset_names=['imdb'],
            model_name='sae-classifier-gpt',
            max_seq_len=256
        )

        for ft in ['sae_ft', 'hs', 'hs_final']:
            train(
                f'{LOCAL_DATA_PATH}/{ft}-imdb-train.pt',
                f'{LOCAL_DATA_PATH}/{ft}-imdb-test.pt',
            )
