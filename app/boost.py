import torch
import xgboost as xgb
import numpy as np
import wandb

from app.constants import *


class LogEvaluation(xgb.callback.TrainingCallback):
    def __init__(self, period=1):
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            result = {}
            for data_name, data in evals_log.items():
                for metric_name, score in data.items():
                    result[f'{data_name}-{metric_name}'] = score[-1]
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

    X_val = X_test[:20000]
    y_val = y_test[:20000]

    dval = xgb.DMatrix(X_val, label=y_val)
    X_test = X_test[20000:]
    y_test = y_test[20000:]

    dholdout = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dval, dholdout

def accuracy(preds, ds):
    labels = ds.get_label()
    preds = preds > 0.5

    accuracy = np.sum(preds == labels) / len(labels)
    return accuracy

def accuracy_metric(preds, ds):
    return 'accuracy', accuracy(preds, ds)

def train():
    config_defaults = {
        'boosting': 'gbtree',  # Use tree based models
        'max_depth': 7,  # Increase depth
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],  # Include both logloss and error
        'subsample': 0.8,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'alpha': 0.1,  # L1 regularization term
        'device': 'cuda',
        'learning_rate': 0.01,
    }
    
    dtrain, dval, dholdout = load_sae_feature_dataset(
        f'{LOCAL_DATA_PATH}/avg-emb-gpt2-256-train-pca-64.pt',
        f'{LOCAL_DATA_PATH}/avg-emb-gpt2-256-test-pca-64.pt'
    )

    # Create watchlist
    watchlist = [(dtrain, 'train'), (dval, 'val'), (dholdout, 'holdout')]

    wandb.init(config=config_defaults, project="xgb-test")  # defaults are over-ridden during the sweep

    # Train the model with evaluation on the training and test sets, and early stopping
    bst = xgb.train(config_defaults, dtrain, 10000, watchlist, custom_metric=accuracy_metric, maximize=True, callbacks=[LogEvaluation(1)])


    wandb.finish()
if __name__ == '__main__':  
    train()