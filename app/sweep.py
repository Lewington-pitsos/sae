import xgboost as xgb
import wandb

from app.constants import *
from app.boost import *
    
def train():
    config_defaults = {
        'booster': 'gbtree',  # Use tree based models
        'max_depth': 6,  # Increase depth
        'eta': 0.001,  # Decrease learning rate
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],  # Include both logloss and error
        'subsample': 0.8,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'alpha': 0.1,  # L1 regularization term
        'lambda': 1.0,  # L2 regularization term
        'device': 'cuda',
        "learning_rate": 0.1,
    }

    wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep
    config = wandb.config

    dtrain, dval, dholdout = load_sae_feature_dataset()

    watchlist = [(dtrain, 'train'), (dval, 'val'), (dholdout, 'holdout')]

    run_config = {}
    for k, v in config.items():
        run_config[k] = config[k]

    print(run_config)

    model = xgb.train(run_config, dtrain, 4000, watchlist, custom_metric=accuracy, early_stopping_rounds=75, maximize=True)

    val_pred = model.predict(dval)
    val_acc = accuracy(val_pred, dval)
    wandb.log({"val_accuracy": val_acc})
    print(f"Validation accuracy: {val_acc}")

    holdout_pred = model.predict(dholdout)
    holdout_acc = accuracy(holdout_pred, dholdout)
    wandb.log({"holdout_accuracy": holdout_acc})
    print(f"Holdout accuracy: {holdout_acc}")


sweep_config = {
    "method": "random", # try grid or random
    "metric": {
      "name": "val_accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "max_depth": {
            "values": [3, 6, 9, 12, 15]
        },
        "lambda": {
            "values": [0.1, 1, 2]
        },
        "learning_rate": {
            "values": [0.1, 0.05, 0.2]
        },
        "subsample": {
            "values": [1, 0.5, 0.3]
        },
        "colsample_bytree": {
            "values": [0.9, 0.8, 0.6, 0.4]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="sae-cls-gpt-XGBoost")

wandb.agent(sweep_id, train, count=35)