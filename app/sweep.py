import xgboost as xgb
import wandb

from app.constants import *
from app.boost import *
    
def train():
    config_defaults = {
        'booster': 'gbtree',  # Use tree based models
        'max_depth': 5,  # Increase depth
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],  # Include both logloss and error
        'subsample': 0.85,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'alpha': 0.1,  # L1 regularization term
        'lambda': 1.0,  # L2 regularization term
        'device': 'cuda',
        'min_child_weight': 1.0,  # Minimum sum of instance weight needed in a child
        "learning_rate": 0.05,
    }

    wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep
    config = wandb.config

    dtrain, dval, dholdout = load_sae_feature_dataset()

    watchlist = [(dtrain, 'train'), (dval, 'val'), (dholdout, 'holdout')]

    run_config = {}
    for k, v in config.items():
        run_config[k] = config[k]

    print(run_config)

    model = xgb.train(run_config, dtrain, 5000, watchlist, custom_metric=f1_metric, early_stopping_rounds=200, maximize=True, callbacks=[LogEvaluation(5)])

    val_pred = model.predict(dval)
    val_acc = f1(val_pred, dval)
    wandb.log({"final-val-accuracy": val_acc})
    print(f"Validation accuracy: {val_acc}")

    holdout_pred = model.predict(dholdout)
    holdout_acc = f1(holdout_pred, dholdout)
    wandb.log({"final-holdout-accuracy": holdout_acc})
    print(f"Holdout accuracy: {holdout_acc}")


sweep_config = {
    "method": "random",
    "metric": {
      "name": "final-val-accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "min_child_weight": {
            "values": [0.2, 0.7, 1.0, 1.5, 3], 
        },
        "learning_rate": {
            "values": [0.03, 0.7]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="sae-cls-gpt-XGBoost-2")

wandb.agent(sweep_id, train, count=15)