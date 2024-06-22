import torch
import xgboost as xgb
import numpy as np

train = torch.load('cruft/datasets/emb-train.pt')
test = torch.load('cruft/datasets/emb-test.pt')

X = train[:, :-1].numpy()
y = train[:, -1].numpy()

dtrain = xgb.DMatrix(X, label=y)

X_test = test[:, :-1].numpy()
y_test = test[:, -1].numpy()

dtest = xgb.DMatrix(X_test, label=y_test)


def eval_accuracy(preds, ds):
    labels = ds.get_label()
    preds = preds > 0.5

    accuracy = np.sum(preds == labels) / len(labels)
    return 'accuracy', accuracy

# Set parameters
params = {
    'max_depth': 6,  # Increase depth
    'eta': 0.001,  # Decrease learning rate
    'objective': 'binary:logistic',
    'eval_metric': ['logloss'],  # Include both logloss and error
    'subsample': 0.8,  # Subsample ratio of the training instances
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'alpha': 0.1,  # L1 regularization term
    'lambda': 1.0,  # L2 regularization term
    'tree_method': 'gpu_hist',
    'device': 'cuda'
}
num_round = 5000  # Increase number of boosting rounds

# Create watchlist
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# Train the model with evaluation on the training and test sets, and early stopping
bst = xgb.train(params, dtrain, num_round, watchlist, custom_metric=eval_accuracy)