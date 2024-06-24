import json
import numpy as np
import torch
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import is_classifier

from app.constants import *

# 'avg-emb-gpt2-mistral-test-raft_ade_corpus_v2.pt',
# 'avg-emb-gpt2-mistral-train-raft_ade_corpus_v2.pt',

results = {}

for dataset_name in ['raft_tweet_eval_hate', 'raft_ade_corpus_v2']:
    results[dataset_name] = {}
    train_filename = f'{LOCAL_DATA_PATH}/avg-emb-gpt2-mistral-test-{dataset_name}.pt'
    test_filename = f'{LOCAL_DATA_PATH}/avg-emb-gpt2-mistral-train-{dataset_name}.pt'

    # Load your data
    train = torch.load(train_filename)
    test = torch.load(test_filename)

    # Prepare the training data
    X_train = train[:, :-1].numpy()
    y_train = train[:, -1].numpy()

    # Prepare the testing data
    X_test = test[:, :-1].numpy()
    y_test = test[:, -1].numpy()

    # Get all the classifiers from scikit-learn
    classifiers = all_estimators(type_filter='classifier')

    for name, ClassifierClass in classifiers:
        print(name)

    # Loop through all classifiers
    for name, ClassifierClass in classifiers:
        if name in [
            'LogisticRegression',
            'LogisticRegressionCV'
        ]:
            print('skipping', name)
            continue

        try:
            # Initialize the classifier
            clf = ClassifierClass()
            # Fit the classifier
            clf.fit(X_train, y_train)
            # Predict on the test set
            y_pred = clf.predict(X_test)
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f'{name}: acc:{accuracy}, f1:{f1}')

            results[dataset_name][name] = {
                'acc': accuracy,
                'f1': f1
            }

            with open('cruft/results.json', 'w') as f:
                json.dump(results, f)
        except Exception as e:
            print(f'{name} failed: {str(e)}')
