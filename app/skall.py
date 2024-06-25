import json
import torch
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, f1_score

from app.constants import *

results = {}

for dataset_name in RAFT_DATASETS:
    results[dataset_name] = {}
    train_filename = f'{LOCAL_DATA_PATH}/avg-emb-256-sae-classifier-gpt2-train-{dataset_name}.pt'
    test_filename = f'{LOCAL_DATA_PATH}/avg-emb-256-sae-classifier-gpt2-test-{dataset_name}.pt'

    train = torch.load(train_filename)
    test = torch.load(test_filename)

    X_train = train[:, :-1].numpy()
    y_train = train[:, -1].numpy()

    X_test = test[:, :-1].numpy()
    y_test = test[:, -1].numpy()

    print(f'{dataset_name}: {X_train.shape}, {X_test.shape}')

    classifiers = all_estimators(type_filter='classifier')

    for name, ClassifierClass in classifiers:
        if name in ['HistGradientBoostingClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier']:
            clf = ClassifierClass()
            clf.fit(X_train, y_train)
    
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f'{name}: acc:{accuracy}, f1:{f1}')

            results[dataset_name][name] = {
                'acc': accuracy,
                'f1': f1
            }

            with open('cruft/results-macrof1.json', 'w') as f:
                json.dump(results, f)
