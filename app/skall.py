import json
import torch
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, f1_score

from app.constants import *
from app.embed import get_dataset_name

results = {}

for dataset_name in RAFT_DATASETS:
    results[dataset_name] = {}

    for embedding_name in ['sae_ft', 'hs', 'hs_final']:
        results[dataset_name][embedding_name] = {}
    
        train_filename = get_dataset_name(dataset_name, embedding_name, 'train')
        test_filename = get_dataset_name(dataset_name, embedding_name, 'test')
    

        train = torch.load(train_filename)
        test = torch.load(test_filename)

        X_train = train[:, :-1].numpy()
        y_train = train[:, -1].numpy()

        X_test = test[:, :-1].numpy()
        y_test = test[:, -1].numpy()

        print(f'{dataset_name}: {X_train.shape}, {X_test.shape}')

        classifiers = all_estimators(type_filter='classifier')

        for name, ClassifierClass in classifiers:
            if name in ['Perceptron', 'HistGradientBoostingClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier']:
                clf = ClassifierClass()
                clf.fit(X_train, y_train)
        
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                print(f'{name}: acc:{accuracy}, f1:{f1}')

                results[dataset_name][embedding_name][name] = {
                    'acc': accuracy,
                    'f1': f1
                }

                with open('cruft/results-sklearn.json', 'w') as f:
                    json.dump(results, f)
