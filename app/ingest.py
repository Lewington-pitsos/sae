from datasets import load_dataset

from app.constants import *
from app.load import get_text_column

def validate_new_dataset(ds, expected_test_len):
    for split in ['train', 'test']:
        all_labels = set()
        for row in ds[split]:
            all_labels.add(row['Label'])
        print(f'number of labels in {split}:', len(all_labels))
    
    # 48 when we remove the 2 ambiguously labelled tweet_eval_hate examples
    assert len(ds['train']) in [50, 48], f"train size {len(ds['train'])}"
    assert len(ds['test']) == expected_test_len, f"test size {len(ds['test'])}"

def ingest_raft(raft_subset_name):
    params = RAFT_PARAMS[raft_subset_name]
    dataset = params['dataset']
    dataset_subset = params['dataset_subset']
    label_mapping = params['label_mapping']
    text_key = params['text_key']

    label_data = load_dataset(dataset, dataset_subset)

    print(label_data['train'].features)

    text_dict = dict()
    mis_labelled = []
    duplicates = []

    if label_mapping is None:
        label_mapping = dict()
        for label in label_data['train'].features['label'].names:
            label_mapping[label] = label

    for split in [k for k in label_data.keys() if k in ['train', 'test']]:
        print(split)
        for row in label_data[split]:
            
            if isinstance(list(label_mapping.keys())[0], int):
                label = row['label']
            else:
                label = label_data['train'].features['label'].int2str(row['label'])

            if row[text_key] in text_dict:
                if text_dict[row[text_key]] != label:
                    
                    mis_labelled.append(row[text_key])
                duplicates.append(row[text_key])
            text_dict[row[text_key]] = label

    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicates")

    if len(mis_labelled) > 0:
        print(f"Warning, Found {len(mis_labelled)} duplicates with mismatched labels")
        for text in mis_labelled[:5]:
            print(text)

    raft = load_dataset("ought/raft", raft_subset_name, cache_dir="cache")
    raft_text_name = get_text_column(raft)
    if len(mis_labelled) > 0:
        print('filtering out mislabelled instances')
        raft = raft.filter(lambda x: x[raft_text_name] not in mis_labelled)

    def add_matching_label(example):
        text = example[raft_text_name]
        
        try:
            label_for_text = text_dict[text]
        except KeyError:
            for variation in [" ", "  "]:
                if variation + text in text_dict:
                    label_for_text = text_dict[variation + text]
                    break
                else:
                    raise ValueError(f"could not find label for text: {text}")

        raft_label = label_mapping[label_for_text]
        
        new_raft_label = raft['train'].features['Label'].str2int(raft_label) - 1

        if new_raft_label <= -1:
            raise ValueError(f"invalid label: {new_raft_label}, {text}")

        if new_raft_label != example['Label'] and example['Label'] != 0:
            raise ValueError(f"mismatched labels: {new_raft_label}, {example['Label']}, {text}")

        example['Label'] = new_raft_label

        return example
    
    expected_test_len = len(raft['test'])

    def sub_one(example):
        example['Label'] = example['Label'] - 1

        if example['Label'] <= -1:
            raise ValueError(f"invalid label: {example['Label']}, {example['text']}")

        return example

    raft['test'] = raft['test'].map(add_matching_label)
    raft['train'] = raft['train'].map(sub_one)

    validate_new_dataset(raft, expected_test_len)

    return raft

def ingest_and_save_raft(raft_subset_name):
    labelled_dataset = ingest_raft(raft_subset_name)

    labelled_dataset.save_to_disk(f'data/raft_{raft_subset_name}')


RAFT_PARAMS = {
    'tweet_eval_hate': {'dataset': 'cardiffnlp/tweet_eval', 'dataset_subset': 'hate', 'label_mapping':{
        'non-hate': 'not hate speech', 'hate': 'hate speech'
    }, 'text_key': 'text'},
    'ade_corpus_v2': {'dataset': 'ade-benchmark-corpus/ade_corpus_v2', 'dataset_subset': 'Ade_corpus_v2_classification', 'label_mapping':{
        'Not-Related': 'not ADE-related', 'Related': 'ADE-related'
    }, 'text_key': 'text'},
    'overruling': {'dataset': 'LawInformedAI/overruling', 'dataset_subset': None, 'label_mapping':{
        0: 'not overruling', 1: 'overruling'
    }, 'text_key': 'sentence1'},
    'banking_77': {'dataset': 'legacy-datasets/banking77', 'dataset_subset': None, 'label_mapping': None, 'text_key': 'text'},
}

if __name__ == '__main__':
    ingest_and_save_raft('tweet_eval_hate')
    # for dataset_name in RAFT_DATASETS:
        # ingest_and_save_raft(dataset_name)