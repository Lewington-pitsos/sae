from datasets import load_dataset

from app.constants import *
from app.load import get_text_column

todo = {
    # 'neurips_impact_statement_risks': None, < ---------- no labels
    'tweet_eval_hate': 'cardiffnlp/tweet_eval', # 'hate'
    # 'overruling',
    # 'semiconductor_org_types', <--------- no labels
    # 'tai_safety_research', < ------- compiled by RAFT from public sources, big effort
    # 'terms_of_service', < ------ need to compile from claudette
    # 'twitter_complaints', < ---------
}

def levenshtein_distance(s1, s2):
    # Initialize the matrix
    d = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]

    # Fill the first row and first column
    for i in range(len(s1) + 1):
        d[i][0] = i
    for j in range(len(s2) + 1):
        d[0][j] = j

    # Compute the Levenshtein distance
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,    # Deletion
                          d[i][j - 1] + 1,    # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution

    return d[-1][-1]

def string_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = (max_len - distance) / max_len
    return similarity


def validate_new_dataset(ds, expected_test_len):
    for split in ['train', 'test']:
        all_labels = set()
        for row in ds[split]:
            all_labels.add(row['Label'])
        print(f'number of labels in {split}:', len(all_labels))
    
    assert len(ds['train']) == 50, f"train size {len(ds['train'])}"
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
    def add_matching_label(example):
        text = example[raft_text_name]
        
        try:
            label_for_text = text_dict[text]
        except KeyError:
            # strip whitespace from beginning of text and try again
            try: 
                label_for_text = text_dict[" " + text]
            except KeyError:
                label_for_text = text_dict["  " + text]
        
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

    labelled_dataset.save_to_disk(f'cruft/raft_{raft_subset_name}')


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
    ingest_and_save_raft('ade_corpus_v2')
    ingest_and_save_raft('overruling')
    ingest_and_save_raft('banking_77')    