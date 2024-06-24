from datasets import load_dataset, get_dataset_config_names
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value

from app.constants import *


todo = {
    # 'neurips_impact_statement_risks': None,
    'tweet_eval_hate': 'cardiffnlp/tweet_eval', # 'hate'
    # 'overruling',
    # 'semiconductor_org_types',
    # 'tai_safety_research',
    # 'terms_of_service',
    # 'twitter_complaints',
}

def validate_new_dataset(ds, expected_test_len):
    label_counts = {0: 0, 1: 0, 2: 0}
    for sample in ds['train']:
        assert sample['label'] is not None
        assert sample['label'] == sample['Label']
        label_counts[sample['label']] += 1

    for label, count in label_counts.items():
        print(f"train {label}: {count}, {count / len(ds['train'])}")

    for sample in ds['test']:
        assert sample['label'] is not None

    for label, count in label_counts.items():
        print(f"test {label}: {count}, {count / len(ds['train'])}")

    assert len(ds['train']) == 50, f"train size {len(ds['train'])}"
    assert len(ds['test']) == expected_test_len, f"test size {len(ds['test'])}"

def reformat(example, text_name):
    label_remapping = {2: 0, 1: 1}
    example['text'] = example[text_name]
    example['label'] = label_remapping[example['label']]

    del example[text_name]
    del example['ID']
    del example['Label']

    return example

def save_dataset(raft, name):
    dataset_name = f'{LOCAL_DATA_PATH}/raft_{name}'

    print('saving to:', dataset_name)

    train_dataset = Dataset.from_dict({'text': [item['text'] for item in raft['train']],
                                    'label': [item['label'] for item in raft['train']]})
    test_dataset = Dataset.from_dict({'text': [item['text'] for item in raft['test']],
                                    'label': [item['label'] for item in raft['test']]})


    # Optionally specify features if your dataset needs specific types
    # For example, if your dataset has a label field which is a class label
    features = Features({
        'text': Value('string'),  # Adjust this according to the actual data type
        'label': ClassLabel(names=['negative', 'positive'])
    })

    train_dataset = train_dataset.cast(features)
    test_dataset = test_dataset.cast(features)

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Save to disk
    dataset_dict.save_to_disk(dataset_name)

def ingest_raft(raft_subset_name, raft_text_name, dataset, dataset_subset):

    label_data = load_dataset(dataset, dataset_subset)
    raft = load_dataset("ought/raft", raft_subset_name, cache_dir="cache")
    
    expected_test_len = len(raft['test'])

    text_dict = dict()
    mis_labelled = []
    duplicates = []
    for split in [k for k in label_data.keys() if k in ['train', 'test']]:
        print(split)
        for row in label_data[split]:
            if row['text'] in text_dict:
                if text_dict[row['text']] != row['label']:
                    print(f"Inconsistent label on: {row['text']}, {row['label']}")

                    mis_labelled.append(row['text'])
                duplicates.append(row['text'])
            text_dict[row['text']] = row['label']


    print(f"Found {len(duplicates)} duplicates")
    for dupe in list(set(duplicates))[:10]:
        print('duplicate: ', dupe)

    print(f"Found {len(mis_labelled)} duplicates with mismatched labels")

    label_mapping ={
        0: 2,
        1: 1,
    }

    def add_matching_label(example):
        text = example[raft_text_name]
        
        if text in mis_labelled and example['Label'] != 0:
            example['label'] = example['Label']
        else:
            example['label'] = label_mapping[text_dict[text]] 
        return example

    raft = raft.map(add_matching_label)

    validate_new_dataset(raft, expected_test_len)
    raft = raft.map(lambda x: reformat(x, raft_text_name))

    save_dataset(raft, raft_subset_name)

ingest_raft('tweet_eval_hate', 'Tweet', 'cardiffnlp/tweet_eval', 'hate')
ingest_raft('ade_corpus_v2', 'Sentence', 'ade-benchmark-corpus/ade_corpus_v2', 'Ade_corpus_v2_classification')