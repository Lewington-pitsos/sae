import torch
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value


from app.constants import *

def ingest_ade_raft():
    raft_ade = load_dataset("ought/raft", 'ade_corpus_v2', cache_dir="cache")

    ade = load_dataset('ade-benchmark-corpus/ade_corpus_v2', 'Ade_corpus_v2_classification')


    ade_dict = {row['text']: row['label'] for row in ade['train']}

    label_mapping ={
        0: 2,
        1: 1,
    }

    # Function to add a new column 'label2'
    def add_matching_label(example):
        text = example['Sentence']
        # Check if the text matches any text in the ade_dict
        example['label'] = label_mapping[ade_dict.get(text, None)]  # Adds None if no match found
        

        return example

    # Apply the function to both train and test sets of raft_subset
    raft_subset = raft_ade.map(add_matching_label)

    # Example: print the updated raft_subset
    for sample in raft_subset['train']:
        assert sample['label'] is not None
        assert sample['label'] == sample['Label']

    for sample in raft_subset['test']:
        assert sample['label'] is not None


    label_remapping = {2: 0, 1: 1}


    def reformat(example):
        example['text'] = example['Sentence']
        example['label'] = label_remapping[example['label']]

        del example['Sentence']
        del example['ID']
        del example['Label']

        return example
    
    raft_subset = raft_subset.map(reformat)

    # Assuming each item in raft_subset['train'] and raft_subset['test'] has 'data' and 'label'
    train_dataset = Dataset.from_dict({'text': [item['text'] for item in raft_subset['train']],
                                    'label': [item['label'] for item in raft_subset['train']]})
    test_dataset = Dataset.from_dict({'text': [item['text'] for item in raft_subset['test']],
                                    'label': [item['label'] for item in raft_subset['test']]})


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
    dataset_dict.save_to_disk(F'{LOCAL_DATA_PATH}/ade_raft_labelled')

ingest_ade_raft()