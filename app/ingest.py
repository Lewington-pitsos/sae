import torch
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names

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

    for split in ['train', 'test']:
        for example in tqdm(raft_subset[split]):
            example['text'] = example['Sentence']
            example['label'] = label_remapping[example['label']]

            del example['Sentence']
            del example['ID']
            del example['Label']

    dataset_dict = {'train': list(raft_subset['train']), 'test': list(raft_subset['test'])}     

    torch.save(dataset_dict, F'{LOCAL_DATA_PATH}/ade_raft_labelled.pt') 