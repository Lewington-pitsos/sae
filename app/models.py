import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2ForSequenceClassification, PretrainedConfig, AutoConfig, GPT2Tokenizer, GPT2LMHeadModel
from sae_lens import SAE
from transformer_lens import HookedTransformer

from app.constants import *

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, model_size:str, max_seq_len: int, num_classes: int = 2):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(model_size)
        final_hidden_size = self.gpt2model.config.n_embd

        for param in self.gpt2model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(final_hidden_size * max_seq_len, num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output


class BigHeadGPT2SequenceClassifier(nn.Module):
    def __init__(self,
            model_size: str, 
            max_seq_len: int, 
            num_classes: int = 2, 
            activation: str = 'relu',
            head_hidden_size: int = 8192
        ):
        super(BigHeadGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(model_size)
        hidden_size = self.gpt2model.config.n_embd

        for param in self.gpt2model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(hidden_size, head_hidden_size)

        self.activation = self._get_activation(activation)

        self.fc2 = nn.Linear(head_hidden_size * max_seq_len, num_classes, bias=False)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = self.fc1(gpt_out)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)

        return x

def inject_phrase(input_ids, phrase, pad_token, attention_mask, device):
    phrase_length = phrase.size(0)
    seq_len = input_ids.size(1)

    new_seq_length = seq_len + phrase_length
    new_input_ids = torch.full((input_ids.size(0), new_seq_length), pad_token, dtype=input_ids.dtype).to(device)
    new_input_ids[:, :seq_len] = input_ids

    batch_padding_counts = torch.sum(attention_mask, dim=1)

    stop_indices = batch_padding_counts + phrase_length

    range_tensor = torch.arange(new_input_ids.size(1)).expand(new_input_ids.size(0), -1).to(device)

    mask = (range_tensor >= batch_padding_counts.unsqueeze(1)) & (range_tensor < stop_indices.unsqueeze(1))

    new_input_ids[mask] = torch.concat([phrase] * new_input_ids.size(0)).to(device).to(torch.int64)

    new_batch_padding_counts = batch_padding_counts + phrase_length
    new_attention_mask = torch.full((attention_mask.size(0), new_seq_length), 0, dtype=attention_mask.dtype).to(device)

    mask = (range_tensor < new_batch_padding_counts.unsqueeze(1))
    new_attention_mask[mask] = 1

    return new_input_ids, new_attention_mask

class GPTProbabilityClassifier(nn.Module):
    def __init__(self, model_size, phrase, labels, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.labels = labels

        self.model = GPT2LMHeadModel.from_pretrained(model_size)
        self.classification_phrase = phrase

        tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.eos_token_id
        self.tokenized_labels = []

        for l in self.labels:
            input_ids = tokenizer(l, return_tensors='pt', padding=True, truncation=True)['input_ids'].view(-1)
            input_ids.required_grad=False
            self.tokenized_labels.append(input_ids)

        self.tokenized_classification_phrase = tokenizer(phrase, return_tensors='pt', padding=True, truncation=True)['input_ids'][0]
        self.classification_phrase_length = self.tokenized_classification_phrase.shape[0]
        self.device = device

        self.softmax = nn.Softmax(dim=1)    

        # freeze all parameters

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        probabilities = []

        for l in self.tokenized_labels:
            class_input_ids, class_attention_mask = inject_phrase(input_ids, torch.concat([self.tokenized_classification_phrase, l]), self.pad_token_id, attention_mask, self.device)
            outputs = self.model(input_ids=class_input_ids, attention_mask=class_attention_mask)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

            shifted_input_ids = class_input_ids[:, 1:]
            shifted_log_probs = log_probs[:, :-1, :]

            log_likelihood = torch.gather(shifted_log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

            total_log_likelihood = log_likelihood.mean(dim=1)
            probabilities.append(total_log_likelihood)


        probabilities = torch.stack(probabilities, dim=1)

        return probabilities


class RandomClassifier(torch.nn.Module):
    def __init__(self, device):
        super(RandomClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=False) # only here so we don't get error when no gradients are being computed
        self.device = device
        self.fc1.to(device)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        random_predictions = (torch.rand(1, batch_size) < 0.5).long()

        one_hot_tensor = torch.zeros(batch_size, 2)
        one_hot_tensor[torch.arange(batch_size), random_predictions[0]] = 1

        one_hot_tensor = one_hot_tensor.to(self.device)

        x = self.fc1(one_hot_tensor)
        return x

class SAEBaseModel(nn.Module):
    def __init__(self, 
                 transformer_name: str, 
                 sae_release: str,
                 sae_id:str,
                 device, 
                 max_seq_len:int = None, 
                 num_classes: int = 2,
                 freeze_sae: bool = True
        ):
        super(SAEBaseModel, self).__init__()

        sae, _, _ = SAE.from_pretrained(release = sae_release, sae_id = sae_id, device = device)

        self.sae = sae

        if freeze_sae:
            for param in self.sae.parameters():
                param.requires_grad = False

        seq_len = int(max_seq_len / 8)

        self.fc1 = nn.Linear(sae.cfg.d_sae * seq_len, num_classes, bias=False)
        self.device = device

        self.model = HookedTransformer.from_pretrained(transformer_name, device=device)
        self.hook_layer = sae.cfg.hook_layer
        self.hook_name = sae.cfg.hook_name

        for param in self.model.parameters():
            param.requires_grad = False


        self.dropout = nn.Dropout(0.3)
        self.avg_pool = nn.AvgPool1d(kernel_size=8, stride=8)


class SAEFeaturesModel(SAEBaseModel):
    def forward(self, input_ids, attention_mask):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True, stop_at_layer=self.hook_layer + 1)

        hidden_states = cache[self.hook_name]

        features = self.sae.encode(hidden_states)
        return features

class SAEClassifier(SAEBaseModel):
    def forward(self, input_ids, attention_mask):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True, stop_at_layer=self.hook_layer + 1)

        hidden_states = cache[self.hook_name]

        features = self.sae.encode(hidden_states)

        features = features.transpose(1, 2)

        features = self.avg_pool(features)

        features = features.transpose(1, 2)

        features = features.view(features.shape[0], -1)
        features = self.dropout(features)

        x = self.fc1(features)
        return x

class GPT2Classifier(nn.Module):
    def __init__(self, model_size):
        super(GPT2Classifier, self).__init__()

        config = AutoConfig.from_pretrained(model_size)
        config.num_labels = 2
        config.pad_token_id = config.eos_token_id
        config.problem_type = "single_label_classification"
        self.model = GPT2ForSequenceClassification.from_pretrained(model_size, config=config)

        for param in self.model.transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def get_sae_model_config(model_name):
    if model_name == 'sae-classifier-gpt':
        return {
            'transformer_name': 'gpt2',
            'sae_release': 'gpt2-small-res-jb',
            'sae_id': 'blocks.8.hook_resid_pre',
        }
    elif model_name == 'sae-classifier-mistral7b':
        return {
            'transformer_name': 'mistralai/Mistral-7B-v0.1',
            'sae_release': 'mistral-7b-res-wg',
            'sae_id': 'blocks.8.hook_resid_pre',
        }
    
    raise ValueError(f"Invalid model name: {model_name}")

def get_probability_model_config(dataset_name):
    if dataset_name == 'imdb': 
        return {
            'labels': ['negative', 'positive'],
            'phrase': ''
        }
    elif dataset_name == 'data/raft_ade_corpus_v2':
        return {
            'labels': ['no adverse drug event', 'adverse drug event',],
            'phrase': '\n\n the preceding sentence would be classified (adverse drug event|no adverse drug event) as '
        }
    elif dataset_name == 'data/raft_overruling':
        return {
            'labels': ['not overruling', 'overruling'],
            'phrase': '\n\n the preceding legal text would be considered (overruling|not overruling) an '
        }
    elif dataset_name == 'data/raft_tweet_eval_hate':
        return {
            'labels': ['not hateful', 'hateful'],
            'phrase': '\n\n classification:  '
        }

    raise ValueError(f"Invalid dataset name: {dataset_name}")

def build_model(model_type, model_size, dataset_name, max_seq_len, freeze, activation, device):
    if model_type == 'simple':
        return SimpleGPT2SequenceClassifier(model_size=model_size, max_seq_len=max_seq_len)
    if model_type == 'big-head':
        return BigHeadGPT2SequenceClassifier(model_size=model_size, max_seq_len=max_seq_len, activation=activation)
    elif model_type == 'probability':
        return GPTProbabilityClassifier(model_size=model_size, **get_probability_model_config(dataset_name), device=device)
    elif model_type == 'sae-classifier-gpt':
        return SAEClassifier(device=device, max_seq_len=max_seq_len, freeze_sae=freeze, **get_sae_model_config(model_type))
    elif model_type == 'sae-classifier-mistral7b':
        return SAEClassifier(device=device, max_seq_len=max_seq_len, freeze_sae=freeze, **get_sae_model_config(model_type))
    elif model_type == 'random':
        return RandomClassifier(device)
    elif model_type == 'gpt2-classifier': 
        return GPT2Classifier(model_size=model_size).to(device)
        
    raise ValueError(f"Invalid model type: {model_type}")

def masked_avg(embedding_matrix, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    
    sum_embedding = (embedding_matrix * attention_mask_expanded).sum(dim=1)
    non_masked_count = attention_mask.sum(dim=1, keepdim=True)
    
    non_masked_count = non_masked_count.clamp(min=1)
    
    average_embedding = sum_embedding / non_masked_count

    return average_embedding
