import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2ForSequenceClassification, PretrainedConfig, AutoConfig
from sae_lens import SAE
from transformer_lens import HookedTransformer

from app.constants import *

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, gpt_model_name: str, num_classes: int = 2, freeze=False):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        if freeze:
            for param in self.gpt2model.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output

# def classify_sentiment(model, batch):
#     with torch.no_grad():
#         outputs = model(input_ids=batch)
#     logits = outputs.logits[:, -1, :]
#     probabilities = softmax(logits, dim=-1)
#     probabilities = probabilities[:, TOKENIZED_LABELS]
#     predicted_idxs = torch.argmax(probabilities, dim=-1)
#     predicted_sentiments = [LABELS[idx.item()] for idx in predicted_idxs]
#     return predicted_sentiments

# def classify_sentiment_generate(model, batch):
#     outputs = model.generate(input_ids=batch, max_length=batch.shape[1] + 1, num_return_sequences=1)
#     decoded_outputs = [tokenizer.decode(output) for output in outputs]
#     predicted_sentiments = []
#     for decoded_output in decoded_outputs:
#         last_word = decoded_output.split()[-1].lower()
#         if last_word in LABELS:
#             predicted_sentiments.append(last_word)
#         else:
#             predicted_sentiments.append(classify_sentiment_random(model, [decoded_output])[0])
#     return predicted_sentiments

class RandomClassifier(torch.nn.Module):
    def __init__(self):
        super(RandomClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        random_predictions = (torch.rand(1, batch_size) < 0.5).long()

        one_hot_tensor = torch.zeros(batch_size, 2)
        one_hot_tensor[torch.arange(batch_size), random_predictions[0]] = 1

        x = self.fc1(one_hot_tensor)
        return x

def save_tensor_activations(tensor, file_prefix='activation'):
    import matplotlib.pyplot as plt
    # Check the shape of the tensor
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError("Tensor must be of shape [2, 50, 768]")
    
    for i in range(tensor.shape[0]):
        activation = tensor[i].detach().numpy()  # Convert to numpy array
        plt.imshow(activation, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Activation {i+1}')
        plt.xlabel('Features')
        plt.ylabel('Activation Index')
        plt.savefig(f'{file_prefix}_{i+1}.png')
        plt.close()

class BigHeadGPT2SequenceClassifier(nn.Module):
    def __init__(self, 
            hidden_size: int, 
            max_seq_len: int, 
            gpt_model_name: str, 
            num_classes: int = 2, 
            freeze=False, 
        ):
        super(BigHeadGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        if freeze:
            for param in self.gpt2model.parameters():
                param.requires_grad = False
        head_hidden_size = 8192
        self.fc1 = nn.Linear(hidden_size, head_hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(head_hidden_size * max_seq_len, num_classes, bias=False)

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt2model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        x = self.fc1(gpt_out)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)

        return x


class SAEBaseModel(nn.Module):
    def __init__(self, 
                 transformer_name: str, 
                 sae_release: str,
                 sae_id:str,
                 device, 
                 max_seq_len:int = None, 
                 num_classes: int = 2,
        ):
        super(SAEBaseModel, self).__init__()

        sae, _, _ = SAE.from_pretrained(release = sae_release, sae_id = sae_id, device = device)

        self.sae = sae

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
    def __init__(self, gpt_model_name):
        super(GPT2Classifier, self).__init__()

        config = AutoConfig.from_pretrained(gpt_model_name)
        config.num_labels = 2
        config.pad_token_id = config.eos_token_id
        config.problem_type = "single_label_classification"
        self.model = GPT2ForSequenceClassification.from_pretrained(gpt_model_name, config=config)

        for param in self.model.transformer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def get_sae_model_config(model_name):
    if model_name == 'sae-classifier-gpt2':
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

def build_model(model_name, hidden_size, max_seq_len, freeze, device):
    if model_name == 'simple':
        return SimpleGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=max_seq_len, gpt_model_name='gpt2', freeze=freeze)
    elif model_name == 'big-head':
        return BigHeadGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=max_seq_len, gpt_model_name='gpt2', freeze=freeze)
    elif model_name == 'sae-classifier-gpt2':
        return SAEClassifier(device=device, max_seq_len=max_seq_len, **get_sae_model_config(model_name))
    elif model_name == 'sae-classifier-mistral7b':
        return SAEClassifier(device=device, max_seq_len=max_seq_len, **get_sae_model_config(model_name))
    elif model_name == 'random':
        return RandomClassifier()
    elif model_name == 'gpt2-classifier': 
        return GPT2Classifier('gpt2').to(device)
        
    raise ValueError(f"Invalid model name: {model_name}")


def masked_avg(embedding_matrix, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    
    sum_embedding = (embedding_matrix * attention_mask_expanded).sum(dim=1)
    non_masked_count = attention_mask.sum(dim=1, keepdim=True)
    
    non_masked_count = non_masked_count.clamp(min=1)
    
    average_embedding = sum_embedding / non_masked_count

    return average_embedding
