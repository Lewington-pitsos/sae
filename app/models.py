import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2ForSequenceClassification, GPT2Tokenizer
from sae_lens import SAE
from transformer_lens import HookedTransformer


class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, gpt_model_name: str, num_classes: int = 2, freeze=False):
        super(SimpleGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        if freeze:
            for param in self.gpt2model.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        gpt_out = self.gpt2model(input_ids=input_id, attention_mask=mask).last_hidden_state
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

class BigHeadGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, gpt_model_name: str, num_classes: int = 2, freeze=False):
        super(BigHeadGPT2SequenceClassifier, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        if freeze:
            for param in self.gpt2model.parameters():
                param.requires_grad = False
        head_hidden_size = 8192
        self.fc1 = nn.Linear(hidden_size, head_hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(head_hidden_size * max_seq_len, num_classes)

    def forward(self, input_id, mask):
        gpt_out = self.gpt2model(input_ids=input_id, attention_mask=mask).last_hidden_state
        x = self.fc1(gpt_out)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)

        return x

class ActivationModel(nn.Module):
    def __init__(self, gpt_model_name: str, hook_name: str, hook_layer: str, device):
        super(ActivationModel, self).__init__()
        
        self.model = HookedTransformer.from_pretrained(gpt_model_name, device=device)
        self.hook_layer = hook_layer
        self.hook_name = hook_name

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True, stop_at_layer=self.hook_layer + 1)

        return cache[self.hook_name]

class SAEClassifier(nn.Module):
    def __init__(self, gpt_model_name: str, hook_name: str, hook_layer: str, device, max_seq_len:int, num_classes: int = 2):
        super(SAEClassifier, self).__init__()

        sae, _, _ = SAE.from_pretrained(
            release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
            sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
            device = device
        )

        self.sae = sae

        for param in self.sae.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(sae.cfg.d_sae * max_seq_len, num_classes)
        self.device = device

        self.model = HookedTransformer.from_pretrained(gpt_model_name, device=device)
        self.hook_layer = hook_layer
        self.hook_name = hook_name

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True, stop_at_layer=self.hook_layer + 1)

        activations = cache[self.hook_name]

        features = self.sae.encode(activations)

        features = features.view(features.shape[0], -1)

        x = self.fc1(features)
        return x

def build_model(model_name, hidden_size, max_seq_len, gpt_model_name, freeze, device):
    if model_name == 'simple':
        return SimpleGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=max_seq_len, gpt_model_name=gpt_model_name, freeze=freeze)
    elif model_name == 'big-head':
        return BigHeadGPT2SequenceClassifier(hidden_size=hidden_size, max_seq_len=max_seq_len, gpt_model_name=gpt_model_name, freeze=freeze)
    elif model_name == 'sae-classifier':
        return SAEClassifier(gpt_model_name=gpt_model_name, hook_name='blocks.8.hook_resid_pre', hook_layer=8, device=device, max_seq_len=max_seq_len)
    elif model_name == 'random':
        return RandomClassifier()

    raise ValueError(f"Invalid model name: {model_name}")