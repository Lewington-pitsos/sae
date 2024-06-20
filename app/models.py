import torch.nn as nn
from transformers import GPT2Model, GPT2ForSequenceClassification
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

    raise ValueError(f"Invalid model name: {model_name}")