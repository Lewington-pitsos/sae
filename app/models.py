import torch.nn as nn
from transformers import GPT2Model
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

    def forward(self, activations, input_id, mask):
        gpt_out = self.gpt2model(input_ids=input_id, attention_mask=mask).last_hidden_state
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output

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
    def __init__(self, num_classes: int = 2, device='cuda'):
        super(SAEClassifier, self).__init__()

        sae, _, _ = SAE.from_pretrained(
            release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
            sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
            device = device
        )

        self.sae = sae

        for param in self.sae.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(sae.cfg.d_sae, num_classes)
        self.device = device

    def forward(self, activations, input_id, mask):
        features = self.sae.decode(activations)

        x = self.fc1(features)
        return x