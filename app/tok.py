import transformers

def load_tokenizer(model_name):
    if model_name in ['sae-classifier-gpt2', 'big-head', 'simple', 'random', 'gpt2-classifier']:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name in ['sae-classifier-mistral7b']:
        tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    raise ValueError(f"Invalid model_name: {model_name}")