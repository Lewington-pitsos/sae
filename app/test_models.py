from app.models import RandomClassifier, GPT2Classifier, SimpleGPT2SequenceClassifier
from transformers import GPT2Tokenizer
import torch

def test_random_model():
    random_model = RandomClassifier('cpu')
    input_ids = torch.randint(0, 50256, (4, 50))
    attention_mask = torch.ones_like(input_ids)
    random_predictions = random_model(input_ids, attention_mask)

    assert random_predictions.shape == (4, 2)

def test_official_classifier():
    gpt_model_name = 'gpt2' 
    max_seq_len = 128


    phrases = [
        "I hated this movie",
        "I loved this movie, the reason being that it was so good, you see I loved it, and furthermore, I loved it!"
    ]


    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(phrases, return_tensors='pt', padding='max_length', truncation=True, max_length=max_seq_len)['input_ids']

    print(input_ids)

    model = GPT2Classifier(gpt_model_name)

    attention_mask = torch.ones_like(input_ids)

    predictions = model(input_ids=input_ids, attention_mask=attention_mask)

    print(predictions)

    assert predictions.shape == (2, 2)