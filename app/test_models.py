from app.models import RandomClassifier
import torch

def test_random_model():
    random_model = RandomClassifier()
    input_ids = torch.randint(0, 50256, (4, 50))
    attention_mask = torch.ones_like(input_ids)
    random_predictions = random_model(input_ids, attention_mask)
    
    assert random_predictions.sum() == 4
    assert random_predictions.shape == (4, 2)