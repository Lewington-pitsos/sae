import torch.nn as nn
import wandb

def model_parameters_info(model: nn.Module):
    """
    Prints the number of trainable and non-trainable parameters in the model
    as well as the ratio of trainable to non-trainable parameters.
    
    Args:
    model (nn.Module): The model to analyze.
    
    Returns:
    None
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    ratio = trainable_params / non_trainable_params if non_trainable_params != 0 else float('inf')

    print(f"Total parameters: {total_params:,} (roughly {total_params:.2e})")
    print(f"Trainable parameters: {trainable_params:,} (roughly {trainable_params:.2e})")
    print(f"Non-trainable parameters: {non_trainable_params:,} (roughly {non_trainable_params:.2e})")
    print(f"Ratio (Trainable/Non-trainable): {ratio:,} (roughly {ratio:.2e})")

    # log to wandb too 

    wandb.log({"total_parameters": total_params, "trainable_parameters": trainable_params, "non_trainable_parameters": non_trainable_params, "ratio": ratio})
