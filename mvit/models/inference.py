from mvit_model import MViT 
from mvit.config.defaults import get_cfg 

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a given model.

    Args:
        model: The PyTorch model for which to count parameters.

    Returns:
        The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


config = get_cfg()
print("config",config)
model = MVIT(config)
print(count_parameters(model))
